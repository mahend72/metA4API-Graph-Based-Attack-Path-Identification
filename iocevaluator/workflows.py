"""High-level workflows behind the CLI: build-tikg, evaluate, rank."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .attack_paths import PathConfig, evaluate_reference_paths, trace_candidate_paths
from .experiments import (ABLATION_VARIANTS, BASELINE_VARIANTS, DEPTH_VARIANTS, PER_STRUCTURE_VARIANTS, VARIANTS,
                          ExperimentConfig, run_cv, write_outputs, _type_ids)
from .io_loader import load_threats_json
from .labels import load_labels, load_reference_paths, load_severity
from .megits import megits_adjacency
from .models import GraphData, TrainConfig, fit, make_graph_data, predict_proba
from .prioritisation import eigenvector_centrality, explain_ioc, fused_risk, rank_adjacency, timed_centrality
from .tikg import TIKG, Entity, load_tikg, tikg_from_dict, tikg_from_threats
from .tikg_features import FeatureBuilder, FeatureConfig

VARIANT_GROUPS: Dict[str, List[str]] = {
    "main": ["metA4API"], "ablation": ABLATION_VARIANTS, "depth": DEPTH_VARIANTS, "baselines": BASELINE_VARIANTS,
    "per-structure": PER_STRUCTURE_VARIANTS,
    "all": sorted(set(ABLATION_VARIANTS + DEPTH_VARIANTS + BASELINE_VARIANTS)),
}


def build_tikg(threats_json: Path, out: Path, extra_json: Optional[Path] = None) -> TIKG:
    """OTX-normalised threats -> TIKG. `extra_json` (native TIKG JSON: entities + triplets) supplies everything OTX
    lacks (devices, platforms, attack types, T2-T10 relations, labels' campaigns, attrs)."""
    extra_e, extra_t = [], []
    if extra_json:
        d = json.loads(Path(extra_json).read_text(encoding="utf-8"))
        x = tikg_from_dict({"entities": d.get("entities", []), "triplets": []})
        extra_e = x.entities
        extra_t = [(t["s"], t["r"], t["t"]) if isinstance(t, dict) else tuple(t) for t in d.get("triplets", [])]
    g = tikg_from_threats(load_threats_json(threats_json), extra_triplets=extra_t, extra_entities=extra_e)
    g.save(out)
    return g


def evaluate(tikg_path: Path, labels_path: Path, out: Path, variants: Sequence[str], cfg: ExperimentConfig):
    g = load_tikg(tikg_path)
    ls = load_labels(g, labels_path)
    names: List[str] = []
    for v in variants:
        names += VARIANT_GROUPS.get(v, [v])
    names = list(dict.fromkeys(names))
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; valid: {sorted(VARIANTS)} or groups {sorted(VARIANT_GROUPS)}")
    res = run_cv(g, ls, names, cfg)
    return write_outputs(out, res, g, ls, cfg, names, data_files=[tikg_path, labels_path]), res


def rank(tikg_path: Path, labels_path: Path, out: Path, *, oof_probs: Optional[Path] = None,
         severity_path: Optional[Path] = None, alpha: float = 0.5, tau: Optional[float] = None,
         reference_paths: Optional[Path] = None, seed: int = 0, top: int = 20, path_cfg: PathConfig = PathConfig(),
         train_cfg: TrainConfig = TrainConfig(), feature_cfg: FeatureConfig = FeatureConfig()) -> Dict:
    """EC / fused-risk ranking, attack-path tracing and (if reference paths exist) path-level evaluation.

    Predictions come from `oof_probs` (out-of-fold, required for honest path evaluation) or, if omitted, from a
    model trained on all labelled nodes (suitable for operational ranking but NOT for evaluation)."""
    out.mkdir(parents=True, exist_ok=True)
    g = load_tikg(tikg_path)
    ls = load_labels(g, labels_path)
    if reference_paths:
        # Path EVALUATION is an experiment: at manuscript scale it must go through the frozen, integrity-enforced harness.
        from .protocol import refuse_at_scale
        refuse_at_scale("rank with --reference-paths (path evaluation)", n_nodes=g.n)
    res = megits_adjacency(g, keep_components=True)
    A = rank_adjacency(res.adj, tau)
    ec, ec_time = timed_centrality(A)
    sev, known = load_severity(g, severity_path) if severity_path else (None, None)
    risk = fused_risk(ec, sev, known, alpha)
    if oof_probs:
        probs = np.nan_to_num(np.load(oof_probs), nan=0.0)
    else:
        lab = np.flatnonzero(ls.labelled)
        rng = np.random.RandomState(seed)
        rng.shuffle(lab)
        nv = max(1, len(lab) // 10)
        x = FeatureBuilder(feature_cfg).fit_transform(g, lab)
        gd = make_graph_data(res.adj, _type_ids(g))
        probs = predict_proba(fit("gcn", x, gd, ls.Y, ls.valid, lab[nv:], lab[:nv], train_cfg, seed).model, x, gd)
    order = np.argsort(-risk)[:top]
    df = pd.DataFrame({"ioc": [g.entities[i].id for i in order], "type": [g.entities[i].type.value for i in order],
                       "eigenvector_centrality": ec[order], "risk": risk[order],
                       "severity_known": [bool(known[i]) if known is not None else False for i in order]})
    df.to_csv(out / "ranked_iocs.csv", index=False)
    expl = [explain_ioc(g, int(i), res.components, res.adj, ec, sev, known, risk) for i in order]
    (out / "explanations.json").write_text(json.dumps(expl, indent=2), encoding="utf-8")
    conf = probs.max(1)
    seeds = np.flatnonzero(conf >= path_cfg.conf_threshold)
    paths = trace_candidate_paths(g, seeds, risk, conf, res.adj, res.weights, path_cfg)[:top]
    (out / "ranked_paths.json").write_text(json.dumps([{
        "nodes": [g.entities[i].id for i in p.nodes], "relations": list(p.relations),
        "structures": [f"chi{k}" for k in p.structures], "score": p.score} for p in paths], indent=2), encoding="utf-8")
    summary = {"centrality_time_s": ec_time, "n_unknown_severity": None if known is None else int((~known).sum()),
               "predictions": "out-of-fold" if oof_probs else "in-sample (NOT valid for evaluation)"}
    if reference_paths:
        summary["path_evaluation"] = evaluate_reference_paths(g, load_reference_paths(reference_paths), probs, risk,
                                                              res.adj, res.weights, path_cfg)
    summary["reportability"] = {"reportable": False, "not_reportable": True,
                                "reason": "legacy ranking / path-evaluation route: does not expose search_budget_hit or the frozen exhaustive-search "
                                          "diagnostics (max_edges 6, confidence 0.5, 5000 candidates, 5,000,000 expansion ceiling); development only"}
    if "path_evaluation" in summary and isinstance(summary["path_evaluation"], dict):
        summary["path_evaluation"]["not_reportable"] = True
    (out / "rank_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return summary
