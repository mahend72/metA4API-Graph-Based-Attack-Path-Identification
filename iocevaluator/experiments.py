"""Cross-validated experiments (manuscript Sec. 4): main model, semantic ablation, GCN-depth, GAT/HGT baselines,
per-structure evaluation (Table 13), reduced-label robustness, cost measurements and reproducibility outputs.

Protocol: K-fold (default 10) campaign-isolated stratified CV x several seeds (default 5).  Everything that
depends on the training data - MeGiTS adjacency, feature statistics, model, early stopping - is recomputed per
outer fold from the outer-train partition only (see megits.fold_megits_adjacency and tikg_features).
"""
from __future__ import annotations

import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from .labels import LabelSpace
from .megits import fold_megits_adjacency
from .metagraphs import STRUCTURES, select_structures
from .metrics import auc_scores, mean_std, multilabel_prf, paired_wilcoxon, per_type_prf
from .models import GraphData, TrainConfig, fit, make_graph_data, predict_proba
from .prioritisation import rank_adjacency, timed_centrality
from .splits import Fold, group_stratified_folds
from .tikg import TIKG, EntityType
from .tikg_features import FeatureBuilder, FeatureConfig


@dataclass(frozen=True)
class Variant:
    name: str
    adjacency: str          # megits | binary | paths | graphs | single | original | typed
    model: str = "gcn"      # gcn | gat | hgt
    layers: int = 2
    structure_id: Optional[int] = None


def _variants() -> Dict[str, Variant]:
    v = {
        "metA4API": Variant("metA4API", "megits"),
        "original_adjacency_gcn": Variant("original_adjacency_gcn", "original"),
        "binary_semantic_gcn": Variant("binary_semantic_gcn", "binary"),
        "megits_paths_only": Variant("megits_paths_only", "paths"),
        "megits_graphs_only": Variant("megits_graphs_only", "graphs"),
        "gat": Variant("gat", "original", "gat"),
        "hgt": Variant("hgt", "typed", "hgt"),
    }
    for L in (1, 3, 4):
        v[f"metA4API_{L}layer"] = Variant(f"metA4API_{L}layer", "megits", "gcn", L)
    for s in STRUCTURES:
        v[f"chi{s.id}"] = Variant(f"chi{s.id}", "single", "gcn", 2, s.id)
    return v


VARIANTS = _variants()
ABLATION_VARIANTS = ["original_adjacency_gcn", "binary_semantic_gcn", "megits_paths_only", "megits_graphs_only", "metA4API"]
DEPTH_VARIANTS = ["metA4API_1layer", "metA4API", "metA4API_3layer", "metA4API_4layer"]
BASELINE_VARIANTS = ["original_adjacency_gcn", "gat", "hgt", "metA4API"]
PER_STRUCTURE_VARIANTS = [f"chi{s.id}" for s in STRUCTURES]


@dataclass
class ExperimentConfig:
    n_splits: int = 10
    seeds: Sequence[int] = (0, 1, 2, 3, 4)
    fold_seed: int = 0
    max_folds: Optional[int] = None            # run only the first folds (smoke runs)
    label_fraction: float = 1.0                # 0.25 / 0.5 / 0.75 / 1.0 reduced-label experiment
    allow_ungrouped: bool = False
    train: TrainConfig = field(default_factory=TrainConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)


def build_adjacency(variant: Variant, tikg: TIKG, test_nodes: np.ndarray):
    """Returns (adjacency, MeGiTS weights or {}) for a variant under the fold's leakage-safe protocol."""
    a = variant.adjacency
    if a in ("original", "typed"):
        return tikg.observed_adjacency(), {}
    if a == "megits":
        r = fold_megits_adjacency(tikg, test_nodes)
    elif a == "binary":
        r = fold_megits_adjacency(tikg, test_nodes, binary=True)
    elif a == "paths":
        r = fold_megits_adjacency(tikg, test_nodes, select_structures(kinds=["path"]))
    elif a == "graphs":
        r = fold_megits_adjacency(tikg, test_nodes, select_structures(kinds=["graph"]))
    elif a == "single":
        r = fold_megits_adjacency(tikg, test_nodes, select_structures(ids=[variant.structure_id]))
    else:
        raise ValueError(a)
    return r.adj, r.weights


def _type_ids(tikg: TIKG) -> np.ndarray:
    order = {t.value: i for i, t in enumerate(EntityType)}
    return np.array([order[t] for t in tikg.types])


def _peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024.0 if sys.platform != "darwin" else r / 1024.0 / 1024.0


@dataclass
class ExperimentResults:
    rows: List[Dict[str, Any]]
    oof: Dict[str, np.ndarray]                 # variant -> N x K mean OOF probability (NaN outside test folds)
    folds: List[Fold]
    adjacency: Dict[str, Any] = field(default_factory=dict)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def summary(self, reference: str = "metA4API") -> Dict[str, Any]:
        df = self.frame()
        out: Dict[str, Any] = {}
        metrics = [c for c in df.columns if c not in ("variant", "fold", "seed") and df[c].dtype != object]
        for v, g in df.groupby("variant"):
            out[v] = {m: dict(zip(("mean", "std"), mean_std(g[m].tolist()))) for m in metrics}
        if reference in out:
            ref = df[df.variant == reference].groupby("fold")["macro_f1"].mean()
            for v, g in df.groupby("variant"):
                if v == reference:
                    continue
                cur = g.groupby("fold")["macro_f1"].mean()
                common = ref.index.intersection(cur.index)
                out[v]["wilcoxon_vs_" + reference] = paired_wilcoxon(ref.loc[common].values, cur.loc[common].values)
        return out


def run_cv(tikg: TIKG, labels: LabelSpace, variants: Sequence[str], cfg: ExperimentConfig = ExperimentConfig(),
           folds: Optional[List[Fold]] = None, verbose: bool = True) -> ExperimentResults:
    from .protocol import refuse_at_scale
    refuse_at_scale("experiments.run_cv (legacy harness)", n_nodes=tikg.n, n_splits=cfg.n_splits, seeds=cfg.seeds, max_folds=cfg.max_folds)
    folds = folds or group_stratified_folds(tikg, labels, cfg.n_splits, cfg.fold_seed, allow_ungrouped=cfg.allow_ungrouped)
    use = folds[: cfg.max_folds] if cfg.max_folds else folds
    tids = _type_ids(tikg)
    type_cols = {t.value: labels.type_columns(t) for t in EntityType}
    rel_adjs = tikg.relation_adjacencies()
    rows: List[Dict[str, Any]] = []
    oof_sum = {v: np.zeros_like(labels.Y, dtype=np.float64) for v in variants}
    oof_n = {v: np.zeros(tikg.n) for v in variants}
    for fold in use:
        fb = FeatureBuilder(cfg.features)
        x = fb.fit_transform(tikg, np.concatenate([fold.train, fold.val]))
        for vname in variants:
            var = VARIANTS[vname]
            t0 = time.perf_counter()
            adj, weights = build_adjacency(var, tikg, fold.test_nodes_mask)
            t_adj = time.perf_counter() - t0
            g = make_graph_data(adj, tids, rel_adjs if var.model == "hgt" else None)
            ec_time = float("nan")
            if var.adjacency in ("megits", "binary", "paths", "graphs", "single"):
                _, ec_time = timed_centrality(rank_adjacency(adj))
            for seed in cfg.seeds:
                tr = fold.train
                if cfg.label_fraction < 1.0:
                    rng = np.random.RandomState(1000 * (seed + 1) + fold.index)
                    tr = np.sort(rng.choice(tr, max(1, int(round(cfg.label_fraction * len(tr)))), replace=False))
                tcfg = TrainConfig(**{**asdict(cfg.train), "layers": var.layers})
                res = fit(var.model, x, g, labels.Y, labels.valid, tr, fold.val, tcfg, seed)
                t1 = time.perf_counter()
                _ = fb.transform(tikg)
                prob = predict_proba(res.model, x, g)
                latency = time.perf_counter() - t1
                pred = prob > cfg.train.threshold
                m = multilabel_prf(labels.Y, pred, labels.valid, fold.test)
                m.update(auc_scores(labels.Y, prob, labels.valid, fold.test))
                row = {"variant": vname, "fold": fold.index, "seed": seed, **m,
                       "best_epoch": res.best_epoch, "epochs_run": res.epochs_run,
                       "train_time_per_epoch_s": float(np.mean(res.epoch_times)), "train_time_total_s": res.total_time,
                       "inference_latency_ms": latency * 1e3, "adjacency_build_s": t_adj,
                       "centrality_time_s": ec_time, "peak_rss_mb": _peak_rss_mb(),
                       "n_train": int(len(tr)), "n_val": int(len(fold.val)), "n_test": int(len(fold.test))}
                for t, mm in per_type_prf(labels.Y, pred, labels.valid, fold.test, type_cols, tikg.types).items():
                    row[f"{t}__macro_f1"], row[f"{t}__micro_f1"] = mm["macro_f1"], mm["micro_f1"]
                rows.append(row)
                oof_sum[vname][fold.test] += prob[fold.test]
                oof_n[vname][fold.test] += 1
                if verbose:
                    print(f"[fold {fold.index} seed {seed}] {vname}: macro-F1={m['macro_f1']:.4f} micro-F1={m['micro_f1']:.4f}",
                          flush=True)
    oof = {}
    for v in variants:
        o = np.full_like(labels.Y, np.nan, dtype=np.float64)
        mk = oof_n[v] > 0
        o[mk] = oof_sum[v][mk] / oof_n[v][mk, None]
        oof[v] = o
    return ExperimentResults(rows, oof, use)


def evaluate_external_oof(name: str, probs: np.ndarray, labels: LabelSpace, folds: Sequence[Fold],
                          threshold: float = 0.5) -> pd.DataFrame:
    """Score pre-computed out-of-fold predictions of an external method (e.g. AttacKG, LADDER) with the same
    folds and metrics. `probs`: N x K array aligned with this repo's node order and label columns."""
    rows = []
    for f in folds:
        m = multilabel_prf(labels.Y, probs > threshold, labels.valid, f.test)
        m.update(auc_scores(labels.Y, probs, labels.valid, f.test))
        rows.append({"variant": name, "fold": f.index, "seed": -1, **m})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ reproducibility
def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def write_outputs(out_dir: str | Path, res: ExperimentResults, tikg: TIKG, labels: LabelSpace, cfg: ExperimentConfig,
                  variants: Sequence[str], data_files: Sequence[str | Path] = ()) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.frame().to_csv(out / "results_per_run.csv", index=False)
    summ = res.summary()
    (out / "summary.json").write_text(json.dumps(summ, indent=2, default=float), encoding="utf-8")
    (out / "folds.json").write_text(json.dumps([{"fold": f.index, "train": [tikg.entities[i].id for i in f.train],
                                                 "val": [tikg.entities[i].id for i in f.val],
                                                 "test": [tikg.entities[i].id for i in f.test]} for f in res.folds]), encoding="utf-8")
    for v, o in res.oof.items():
        np.save(out / f"oof_probs_{v}.npy", o)
    (out / "label_columns.json").write_text(json.dumps(labels.names), encoding="utf-8")
    hashes = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in data_files if Path(p).exists()}
    manifest = {
        "config": json.loads(json.dumps(asdict(cfg), default=list)),
        "variants": list(variants), "graph": tikg.stats(),
        "structures": [{"id": s.id, "kind": s.kind, "node_type": s.node_type.value, "schema": s.schema, "formula": s.formula,
                        "provenance": s.provenance} for s in STRUCTURES],
        "n_labelled": int(labels.labelled.sum()), "n_label_columns": labels.K,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "backend": "PyTorch CPU (the manuscript states TensorFlow 2.x / TPU v3-8 - see docs/MANUSCRIPT_MAPPING.md)",
        "git_commit": _git_commit(), "data_sha256": hashes,
        "preflight": {"performed": False, "run_class": "legacy_development_harness", "reportable": False, "not_reportable": True,
                      "note": "legacy harness, not the frozen protocol: never reportable as a manuscript result"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
