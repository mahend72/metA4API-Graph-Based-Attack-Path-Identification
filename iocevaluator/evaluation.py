"""Leakage-safe evaluation harness for the supervised GCN (manuscript Sec. 4.2, 4.5).

Protocol (defaults = manuscript):
  * 10-fold outer cross-validation, group-stratified with campaign-level isolation (``splits.group_stratified_folds``):
    all nodes of a campaign are in the same fold; multi-label stratification uses each node's rarest label;
  * the validation set for early stopping is a campaign-isolated 10% carve-out of the OUTER-TRAIN partition;
  * 5 random initialisations (seeds) per outer fold; folds are fixed across seeds (seeds change the weight
    initialisation and dropout masks only);
  * everything train-dependent is recomputed inside each outer fold from the outer-train partition only
    (``supervised_gcn.prepare_fold``): feature scaling / vocabularies (training nodes only), chi_1..chi_20 commuting matrices, MeGiTS
    similarities (weights are the manuscript's fixed uniform 1/20), normalised adjacency.  Training / early stopping run
    on a graph in which the test-campaign nodes are isolated; test nodes enter only at inference (semi-inductive).
  * every node is scored only against the labels valid for its entity type.

Outputs (``write_evaluation``): results_per_run.csv (fold x seed), results_per_fold.csv (mean over seeds),
results_per_seed.csv (mean over folds), results_per_type.csv, summary.json (mean / std / confidence intervals) and a
deterministic manifest.json.  Wall-clock timings are kept out of the manifest.

Path-level evaluation (optional ``reference_paths``): after the GCN has predicted a fold's held-out campaigns, candidate
attack paths are traced for every reference case whose campaign is in the fold's test partition
(``path_tracing.py``) from the high-confidence / high-priority nodes, and scored against the reference path.  The
reference paths are used only to score; tracing, seeds and scope never see them.

NO GAT / HGT or other baselines here.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy import stats

from .labels import LabelSpace, ReferencePath
from .megits import normalise_weights, record_realisation
from .metagraphs import STRUCTURES, default_structures
from .metrics import auc_scores, multilabel_prf, per_type_prf, topk_hit_rate
from .path_tracing import (PathConfig, StructureSupport, _neighbours, evaluate_case, path_config_info, summarise_cases,
                           trace_case)
from .protocol import (attach_preflight, check_case_integrity, chi_realisation_block, enforce_launch_guard, exhaustive_path_config,
                       realised_fold_record)
from .ranking import ProbabilityBaselineRanker, RankingInputs, ThreatPrioritisationRanker, rank_nodes
from .splits import Fold, group_stratified_folds
from .supervised_gcn import (LabelLeakageError, SupervisedGCNConfig, fit_predict, prepare_fold, reduce_training_labels)
from .tikg import TIKG, EntityType
from .tikg_features import FeatureConfig

MANIFEST_VERSION = "1.0"
GRAPH_REPRESENTATION = {
    "gcn": "MeGiTS-weighted adjacency over the active chi structures; A^ = D~^-1/2 (Adj + I) D~^-1/2",
    "gat": "the same MeGiTS adjacency as the GCN: its support is the edge set (+ self-loops); learned attention replaces the weights",
    "hgt": "native TIKG: entity-type ids and one forward + one reverse relation per triplet signature (T1-T11, +X1) + a self-loop relation; no MeGiTS edges",
}


# ------------------------------------------------------------------------------------------------ configuration
@dataclass(frozen=True)
class EvalConfig:
    n_splits: int = 10
    seeds: Sequence[int] = (0, 1, 2, 3, 4)
    fold_seed: int = 0                       # seed of the outer split (fixed across the model seeds)
    val_fraction: float = 0.10               # inner validation carve-out from the outer-train partition
    allow_ungrouped: bool = False
    model: SupervisedGCNConfig = field(default_factory=SupervisedGCNConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    ks: Sequence[int] = (5, 10)              # top-k hit-rate cut-offs (Table 12)
    paths: PathConfig = field(default_factory=exhaustive_path_config)     # frozen: exhaustive search, see protocol.py
    enforce_path_integrity: bool = True      # a case whose search hit a budget raises PathIntegrityError instead of yielding a result
    max_folds: Optional[int] = None          # smoke runs only; the manuscript protocol uses all folds
    intent: str = "development"              # "final" / "primary" / "manuscript" / "reportable" / "full": needs an active preflight at ANY size


# ------------------------------------------------------------------------------------------------ top-k hit rate
@dataclass(frozen=True)
class TopKCase:
    """One alert / query case q: the campaign whose IOCs are ranked and the expert-confirmed high-risk set R_q^+."""
    case_id: str
    campaign: str
    confirmed: frozenset


def load_topk_cases(path: str | Path) -> List[TopKCase]:
    """JSON list: [{"case_id", "campaign", "confirmed": [entity ids]}] (expert ground truth must be supplied)."""
    return [TopKCase(str(r["case_id"]), str(r["campaign"]), frozenset(r["confirmed"]))
            for r in json.loads(Path(path).read_text(encoding="utf-8"))]


def topk_for_fold(tikg: TIKG, cases: Sequence[TopKCase], fold: Fold, score: np.ndarray, ks: Sequence[int]) -> Dict[str, float]:
    """Top-k hit rate (1/Q) sum_q 1{R_q^+ intersect R_{q,k} != empty} over the cases whose campaign is in this
    fold's test partition.  R_{q,k}: the k highest-scoring nodes of the case's campaign under ``score`` (deterministic:
    ties by entity id, NaN scores not ranked).  ``score`` is the threat-prioritisation risk R (``ranking.py``)."""
    camp = tikg.campaigns()
    test_campaigns = set(camp[fold.test_nodes_mask])
    out: Dict[str, float] = {}
    used = [c for c in cases if c.campaign in test_campaigns]
    ranked = {}
    for c in used:
        ranked[c.case_id] = [tikg.entities[i].id for i in rank_nodes(tikg, score, np.flatnonzero(camp == c.campaign))]
    for k in ks:
        out[f"top{k}_hit_rate"] = topk_hit_rate([(ranked[c.case_id], c.confirmed) for c in used], k)
    out["topk_n_cases"] = len(used)
    return out


# ------------------------------------------------------------------------------------------------ split verification
def verify_fold(tikg: TIKG, labels: LabelSpace, fold: Fold) -> Dict[str, Any]:
    """Hard leakage checks for one outer fold; returns the campaign bookkeeping used in the manifest."""
    camp = tikg.campaigns()
    tr, va, te = set(fold.train.tolist()), set(fold.val.tolist()), set(fold.test.tolist())
    if tr & va or tr & te or va & te:
        raise LabelLeakageError(f"fold {fold.index}: train / validation / test node sets overlap")
    lab = set(np.flatnonzero(labels.labelled).tolist())
    if not (tr | va | te) <= lab:
        raise LabelLeakageError(f"fold {fold.index}: split contains unlabelled nodes")
    c_tr, c_va, c_te = ({str(camp[i]) for i in s} for s in (tr, va, te))
    c_hidden = {str(c) for c in camp[fold.test_nodes_mask]}
    if c_tr & c_te or c_va & c_te:
        raise LabelLeakageError(f"fold {fold.index}: train/validation and test share campaigns {sorted((c_tr | c_va) & c_te)}")
    if c_hidden & (c_tr | c_va) or (c_te - c_hidden):
        raise LabelLeakageError(f"fold {fold.index}: test-campaign node mask is inconsistent with the split")
    h = lambda s: hashlib.sha256(",".join(map(str, sorted(s))).encode()).hexdigest()
    return {"fold": fold.index,
            "train_campaigns": sorted(c_tr), "val_campaigns": sorted(c_va), "test_campaigns": sorted(c_te),
            "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "n_hidden_nodes": int(fold.test_nodes_mask.sum()),
            "train_nodes_sha256": h([tikg.entities[i].id for i in tr]),
            "val_nodes_sha256": h([tikg.entities[i].id for i in va]),
            "test_nodes_sha256": h([tikg.entities[i].id for i in te])}


# ------------------------------------------------------------------------------------------------ the harness
@dataclass
class EvaluationResult:
    runs: pd.DataFrame                 # one row per (fold, seed)
    by_type: pd.DataFrame              # one row per (fold, seed, entity type)
    folds: List[Fold]
    manifest: Dict[str, Any]
    timings: pd.DataFrame
    paths: pd.DataFrame = field(default_factory=pd.DataFrame)       # one row per (fold, seed, reference case)

    def summary(self) -> Dict[str, Any]:
        s = summarise(self.runs, self.by_type)
        if len(self.paths):
            s["paths"] = summarise_cases(self.paths)
        return s


def _run_metrics(labels: LabelSpace, tikg: TIKG, prob: np.ndarray, fold: Fold, cfg: EvalConfig, heads,
                 cases: Optional[Sequence[TopKCase]], score: Optional[np.ndarray]) -> tuple:
    pred = prob > cfg.model.threshold
    m = multilabel_prf(labels.Y, pred, labels.valid, fold.test)
    m["macro_f1_all_labels"] = multilabel_prf(labels.Y, pred, labels.valid, fold.test, macro_over="all")["macro_f1"]
    m.update(auc_scores(labels.Y, prob, labels.valid, fold.test))
    if cases and score is not None:
        m.update(topk_for_fold(tikg, cases, fold, score, cfg.ks))
    else:
        m.update({f"top{k}_hit_rate": float("nan") for k in cfg.ks}, topk_n_cases=0)
    type_cols = {t.value: h.columns for t, h in heads.items()}
    typed = per_type_prf(labels.Y, pred, labels.valid, fold.test, type_cols, tikg.types)
    rows = []
    for t, tm in typed.items():
        sub = fold.test[tikg.types[fold.test] == t]
        a = auc_scores(labels.Y, prob, labels.valid, sub)         # other types' columns are invalid for these nodes
        rows.append({"entity_type": t, "n_test_nodes": int(len(sub)), **tm, **a})
    return m, rows


def evaluate_cv(tikg: TIKG, labels: LabelSpace, cfg: EvalConfig = EvalConfig(),
                topk_cases: Optional[Sequence[TopKCase]] = None, ranker: Optional[Callable] = None,
                severity: Optional[np.ndarray] = None, severity_known: Optional[np.ndarray] = None,
                reference_paths: Optional[Sequence[ReferencePath]] = None,
                folds: Optional[List[Fold]] = None, dataset_info: Optional[Dict[str, Any]] = None,
                verbose: bool = False, adj_builder: Optional[Callable] = None, label_fraction: float = 1.0,
                variant_info: Optional[Dict[str, Any]] = None) -> EvaluationResult:
    """K outer folds x seeds.  Per run: Macro/Micro-F1, ROC-AUC, PR-AUC, top-k hit rates and per-entity-type metrics.

    Top-k hit rate ranks IOCs with ``ranker`` (default: the manuscript's ``ThreatPrioritisationRanker``,
    R = alpha*EC + (1-alpha)*Severity; ``ProbabilityBaselineRanker`` is the old max-probability placeholder).

    Ablation hooks: ``adj_builder(tikg, fold)`` supplies the fold's inference adjacency (default: MeGiTS over chi_1..chi_20)
    and ``label_fraction`` < 1 trains on a seeded subsample of the training labels (``reduce_training_labels``)."""
    import time
    folds = folds or group_stratified_folds(tikg, labels, cfg.n_splits, cfg.fold_seed, cfg.val_fraction,
                                            cfg.allow_ungrouped)
    ranker = ranker if ranker is not None else ThreatPrioritisationRanker()
    # Backstop: a final / manuscript-scale run needs an active, passing preflight session and the preflighted configuration.
    enforce_launch_guard(what="evaluate_cv", n_nodes=tikg.n, cfg=cfg, ranker=ranker, source=(dataset_info or {}).get("source"))
    primary_builder = adj_builder is None                                     # the frozen MeGiTS chi1..chi20 builder
    chi_folds: List[Dict[str, Any]] = []
    use = folds[: cfg.max_folds] if cfg.max_folds else folds
    fold_info = [verify_fold(tikg, labels, f) for f in use]                 # raises on any leakage
    runs, tr_rows, tim, path_rows = [], [], [], []
    support = StructureSupport() if reference_paths else None
    nbrs = _neighbours(tikg) if reference_paths else None
    camp_all = tikg.campaigns()
    for f in use:
        t0 = time.perf_counter()
        with record_realisation() as realised:
            adj_f = adj_builder(tikg, f) if adj_builder is not None else None
            ctx0 = prepare_fold(tikg, labels, f, cfg.features, adj=adj_f, model=cfg.model.model)   # all train-dependent artefacts, per fold
        chi_folds.append(realised_fold_record(f.index, realised))
        t_prep = time.perf_counter() - t0
        for seed in cfg.seeds:
            ctx = reduce_training_labels(ctx0, labels, f, seed, label_fraction)
            t1 = time.perf_counter()
            res, prob = fit_predict(ctx, labels, cfg.model, seed)
            t_fit = time.perf_counter() - t1
            t2 = time.perf_counter()
            res.predict(ctx.x, ctx.infer_graph)                              # one inference forward pass
            t_inf = time.perf_counter() - t2
            score = None
            if topk_cases or reference_paths:
                score = ranker(RankingInputs(tikg, f, prob, labels.valid, ctx.infer_adj, ctx.train_adj, severity,
                                             severity_known, topk_cases or ()))
            m, typed = _run_metrics(labels, tikg, prob, f, cfg, ctx.heads, topk_cases, score)
            if reference_paths:
                prow = _path_level(tikg, labels, f, seed, prob, score, ctx.infer_adj, reference_paths, cfg, support, nbrs, camp_all)
                path_rows += prow
                m.update(_path_run_metrics(prow, cfg.paths))
            alpha = getattr(ranker, "alpha_by_fold", {}).get(f.index)
            runs.append({"fold": f.index, "seed": seed, "model": cfg.model.model, "n_parameters": int(res.n_parameters), **m,
                         "ranker_alpha": float("nan") if alpha is None else float(alpha), "best_epoch": res.best_epoch, "epochs_run": res.epochs_run,
                         "n_train": int(f.train.size), "n_train_used": int(ctx.masks.train.sum()),
                         "n_val": int(f.val.size), "n_test": int(f.test.size)})
            tr_rows += [{"fold": f.index, "seed": seed, **r} for r in typed]
            tim.append({"fold": f.index, "seed": seed, "prepare_fold_s": t_prep, "fit_predict_s": t_fit,
                        "train_time_per_epoch_s": float(np.mean(res.epoch_times)),
                        "train_time_total_s": float(np.sum(res.epoch_times)), "inference_time_s": t_inf})
            if verbose:
                print(f"[fold {f.index} seed {seed}] macro-F1={m['macro_f1']:.4f} micro-F1={m['micro_f1']:.4f}", flush=True)
    rinfo = ranker.info() if hasattr(ranker, "info") else {"name": getattr(ranker, "__name__", "custom"),
                                                          "manuscript_ranker": False}
    manifest = build_manifest(tikg, labels, cfg, fold_info, dataset_info, len(use), rinfo, reference_paths)
    manifest["variant"] = variant_info or {"name": "megits_full", "adjacency": "megits", "label_fraction": label_fraction}
    manifest["variant"]["label_fraction"] = label_fraction
    manifest["chi_realisation"] = chi_realisation_block(chi_folds, primary_builder)        # what each fold ACTUALLY used (part of the hash)
    manifest["content_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in manifest.items() if k not in ("environment", "content_sha256", "preflight")},
                                                           sort_keys=True, default=str).encode()).hexdigest()
    return EvaluationResult(pd.DataFrame(runs), pd.DataFrame(tr_rows), use, manifest, pd.DataFrame(tim),
                            pd.DataFrame(path_rows))


def evaluate_dataset(ds, cfg: EvalConfig = EvalConfig(), topk_cases: Optional[Sequence[TopKCase]] = None,
                     ranker: Optional[Callable] = None, verbose: bool = False,
                     use_reference_paths: bool = False) -> EvaluationResult:
    """Run the harness on a ``datasets.Dataset`` (synthetic, semi-synthetic or real); the manifest records its origin."""
    meta = ds.metadata or {}
    info = {"source": ds.source, "name": ds.name, "profile": meta.get("profile"), "dataset_type": meta.get("dataset_type"),
            "generator_seed": meta.get("seed"), "is_synthetic": bool(ds.is_synthetic)}
    refs = ds.reference_paths if use_reference_paths else None
    return evaluate_cv(ds.tikg, ds.labels, cfg, topk_cases, ranker, ds.severity, ds.severity_known, refs,
                       dataset_info=info, verbose=verbose)


# ------------------------------------------------------------------------------------------------ path-level evaluation
def gcn_confidence(prob: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-node confidence: the highest predicted probability among the labels valid for the node's entity type."""
    return np.where(valid, prob, 0.0).max(1)


def _path_level(tikg, labels, fold, seed, prob, risk, infer_adj, refs, cfg, support, nbrs, camp_all) -> List[Dict[str, Any]]:
    """Trace + score every reference case whose campaign is held out in this fold.  ``refs`` are touched only inside
    ``evaluate_case``; ``trace_case`` receives the case's campaign name, the GCN confidences and the prioritisation scores."""
    test_c = set(camp_all[fold.test_nodes_mask])
    conf = gcn_confidence(prob, labels.valid)
    rows = []
    for ref in sorted((r for r in refs if r.campaign in test_c), key=lambda r: r.case_id):
        tr = trace_case(tikg, ref.campaign, conf, risk, infer_adj, None, cfg.paths, support, nbrs)
        if cfg.enforce_path_integrity:
            check_case_integrity(ref.case_id, tr.search_budget_hit, tr.n_seeds_path_budget_hit, tr.n_seeds_expansion_budget_hit)
        rows.append({"fold": fold.index, "seed": seed, **evaluate_case(tikg, ref, tr, cfg.paths)})
    return rows


def _path_run_metrics(rows: List[Dict[str, Any]], pcfg: PathConfig) -> Dict[str, float]:
    k = pcfg.top_k
    out: Dict[str, float] = {"n_path_cases": len(rows)}
    names = {"": None, "_2edges": "2", "_3edges": "3", "_ge4edges": ">=4"}
    for suf, b in names.items():
        sel = [r for r in rows if b is None or r["bucket"] == b]
        for col, key in ((f"exact_path_hit{k}{suf}", "exact_hit"), (f"path_edge_f1{suf}", "edge_f1"),
                         (f"path_relation_f1{suf}", "relation_f1")):
            out[col] = float(np.mean([r[key] for r in sel])) if sel else float("nan")
    return out


# ------------------------------------------------------------------------------------------------ aggregation
def _ci_t(x: np.ndarray, level: float = 0.95) -> List[float]:
    n = len(x)
    if n < 2:
        return [float("nan"), float("nan")]
    h = stats.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return [float(x.mean() - h), float(x.mean() + h)]


def _ci_boot(x: np.ndarray, level: float = 0.95, n_boot: int = 2000, seed: int = 0) -> List[float]:
    if len(x) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.RandomState(seed)
    means = x[rng.randint(0, len(x), (n_boot, len(x)))].mean(1)
    return [float(np.quantile(means, (1 - level) / 2)), float(np.quantile(means, 0.5 + level / 2))]


def aggregate_metric(df: pd.DataFrame, metric: str) -> Dict[str, Any]:
    """Aggregates over runs (fold x seed), as in the manuscript, plus intervals that respect the fold structure.

    mean / std       over ALL runs (10 x 5 = 50 by default); std uses ddof=1.
    fold_mean/std    over the per-fold means (seeds averaged within a fold) - the independent units.
    ci95_folds_t     95% Student-t interval of the mean of the fold means (primary interval).
    ci95_folds_boot  95% percentile bootstrap interval over the fold means (2000 resamples, fixed seed).
    ci95_runs_t      95% t interval treating all runs as independent (optimistic: seeds of one fold share data)."""
    runs = df[metric].to_numpy(dtype=float)
    runs = runs[~np.isnan(runs)]
    fm = df.groupby("fold")[metric].mean().dropna().to_numpy(dtype=float)
    return {"n_runs": int(len(runs)), "n_folds": int(len(fm)),
            "mean": float(runs.mean()) if len(runs) else float("nan"),
            "std": float(runs.std(ddof=1)) if len(runs) > 1 else float("nan"),
            "fold_mean": float(fm.mean()) if len(fm) else float("nan"),
            "fold_std": float(fm.std(ddof=1)) if len(fm) > 1 else float("nan"),
            "ci95_folds_t": _ci_t(fm), "ci95_folds_boot": _ci_boot(fm), "ci95_runs_t": _ci_t(runs)}


METRICS = ["macro_f1", "micro_f1", "macro_precision", "macro_recall", "micro_precision", "micro_recall",
           "macro_f1_all_labels", "roc_auc", "pr_auc"]


def summarise(runs: pd.DataFrame, by_type: pd.DataFrame) -> Dict[str, Any]:
    cols = [m for m in METRICS if m in runs] + [c for c in runs if c.startswith("top") and c.endswith("_hit_rate")] + \
           [c for c in runs if c.startswith(("exact_path_hit", "path_edge_f1", "path_relation_f1"))]
    out: Dict[str, Any] = {"overall": {m: aggregate_metric(runs, m) for m in cols}, "by_entity_type": {},
                           "aggregation": {"std_ddof": 1, "ci_level": 0.95,
                                           "primary_interval": "ci95_folds_t (t-interval over per-fold means)"}}
    for t, g in by_type.groupby("entity_type"):
        out["by_entity_type"][t] = {m: aggregate_metric(g, m) for m in
                                    ("macro_f1", "micro_f1", "roc_auc", "pr_auc") if m in g}
    return out


# ------------------------------------------------------------------------------------------------ manifest
def _git() -> Dict[str, Any]:
    def run(*a):
        return subprocess.check_output(["git", *a], stderr=subprocess.DEVNULL, text=True).strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except Exception:
        return {"commit": "unknown", "dirty": None}


def dataset_fingerprint(tikg: TIKG, labels: LabelSpace) -> str:
    """Content hash of the graph (entities, campaigns, triplets) and the label matrix."""
    h = hashlib.sha256()
    h.update(json.dumps([[e.id, e.type.value, e.subtype, e.campaign] for e in tikg.entities]).encode())
    h.update(json.dumps(sorted([tikg.entities[t.s].id, t.r, tikg.entities[t.t].id] for t in tikg.triplets)).encode())
    h.update(json.dumps(labels.names).encode())
    h.update(np.ascontiguousarray(labels.Y).tobytes())
    h.update(np.ascontiguousarray(labels.labelled).tobytes())
    return h.hexdigest()


def build_manifest(tikg: TIKG, labels: LabelSpace, cfg: EvalConfig, fold_info: List[Dict[str, Any]],
                   dataset_info: Optional[Dict[str, Any]], n_folds_run: int,
                   ranker_info: Optional[Dict[str, Any]] = None,
                   reference_paths: Optional[Sequence[ReferencePath]] = None) -> Dict[str, Any]:
    """Deterministic manifest: identical for identical data / configuration / code (no timestamps or timings).
    ``environment`` (git, library versions) is kept separate and excluded from ``content_sha256``."""
    structures = default_structures()
    w = normalise_weights(structures)
    det = {
        "manifest_version": MANIFEST_VERSION,
        "dataset": {**(dataset_info or {}), "n_nodes": tikg.n, "n_triplets": len(tikg.triplets),
                    "n_label_columns": labels.K, "n_labelled": int(labels.labelled.sum()),
                    "n_campaigns": int(len({c for c in tikg.campaigns() if c})),
                    "label_classes_per_type": {t.value: len(c) for t, c in labels.classes.items()},
                    "fingerprint_sha256": dataset_fingerprint(tikg, labels)},
        "protocol": {"n_splits": cfg.n_splits, "folds_run": n_folds_run, "seeds": list(cfg.seeds),
                     "fold_seed": cfg.fold_seed, "val_fraction": cfg.val_fraction,
                     "split": "StratifiedGroupKFold(shuffle) over campaigns + GroupShuffleSplit validation carve-out",
                     "runs": [{"fold": fi["fold"], "seed": s} for fi in fold_info for s in cfg.seeds]},
        "model": asdict(cfg.model),
        "graph_representation": GRAPH_REPRESENTATION.get(cfg.model.model, ""),
        "features": {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in asdict(cfg.features).items()},
        "metrics": {"ks": list(cfg.ks), "threshold": cfg.model.threshold},
        "ranking": ranker_info or {},
        "paths": {**path_config_info(cfg.paths), "enabled": bool(reference_paths),
                  "enforce_path_integrity": cfg.enforce_path_integrity,
                  "structures": [s.id for s in STRUCTURES], "orientation": "higher-priority endpoint first",
                  "reference_paths": {"n_cases": len(reference_paths or []), "role": "evaluation only",
                                      "sha256": hashlib.sha256(json.dumps(
                                          [[r.case_id, r.campaign, r.nodes, r.relations] for r in
                                           sorted(reference_paths or [], key=lambda r: r.case_id)]).encode()).hexdigest()}},
        "feature_scaler_fit_on": "outer-train nodes only (validation excluded)",
        "training_graph": "inductive: held-out-campaign nodes isolated; inference graph is full (no labels)",
        "chi_weights": {f"chi{s.id}": w[s.id] for s in structures},
        "chi_structures": [{"id": s.id, "kind": s.kind, "formula": s.formula} for s in STRUCTURES],
        "folds": fold_info,
    }
    content = hashlib.sha256(json.dumps(det, sort_keys=True, default=str).encode()).hexdigest()
    # `preflight` (git commit, dirty flag, hashes of the code) is kept OUT of content_sha256: that hash stays a function of
    # the data / configuration only.
    return {**det, "content_sha256": content, "preflight": attach_preflight({})["preflight"],
            "environment": {"git": _git(), "python": platform.python_version(), "torch": torch.__version__,
                            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                            "pandas": pd.__version__, "platform": sys.platform}}


# ------------------------------------------------------------------------------------------------ outputs
def write_evaluation(out_dir: str | Path, res: EvaluationResult) -> Dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = res.runs
    metric_cols = [c for c in runs.columns if c not in ("fold", "seed") and pd.api.types.is_numeric_dtype(runs[c])]
    files = {
        "results_per_run.csv": runs,
        "results_per_fold.csv": runs.groupby("fold")[metric_cols].mean().reset_index(),
        "results_per_seed.csv": runs.groupby("seed")[metric_cols].mean().reset_index(),
        "results_per_type.csv": res.by_type,
        "timings.csv": res.timings,
    }
    if len(res.paths):
        files["results_path_cases.csv"] = res.paths
    paths = {}
    for name, df in files.items():
        paths[name] = out / name
        df.to_csv(paths[name], index=False, float_format="%.10g")
    paths["summary.json"] = out / "summary.json"
    paths["summary.json"].write_text(json.dumps(res.summary(), indent=2, sort_keys=True), encoding="utf-8")
    paths["manifest.json"] = out / "manifest.json"
    paths["manifest.json"].write_text(json.dumps(res.manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return paths
