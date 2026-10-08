"""Ablation-study framework (manuscript Sec. 4.2 "Semantic ablation", Sec. 4.3-4.4, Table 13, reduced-label robustness).

RESULT INTEGRITY.  This module contains no manuscript score, tunes nothing and selects no seed or hyper-parameter.
The numbers in ``manuscript.tex`` are historical reference values only; every number produced here comes from the given
dataset, the code and the declared protocol, and is compared with nothing but the full chi_1..chi_20 model run under the
same folds and seeds.  Differences from the manuscript tables are to be reported, never "fixed".

Variants (``VARIANTS``; the reference is ``megits_full``):
  adjacency   original_adjacency   observed TIKG relations only (symmetrised, binary, no self-loops); no chi structure
              binary_semantic      all 20 structures, MeGiTS weights dropped: A_ij = 1 iff i, j share >= 1 instance of
                                   any chi_k (union of the structures' supports)
              megits_paths_only    chi_1..chi_11 (the base meta-paths of Fig. 3), weighted MeGiTS
              megits_graphs_only   chi_12..chi_20 (the composite meta-graphs of Fig. 3), weighted MeGiTS
              megits_full          chi_1..chi_20, weighted MeGiTS, w_k = 1/20            (2-layer GCN, 100% labels)
  structures  chi1 .. chi20        each structure alone, weighted MeGiTS (w = 1)
  depth       megits_full_{1,3,4}layer   same adjacency, 1 / 3 / 4 GCN layers (the 2-layer model is megits_full)
  label       megits_full_lf{25,50,75}   same model trained with 25 / 50 / 75 % of the training labels (100% = megits_full)
For a subset S of structures the uniform weights are renormalised, w_k = 1/|S| (weights must sum to 1, Sec. 3.4).

Paired design.  The outer folds (campaign-isolated 10-fold split), the inner validation carve-outs and the model seeds are
computed ONCE and shared by all variants; every variant recomputes its own fold-dependent quantities (feature scaling,
commuting matrices, MeGiTS statistics, normalised adjacency, model) from the outer-train partition inside every fold.
Paired comparisons align (fold, seed) pairs of a variant with the reference and refuse to run on a mismatch.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from .evaluation import (EvalConfig, EvaluationResult, TopKCase, _ci_boot, _ci_t, _git, aggregate_metric, build_manifest,
                         evaluate_cv, summarise, verify_fold)
from .labels import LabelSpace, ReferencePath
from .megits import fold_megits_adjacency, normalise_weights
from .metagraphs import STRUCTURES, select_structures
from .metrics import holm_adjust, paired_wilcoxon
from .ranking import ThreatPrioritisationRanker
from .splits import Fold, group_stratified_folds
from .tikg import TIKG

REFERENCE = "megits_full"
PATH_IDS = tuple(s.id for s in STRUCTURES if s.kind == "path")        # chi_1..chi_11  (Fig. 3 base meta-paths)
GRAPH_IDS = tuple(s.id for s in STRUCTURES if s.kind == "graph")      # chi_12..chi_20 (Fig. 3 composite meta-graphs)
ALL_IDS = tuple(s.id for s in STRUCTURES)
LABEL_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
DEPTHS = (1, 2, 3, 4)


# ------------------------------------------------------------------------------------------------ variants
@dataclass(frozen=True)
class AblationVariant:
    name: str
    group: str                              # adjacency | structures | depth | label | reference
    adjacency: str                          # "original" | "binary" | "megits"
    structure_ids: Tuple[int, ...] = ()     # active chi structures (empty for the original adjacency)
    layers: int = 2
    label_fraction: float = 1.0
    description: str = ""

    @property
    def weights(self) -> Dict[int, float]:
        """MeGiTS weights of the active structures (uniform, renormalised over the subset); empty for 'original'."""
        if not self.structure_ids:
            return {}
        return normalise_weights(select_structures(ids=self.structure_ids))

    def info(self) -> Dict[str, Any]:
        return {"name": self.name, "group": self.group, "adjacency": self.adjacency,
                "binary": self.adjacency == "binary", "active_structures": list(self.structure_ids),
                "chi_weights": {f"chi{k}": w for k, w in self.weights.items()}, "layers": self.layers,
                "label_fraction": self.label_fraction, "description": self.description}


def _variants() -> Dict[str, AblationVariant]:
    v: List[AblationVariant] = [
        AblationVariant("original_adjacency", "adjacency", "original", (), description="observed TIKG relations only"),
        AblationVariant("binary_semantic", "adjacency", "binary", ALL_IDS,
                        description="all 20 structures, unweighted: 1 iff >= 1 shared instance of any chi_k"),
        AblationVariant("megits_paths_only", "adjacency", "megits", PATH_IDS, description="chi_1..chi_11 (meta-paths)"),
        AblationVariant("megits_graphs_only", "adjacency", "megits", GRAPH_IDS, description="chi_12..chi_20 (meta-graphs)"),
        AblationVariant(REFERENCE, "reference", "megits", ALL_IDS, description="full chi_1..chi_20 MeGiTS, 2 layers, 100% labels"),
    ]
    v += [AblationVariant(f"chi{k}", "structures", "megits", (k,), description=f"chi_{k} alone") for k in ALL_IDS]
    v += [AblationVariant(f"megits_full_{L}layer", "depth", "megits", ALL_IDS, layers=L, description=f"{L}-layer GCN")
          for L in DEPTHS if L != 2]
    v += [AblationVariant(f"megits_full_lf{int(round(f * 100))}", "label", "megits", ALL_IDS, label_fraction=f,
                          description=f"{int(round(f * 100))}% of the training labels") for f in LABEL_FRACTIONS if f < 1.0]
    return {x.name: x for x in v}


VARIANTS: Dict[str, AblationVariant] = _variants()
GROUPS: Dict[str, List[str]] = {
    "adjacency": ["original_adjacency", "binary_semantic", "megits_paths_only", "megits_graphs_only", REFERENCE],
    "structures": [f"chi{k}" for k in ALL_IDS],
    "depth": ["megits_full_1layer", REFERENCE, "megits_full_3layer", "megits_full_4layer"],
    "label": ["megits_full_lf25", "megits_full_lf50", "megits_full_lf75", REFERENCE],
}
GROUPS["all"] = list(dict.fromkeys(sum(GROUPS.values(), [])))


def select_variants(names_or_groups: Sequence[str] = ("all",)) -> List[AblationVariant]:
    out: List[str] = []
    for n in names_or_groups:
        out += GROUPS[n] if n in GROUPS else [n]
    if REFERENCE not in out:
        out.append(REFERENCE)                                   # paired comparisons need the reference
    unknown = [n for n in out if n not in VARIANTS]
    if unknown:
        raise KeyError(f"unknown ablation variant(s): {unknown}")
    return [VARIANTS[n] for n in dict.fromkeys(out)]


def build_variant_adjacency(variant: AblationVariant, tikg: TIKG, fold: Fold) -> sp.csr_matrix:
    """The variant's INFERENCE adjacency for one fold, built from that fold's data (the training graph is derived by
    isolating the held-out nodes, see ``supervised_gcn.prepare_fold``)."""
    if variant.adjacency == "original":
        return tikg.observed_adjacency()
    structures = select_structures(ids=variant.structure_ids)
    return fold_megits_adjacency(tikg, fold.test_nodes_mask, structures, None, binary=variant.adjacency == "binary").adj


# ------------------------------------------------------------------------------------------------ provenance
def provenance_info(ds, protocol_frozen: bool = False) -> Dict[str, Any]:
    """How the results of a dataset may be described.  synthetic -> development only; semi_synthetic -> measured on the
    semi-synthetic benchmark (not a reproduction); real -> eligible for the manuscript once the protocol is frozen."""
    if ds.source == "synthetic":
        status, ok = "development/testing results only", False
    elif ds.source == "semi_synthetic":
        status, ok = ("real measured results on the semi-synthetic benchmark; NOT a reproduction of the original "
                      "real-data experiments", False)
    elif ds.source == "real":
        ok = bool(protocol_frozen)
        status = ("eligible for final manuscript reporting (protocol frozen)" if ok else
                  "eligible for final manuscript reporting once the real dataset is reconstructed and the protocol is "
                  "frozen (protocol NOT frozen)")
    else:
        status, ok = "unknown dataset source: not reportable", False
    return {"source": ds.source, "dataset": ds.name, "status": status, "reportable_in_manuscript": ok,
            "protocol_frozen": bool(protocol_frozen),
            "manuscript_values": "historical reference only; not used, tuned against or compared by this code"}


# ------------------------------------------------------------------------------------------------ run
@dataclass
class AblationResult:
    runs: pd.DataFrame                       # variant x fold x seed (raw)
    by_type: pd.DataFrame
    paths: pd.DataFrame
    timings: pd.DataFrame
    folds: List[Fold]
    manifest: Dict[str, Any]
    variants: List[Any]
    per_variant: Dict[str, EvaluationResult] = field(default_factory=dict)
    reference: str = REFERENCE

    def summary(self) -> Dict[str, Any]:
        return {v: summarise(self.runs[self.runs.variant == v].drop(columns="variant"),
                             self.by_type[self.by_type.variant == v].drop(columns="variant")) for v in
                self.runs.variant.unique()}

    def paired(self, metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
        return paired_comparison(self.runs, self.reference, metrics)


def run_ablation(tikg: TIKG, labels: LabelSpace, variants: Sequence[AblationVariant], cfg: EvalConfig = EvalConfig(),
                 topk_cases: Optional[Sequence[TopKCase]] = None, reference_paths: Optional[Sequence[ReferencePath]] = None,
                 severity: Optional[np.ndarray] = None, severity_known: Optional[np.ndarray] = None,
                 dataset_info: Optional[Dict[str, Any]] = None, provenance: Optional[Dict[str, Any]] = None,
                 folds: Optional[List[Fold]] = None, verbose: bool = False) -> AblationResult:
    """Run every variant with IDENTICAL folds and seeds; each variant recomputes all fold-dependent artefacts itself."""
    folds = folds or group_stratified_folds(tikg, labels, cfg.n_splits, cfg.fold_seed, cfg.val_fraction, cfg.allow_ungrouped)
    use = folds[: cfg.max_folds] if cfg.max_folds else folds
    results: Dict[str, EvaluationResult] = {}
    for v in variants:
        vcfg = replace(cfg, model=replace(cfg.model, layers=v.layers))
        if verbose:
            print(f"[ablation] {v.name}", flush=True)
        results[v.name] = evaluate_cv(
            tikg, labels, vcfg, topk_cases, ThreatPrioritisationRanker(), severity, severity_known, reference_paths,
            folds=folds, dataset_info={**(dataset_info or {}), "provenance": (provenance or {}).get("status")},
            adj_builder=lambda g, f, v=v: build_variant_adjacency(v, g, f), label_fraction=v.label_fraction,
            variant_info=v.info())
    tag = lambda df, v: df.assign(variant=v) if len(df) else df
    cat = lambda attr: pd.concat([tag(getattr(r, attr), n) for n, r in results.items() if len(getattr(r, attr))],
                                 ignore_index=True) if any(len(getattr(r, attr)) for r in results.values()) else pd.DataFrame()
    status = (provenance or {}).get("status", "unspecified")
    runs, by_type, paths, timings = (cat("runs"), cat("by_type"), cat("paths"), cat("timings"))
    for df in (runs, by_type, paths, timings):
        if len(df):
            df["data_provenance"] = status
    ref_res = results.get(REFERENCE) or next(iter(results.values()))
    manifest = build_ablation_manifest(ref_res, variants, results, provenance, use)
    return AblationResult(runs, by_type, paths, timings, use, manifest, list(variants), results)


def build_ablation_manifest(ref_res: EvaluationResult, variants: Sequence[Any],
                            results: Dict[str, EvaluationResult], provenance: Optional[Dict[str, Any]],
                            folds: Sequence[Fold], kind: str = "ablation_study", reference: str = REFERENCE,
                            extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Deterministic manifest: dataset, provenance, protocol (seeds, folds, campaigns), model / feature configuration and,
    per variant, the active chi structures, their weights, layers and label fraction.  ``environment`` is excluded from
    ``content_sha256``."""
    base = {k: v for k, v in ref_res.manifest.items() if k not in ("variant", "environment", "content_sha256", "chi_weights", "model", "preflight", "integrity")}
    det = {**base,
           "kind": kind,
           "provenance": provenance or {},
           "model_base": ref_res.manifest["model"],
           "reference_variant": reference,
           **(extra or {}),
           "variants": [{**v.info(), "content_sha256": results[v.name].manifest["content_sha256"]} for v in variants],
           "pairing": {"folds": [f.index for f in folds], "seeds": ref_res.manifest["protocol"]["seeds"],
                       "same_folds_and_seeds_for_all_variants": True},
           "label_fraction_protocol": ("label-coverage / distribution preserving ordering of the labelled TRAINING nodes "
                                       "(greedy label cover, then stratified interleaving by rarest label), seeded per "
                                       "(fold, seed); the first round(f*n) nodes are kept so 25% < 50% < 75% are nested; "
                                       "validation/test labels untouched; feature scaler and MeGiTS statistics stay those "
                                       "of the full outer-train partition")}
    content = hashlib.sha256(json.dumps(det, sort_keys=True, default=str).encode()).hexdigest()
    return {**det, "content_sha256": content, "preflight": ref_res.manifest.get("preflight"),
            "environment": ref_res.manifest["environment"]}


# ------------------------------------------------------------------------------------------------ paired comparison
PAIRED_METRICS = ["macro_f1", "micro_f1", "roc_auc", "pr_auc", "top5_hit_rate", "top10_hit_rate", "exact_path_hit5", "path_edge_f1"]


def paired_comparison(runs: pd.DataFrame, reference: str = REFERENCE, metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Each variant against ``reference``, aligned on (fold, seed).

    diff = metric(variant) - metric(reference) per (fold, seed); runs with an undefined metric on either side are dropped.
    Reported per (variant, metric): n_pairs, n_folds, mean_diff, std_diff (ddof=1, over runs), the per-fold mean differences'
    95% Student-t interval (primary) and bootstrap interval, the paired Wilcoxon signed-rank test over per-fold mean
    differences (alpha 0.05) with its Holm-adjusted p-value (one family per metric over the compared variants), and the
    number of folds where the variant is better / worse / tied."""
    if reference not in set(runs.variant):
        raise KeyError(f"reference variant {reference!r} not in results")
    metrics = [m for m in (metrics or PAIRED_METRICS) if m in runs.columns]
    ref = runs[runs.variant == reference].set_index(["fold", "seed"])
    rows = []
    for v in sorted(set(runs.variant) - {reference}):
        cur = runs[runs.variant == v].set_index(["fold", "seed"])
        if set(cur.index) != set(ref.index):
            raise ValueError(f"variant {v!r} is not paired with the reference: (fold, seed) sets differ")
        for m in metrics:
            d = (cur[m] - ref.loc[cur.index, m]).dropna()
            if d.empty:
                rows.append({"variant": v, "reference": reference, "metric": m, "n_pairs": 0})
                continue
            fm = d.groupby(level="fold").mean().to_numpy()
            w = paired_wilcoxon(fm, np.zeros_like(fm))
            rows.append({"variant": v, "reference": reference, "metric": m, "n_pairs": int(len(d)), "n_folds": int(len(fm)),
                         "mean_diff": float(d.mean()), "std_diff": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
                         "ci95_folds_t_lo": _ci_t(fm)[0], "ci95_folds_t_hi": _ci_t(fm)[1],
                         "ci95_folds_boot_lo": _ci_boot(fm)[0], "ci95_folds_boot_hi": _ci_boot(fm)[1],
                         "wilcoxon_p": w["p_value"], "significant_0.05": w["significant"],
                         "folds_better": int((fm > 1e-12).sum()), "folds_worse": int((fm < -1e-12).sum()),
                         "folds_tied": int((np.abs(fm) <= 1e-12).sum())})
    df = pd.DataFrame(rows)
    if len(df) and "wilcoxon_p" in df:
        df["wilcoxon_p_holm"] = np.nan
        for m in df.metric.unique():                                  # one Holm family per metric, over the variants
            sel = df.metric == m
            df.loc[sel, "wilcoxon_p_holm"] = holm_adjust(df.loc[sel, "wilcoxon_p"].to_numpy(dtype=float))
        df["significant_holm_0.05"] = df["wilcoxon_p_holm"] < 0.05
    return df


def write_ablation(out_dir: str | Path, res: AblationResult) -> Dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {"ablation_runs.csv": res.runs, "ablation_per_type.csv": res.by_type, "ablation_timings.csv": res.timings,
             "ablation_paired.csv": res.paired()}
    if len(res.paths):
        files["ablation_path_cases.csv"] = res.paths
    paths = {}
    for name, df in files.items():
        paths[name] = out / name
        df.to_csv(paths[name], index=False, float_format="%.10g")
    paths["ablation_summary.json"] = out / "ablation_summary.json"
    paths["ablation_summary.json"].write_text(json.dumps(
        {"provenance": res.manifest.get("provenance"), "reference_variant": REFERENCE, "variants": res.summary()},
        indent=2, sort_keys=True), encoding="utf-8")
    paths["manifest.json"] = out / "manifest.json"
    paths["manifest.json"].write_text(json.dumps(res.manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return paths


def run_ablation_dataset(ds, variants: Sequence[str] = ("all",), cfg: EvalConfig = EvalConfig(),
                         topk_cases: Optional[Sequence[TopKCase]] = None, use_reference_paths: bool = True,
                         protocol_frozen: bool = False, verbose: bool = False) -> AblationResult:
    """Ablation on a ``datasets.Dataset``; the manifest and every output row carry the dataset provenance."""
    from .protocol import require_preflight_if_marked
    require_preflight_if_marked("run_ablation_dataset", cfg, protocol_frozen=protocol_frozen)
    meta = ds.metadata or {}
    info = {"source": ds.source, "name": ds.name, "profile": meta.get("profile"), "dataset_type": meta.get("dataset_type"),
            "generator_seed": meta.get("seed"), "is_synthetic": bool(ds.is_synthetic)}
    return run_ablation(ds.tikg, ds.labels, select_variants(variants), cfg, topk_cases,
                        ds.reference_paths if use_reference_paths else None, ds.severity, ds.severity_known,
                        info, provenance_info(ds, protocol_frozen), verbose=verbose)
