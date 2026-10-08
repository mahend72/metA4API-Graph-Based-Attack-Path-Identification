"""Model comparison under the leakage-safe protocol: the supervised GCN (reference) against GAT and HGT (Sec. 4.3, Table 9).

RESULT INTEGRITY.  No manuscript score is used, tuned against or compared here.  Architectures and hyper-parameters were fixed
from the architecture descriptions and parameter counts alone, before any result existed; validation data are used only for
early stopping.  Whatever the measured results are, they are reported as they come.

Identical across models: the 10 campaign-isolated outer folds, the inner validation partitions, the 5 seeds, the labels (and
the heterogeneous per-type label masks), the node features (scaler fitted on the training nodes only), Adam lr 1e-3, L2 5e-4,
dropout 0.5, early stopping on validation Macro-F1 with patience 20, the metrics and the held-out campaigns.
Only the propagation mechanism / graph input differs:

  gcn   2-layer GCN, hidden 64, no bias: sigmoid(A^ ReLU(A^ F W0) W1) on the MeGiTS-weighted adjacency (chi_1..chi_20).
  gat   2-layer graph attention, hidden 64 = 4 heads x 16 (ELU) then a single-head 64 -> K output layer, over the SAME
        MeGiTS adjacency: its support is the edge set (plus self-loops); attention replaces the MeGiTS edge weights.
  hgt   HGT-style: linear input -> hidden, then 2 heterogeneous attention layers (4 heads) with node-type-specific
        Q/K/V/output maps and relation-specific attention / message matrices, then a linear K-way output; message passing
        runs over the native TIKG (7 entity types; one forward + one reverse relation per triplet signature T1-T11 plus a
        self-loop relation), not over MeGiTS edges.  Default hidden width 16 (heads of width 4): at hidden 64 the type- and
        relation-specific weights make HGT ~30x larger than the GCN, so the width was fixed from the parameter budget
        (about 2-3x the GCN) and a hidden-64 variant ``hgt_h64`` is available for sensitivity.
Heterogeneous label spaces: every model has one K = sum_t K_t output layer; loss, early stopping and predictions are
restricted to the columns valid for each node's entity type (``LabelSpace.valid``).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .ablation import (AblationResult, build_ablation_manifest, provenance_info, write_ablation)
from .evaluation import EvalConfig, EvaluationResult, GRAPH_REPRESENTATION, TopKCase, evaluate_cv
from .labels import LabelSpace, ReferencePath
from .ranking import ThreatPrioritisationRanker
from .splits import Fold, group_stratified_folds
from .supervised_gcn import SupervisedGCNConfig
from .tikg import TIKG

REFERENCE_MODEL = "gcn"


@dataclass(frozen=True)
class ModelVariant:
    name: str
    model: str                      # gcn | gat | hgt
    hidden: int = 64
    heads: int = 4
    description: str = ""

    def config(self, base: SupervisedGCNConfig) -> SupervisedGCNConfig:
        return replace(base, model=self.model, hidden=self.hidden, heads=self.heads)

    def info(self) -> Dict[str, Any]:
        return {"name": self.name, "group": "model", "model": self.model, "hidden": self.hidden,
                "heads": self.heads if self.model != "gcn" else None, "graph": GRAPH_REPRESENTATION[self.model],
                "description": self.description}


MODELS: Dict[str, ModelVariant] = {m.name: m for m in (
    ModelVariant("gcn", "gcn", 64, 4, "reference: 2-layer GCN on the MeGiTS adjacency (chi_1..chi_20)"),
    ModelVariant("gat", "gat", 64, 4, "2-layer GAT, 4 heads x 16, over the same MeGiTS adjacency"),
    ModelVariant("hgt", "hgt", 16, 4, "HGT-style on the native TIKG types / relations, hidden 16 (parameter-budget width)"),
    ModelVariant("hgt_h64", "hgt", 64, 4, "HGT-style on the native TIKG, hidden 64 (same width as the GCN; sensitivity only)"),
)}
DEFAULT_MODELS = ("gcn", "gat", "hgt")


def select_models(names: Sequence[str] = DEFAULT_MODELS) -> List[ModelVariant]:
    out = list(dict.fromkeys([REFERENCE_MODEL, *names]))
    unknown = [n for n in out if n not in MODELS]
    if unknown:
        raise KeyError(f"unknown model(s): {unknown}")
    return [MODELS[n] for n in out]


def parameter_table(runs: pd.DataFrame, reference: str = REFERENCE_MODEL) -> pd.DataFrame:
    """Trainable-parameter counts per model (the input width varies slightly across folds because the feature vocabularies
    are fitted per fold) and the ratio to the reference."""
    g = runs.groupby("variant").n_parameters
    t = pd.DataFrame({"n_runs": g.size(), "params_mean": g.mean(), "params_min": g.min(), "params_max": g.max()}).reset_index()
    ref = float(t.loc[t.variant == reference, "params_mean"].iloc[0]) if (t.variant == reference).any() else float("nan")
    t["ratio_to_reference"] = t.params_mean / ref
    return t


def run_model_comparison(tikg: TIKG, labels: LabelSpace, models: Sequence[ModelVariant], cfg: EvalConfig = EvalConfig(),
                         topk_cases: Optional[Sequence[TopKCase]] = None,
                         reference_paths: Optional[Sequence[ReferencePath]] = None,
                         severity: Optional[np.ndarray] = None, severity_known: Optional[np.ndarray] = None,
                         dataset_info: Optional[Dict[str, Any]] = None, provenance: Optional[Dict[str, Any]] = None,
                         folds: Optional[List[Fold]] = None, verbose: bool = False) -> AblationResult:
    """Every model on IDENTICAL folds, seeds, labels and features; each recomputes its fold-dependent artefacts itself."""
    folds = folds or group_stratified_folds(tikg, labels, cfg.n_splits, cfg.fold_seed, cfg.val_fraction, cfg.allow_ungrouped)
    use = folds[: cfg.max_folds] if cfg.max_folds else folds
    results: Dict[str, EvaluationResult] = {}
    for v in models:
        vcfg = replace(cfg, model=v.config(cfg.model))
        if verbose:
            print(f"[model comparison] {v.name}", flush=True)
        results[v.name] = evaluate_cv(
            tikg, labels, vcfg, topk_cases, ThreatPrioritisationRanker(), severity, severity_known, reference_paths,
            folds=folds, dataset_info={**(dataset_info or {}), "provenance": (provenance or {}).get("status")},
            variant_info=v.info())
    tag = lambda df, n: df.assign(variant=n) if len(df) else df
    cat = lambda attr: pd.concat([tag(getattr(r, attr), n) for n, r in results.items() if len(getattr(r, attr))],
                                 ignore_index=True) if any(len(getattr(r, attr)) for r in results.values()) else pd.DataFrame()
    status = (provenance or {}).get("status", "unspecified")
    runs, by_type, paths, timings = (cat("runs"), cat("by_type"), cat("paths"), cat("timings"))
    for df in (runs, by_type, paths, timings):
        if len(df):
            df["data_provenance"] = status
    ref_res = results[REFERENCE_MODEL]
    ptab = parameter_table(runs)
    manifest = build_ablation_manifest(ref_res, list(models), results, provenance, use, "model_comparison", REFERENCE_MODEL,
                                       {"parameter_counts": ptab.round(6).to_dict(orient="records"),
                                        "shared_across_models": ["folds", "inner validation", "seeds", "labels", "features",
                                                                 "optimiser", "early stopping", "metrics"]})
    return AblationResult(runs, by_type, paths, timings, use, manifest, list(models), results, REFERENCE_MODEL)


def run_model_comparison_dataset(ds, models: Sequence[str] = DEFAULT_MODELS, cfg: EvalConfig = EvalConfig(),
                                 topk_cases: Optional[Sequence[TopKCase]] = None, use_reference_paths: bool = False,
                                 protocol_frozen: bool = False, verbose: bool = False) -> AblationResult:
    from .protocol import require_preflight_if_marked
    require_preflight_if_marked("run_model_comparison_dataset", cfg, protocol_frozen=protocol_frozen)
    meta = ds.metadata or {}
    info = {"source": ds.source, "name": ds.name, "profile": meta.get("profile"), "dataset_type": meta.get("dataset_type"),
            "generator_seed": meta.get("seed"), "is_synthetic": bool(ds.is_synthetic)}
    return run_model_comparison(ds.tikg, ds.labels, select_models(models), cfg, topk_cases,
                                ds.reference_paths if use_reference_paths else None, ds.severity, ds.severity_known,
                                info, provenance_info(ds, protocol_frozen), verbose=verbose)


def write_model_comparison(out_dir, res: AblationResult):
    """Same files as ``write_ablation`` (raw fold x seed rows, per-type, timings, paired table with Holm p-values,
    summary, manifest) plus ``model_parameters.csv``."""
    from pathlib import Path
    paths = write_ablation(out_dir, res)
    paths["model_parameters.csv"] = Path(out_dir) / "model_parameters.csv"
    parameter_table(res.runs, res.reference).to_csv(paths["model_parameters.csv"], index=False)
    return paths
