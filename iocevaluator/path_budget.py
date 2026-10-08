"""Path-tracing budget analysis: per-fold inputs (out-of-fold GCN confidence, prioritisation score R, inference
adjacency), a budget grid, and the budget-sensitivity experiment.

Nothing here selects a budget from path-metric values: ``convergence_report`` reads only the *change* of the metrics
between consecutive budgets and the truncation rate, and the tracing never sees reference paths (they enter only in
``evaluate_case``)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .evaluation import gcn_confidence
from .protocol import (FROZEN_MAX_CANDIDATES, FROZEN_MAX_EDGES, UNCAPPED_SAFETY_EXPANSIONS, UNLIMITED,  # noqa: F401
                       exhaustive_path_config)
from .path_tracing import PathConfig, StructureSupport, _neighbours, evaluate_case, path_config_info, trace_case
from .ranking import RankerConfig, prioritise
from .scalability import campaign_folds
from .supervised_gcn import SupervisedGCNConfig, fit_predict, prepare_fold, seed_everything


@dataclass
class FoldPathInputs:
    """What the tracer needs for the campaigns held out in one outer fold (all fold-safe)."""
    fold: int
    test_campaigns: List[str]
    conf: np.ndarray
    risk: np.ndarray
    infer_adj: sp.csr_matrix


def build_fold_inputs(ds, seed: int = 0, max_folds: Optional[int] = None, alpha: float = 0.5,
                      model_cfg: Optional[SupervisedGCNConfig] = None) -> List[FoldPathInputs]:
    """Leakage-safe path inputs: per outer fold, a supervised GCN (one model seed) trained without the fold's test
    campaigns, its confidence, and the prioritisation score on the inference adjacency (``alpha`` fixed, not tuned on
    path metrics)."""
    _refuse_non_development(ds)
    tikg, labels = ds.tikg, ds.labels
    folds, _ = campaign_folds(tikg, labels)
    camp = tikg.campaigns()
    cfg = model_cfg or SupervisedGCNConfig()
    out = []
    for f in (folds[:max_folds] if max_folds else folds):
        ctx = prepare_fold(tikg, labels, f, None, model=cfg.model)
        _, prob = fit_predict(ctx, labels, cfg, seed)
        scores = prioritise(tikg, ctx.infer_adj, ds.severity, ds.severity_known, RankerConfig(alpha=alpha), alpha=alpha)
        out.append(FoldPathInputs(f.index, sorted({str(c) for c in camp[f.test_nodes_mask]}),
                                  gcn_confidence(prob, labels.valid), scores.risk, sp.csr_matrix(ctx.infer_adj)))
    return out


# ------------------------------------------------------------------------------------------------ budget grid
BASE_BUDGETS = PathConfig()                              # 2,000 paths / 50,000 expansions per seed, 5,000 candidates per case
LEVELS = (("x1", 1), ("x2", 2), ("x5", 5), ("x10", 10), ("uncapped", None))
EDGE_LIMITS = (4, 5, 6)

# Convergence rule, fixed BEFORE the experiment is read.  It looks at how much the outputs change when the budget grows
# and at whether the search was cut short; it never looks at how large Hit@5 or Edge-F1 are.
CONVERGENCE_RULE = {
    "exact": "no search budget (per-seed paths / expansions) was hit in any case: the result equals the uncapped result by construction",
    "converged": "top-5 candidate lists identical to the most complete level in >= 99% of cases, and |dHit@5|, |dEdge-F1| <= 0.01",
    "choice": "the cheapest level (smallest budget) that is 'exact' or 'converged'; if none is, the uncapped level when feasible",
    "min_top5_identical": 0.99, "max_abs_metric_change": 0.01,
}


def level_config(level: str, max_edges: int, base: PathConfig = BASE_BUDGETS) -> PathConfig:
    """PathConfig of a budget level: the candidate cap (5k / 10k / 25k / 50k) and the per-seed path and expansion budgets are
    scaled by the same factor from the current defaults; 'uncapped' removes the caps (expansions keep a recorded safety
    ceiling so a runaway search cannot exhaust memory)."""
    scale = dict(LEVELS)[level]
    if scale is None:
        return replace(base, max_edges=max_edges, max_paths_per_seed=UNLIMITED, max_candidates=UNLIMITED,
                       max_expansions_per_seed=UNCAPPED_SAFETY_EXPANSIONS)
    return replace(base, max_edges=max_edges, max_paths_per_seed=base.max_paths_per_seed * scale,
                   max_expansions_per_seed=base.max_expansions_per_seed * scale, max_candidates=base.max_candidates * scale)


def _case_row(tikg, ref, tr, cfg, seconds, dataset, level, max_edges) -> Dict[str, Any]:
    ev = evaluate_case(tikg, ref, tr, cfg)
    return {"dataset": dataset, "level": level, "max_edges": max_edges, "case_id": ev["case_id"], "campaign": ev["campaign"],
            "ref_edges": ev["n_edges"], "bucket": ev["bucket"], "ref_valid": ev["ref_valid"],
            "ref_within_max_edges": ev["ref_within_max_edges"], "ref_endpoints_seeded": ev["ref_endpoints_seeded"],
            "n_seeds": ev["n_seeds"], "n_scope_nodes": ev["n_scope_nodes"], "n_found": tr.n_found, "n_candidates": ev["n_candidates"],
            "n_expansions": tr.n_expansions, "truncated": ev["truncated"], "search_budget_hit": tr.search_budget_hit,
            "candidate_cap_hit": tr.candidate_cap_hit, "ref_in_candidates": bool(ev["ref_rank"] == ev["ref_rank"]),
            "ref_rank": ev["ref_rank"], "exact_hit": ev["exact_hit"], "edge_f1": ev["edge_f1"], "relation_f1": ev["relation_f1"],
            "top1_path": ev["top1_path"], "top5_paths": ev["top_paths"], "seconds": seconds}


def _refuse_non_development(ds) -> None:
    from .protocol import refuse_at_scale
    refuse_at_scale("path budget sensitivity (diagnostic)", n_nodes=ds.tikg.n, source=ds.source,
                    remedy="a development profile (this study varies non-frozen budgets and is never a primary run)")


def run_budget_sensitivity(ds, inputs: Sequence[FoldPathInputs], levels: Sequence[str] = tuple(n for n, _ in LEVELS),
                           edge_limits: Sequence[int] = EDGE_LIMITS, memory_cases: int = 6, case_ids: Optional[Sequence[str]] = None,
                           verbose: bool = False, base: PathConfig = BASE_BUDGETS, plant_reference: bool = False) -> Dict[str, Any]:
    """Trace every reference case (campaign held out of the fold that produced its inputs) under each (level, max_edges).

    ``plant_reference`` is a STRESS TEST for development data only: the case's reference nodes get maximal confidence and
    priority, so the reference path has a real chance to reach the top 5 and Hit@5 / Edge-F1 become informative about
    truncation.  It uses the reference path as input, so its metrics are never performance results; the rows are marked.

    Timing: one traced call per case after a ``gc.collect()``.  Memory: a separate tracemalloc pass on ``memory_cases``
    evenly spaced cases per setting (tracemalloc slows the call, so these are never used for timing)."""
    import gc
    import time
    _refuse_non_development(ds)
    import tracemalloc
    tikg = ds.tikg
    by_camp = {c: f for f in inputs for c in f.test_campaigns}
    refs = sorted((r for r in ds.reference_paths if r.campaign in by_camp), key=lambda r: r.case_id)
    if case_ids is not None:
        refs = [r for r in refs if r.case_id in set(case_ids)]
    support, nbrs = StructureSupport(), _neighbours(tikg)
    mem_ref = [refs[i] for i in np.unique(np.linspace(0, len(refs) - 1, min(memory_cases, len(refs))).round().astype(int))] if refs else []
    rows, mem = [], []

    def inputs_of(r):
        f = by_camp[r.campaign]
        if not plant_reference:
            return f.conf, f.risk
        idx = [tikg.index[n] for n in r.nodes if n in tikg.index]
        conf, risk = f.conf.copy(), f.risk.copy()
        conf[idx], risk[idx] = 1.0, 1.0
        return conf, risk

    for me in edge_limits:
        for lv in levels:
            cfg = level_config(lv, me, base)
            for r in refs:
                f = by_camp[r.campaign]
                conf, risk = inputs_of(r)
                gc.collect()
                t0 = time.perf_counter()
                tr = trace_case(tikg, r.campaign, conf, risk, f.infer_adj, None, cfg, support, nbrs)
                sec = time.perf_counter() - t0
                rows.append({**_case_row(tikg, r, tr, cfg, sec, ds.source, lv, me), "reference_planted_stress_test": plant_reference})
            for r in mem_ref:
                f = by_camp[r.campaign]
                conf, risk = inputs_of(r)
                gc.collect()
                tracemalloc.start()
                tr = trace_case(tikg, r.campaign, conf, risk, f.infer_adj, None, cfg, support, nbrs)
                peak = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
                mem.append({"dataset": ds.source, "level": lv, "max_edges": me, "case_id": r.case_id,
                            "peak_py_alloc_mb": peak / 2 ** 20, "n_found": tr.n_found})
            if verbose:
                sub = [x for x in rows if x["level"] == lv and x["max_edges"] == me]
                print(f"[{ds.source}] max_edges={me} {lv}: median {np.median([x['seconds'] for x in sub]):.3f}s/case, "
                      f"search_budget_hit {np.mean([x['search_budget_hit'] for x in sub]):.2f}", flush=True)
    return {"cases": pd.DataFrame(rows), "memory": pd.DataFrame(mem)}


def summarise_budget(cases: pd.DataFrame, memory: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, max_edges, level): truncation rates, recall, Hit@5, Edge-F1, candidates, runtime, memory."""
    out = []
    for (d, me, lv), g in cases.groupby(["dataset", "max_edges", "level"], sort=False):
        inr = g[g.ref_within_max_edges & g.ref_valid]
        m = memory[(memory.dataset == d) & (memory.max_edges == me) & (memory.level == lv)]
        out.append({"dataset": d, "max_edges": me, "level": lv, "n_cases": len(g),
                    "pct_truncated_any": 100 * g.truncated.mean(), "pct_search_budget_hit": 100 * g.search_budget_hit.mean(),
                    "pct_candidate_cap_hit": 100 * g.candidate_cap_hit.mean(),
                    "ref_recall_in_candidates": float(g.ref_in_candidates.mean()),
                    "ref_recall_in_candidates_within_limit": float(inr.ref_in_candidates.mean()) if len(inr) else float("nan"),
                    "exact_hit5": float(g.exact_hit.mean()), "edge_f1": float(g.edge_f1.mean()), "relation_f1": float(g.relation_f1.mean()),
                    "exact_hit5_within_limit": float(inr.exact_hit.mean()) if len(inr) else float("nan"),
                    "edge_f1_within_limit": float(inr.edge_f1.mean()) if len(inr) else float("nan"),
                    "n_cases_ref_beyond_limit": int((~g.ref_within_max_edges).sum()),
                    "mean_candidates": float(g.n_candidates.mean()), "median_candidates": float(g.n_candidates.median()),
                    "mean_found_before_cap": float(g.n_found.mean()), "max_found_before_cap": int(g.n_found.max()),
                    "median_seconds_per_case": float(g.seconds.median()), "mean_seconds_per_case": float(g.seconds.mean()),
                    "max_seconds_per_case": float(g.seconds.max()),
                    "median_peak_py_alloc_mb": float(m.peak_py_alloc_mb.median()) if len(m) else float("nan"),
                    "max_peak_py_alloc_mb": float(m.peak_py_alloc_mb.max()) if len(m) else float("nan"),
                    "exhaustive": bool(not g.search_budget_hit.any())})
    return pd.DataFrame(out)


def convergence_report(cases: pd.DataFrame, summary: pd.DataFrame, level_order: Sequence[str] = tuple(n for n, _ in LEVELS)
                       ) -> pd.DataFrame:
    """Change of the path metrics between each level and the most complete (last) level of the same dataset / max_edges,
    plus the rule of ``CONVERGENCE_RULE``.  The rule uses changes and truncation only, never the metric level."""
    out = []
    for (d, me), g in cases.groupby(["dataset", "max_edges"], sort=False):
        present = [lv for lv in level_order if lv in set(g.level)]
        ref_lv = present[-1]
        full = g[g.level == ref_lv].set_index("case_id")
        for lv in present:
            cur = g[g.level == lv].set_index("case_id")
            same5 = float((cur.top5_paths == full.loc[cur.index, "top5_paths"]).mean())
            same1 = float((cur.top1_path == full.loc[cur.index, "top1_path"]).mean())
            dh = float(cur.exact_hit.mean() - full.exact_hit.mean())
            de = float(cur.edge_f1.mean() - full.edge_f1.mean())
            s = summary[(summary.dataset == d) & (summary.max_edges == me) & (summary.level == lv)].iloc[0]
            exact = bool(s.exhaustive)
            conv = same5 >= CONVERGENCE_RULE["min_top5_identical"] and max(abs(dh), abs(de)) <= CONVERGENCE_RULE["max_abs_metric_change"]
            out.append({"dataset": d, "max_edges": me, "level": lv, "compared_to": ref_lv, "top5_identical_share": same5,
                        "top1_identical_share": same1, "delta_exact_hit5": dh, "delta_edge_f1": de,
                        "n_cases_top5_changed": int(round((1 - same5) * len(cur))), "exact": exact, "converged": bool(conv),
                        "meets_rule": bool(exact or conv),
                        "median_seconds_per_case": float(s.median_seconds_per_case)})
    return pd.DataFrame(out)


def choose_level(conv: pd.DataFrame, level_order: Sequence[str] = tuple(n for n, _ in LEVELS)) -> Dict[str, Any]:
    """Cheapest level meeting the rule for EVERY (dataset, max_edges) setting; falls back to the most complete level."""
    for lv in level_order:
        rows = conv[conv.level == lv]
        if len(rows) and rows.meets_rule.all():
            return {"level": lv, "reason": "meets CONVERGENCE_RULE in every dataset and path-length setting"}
    return {"level": level_order[-1], "reason": "no capped level met the rule everywhere; use the most complete level"}


def write_budget(res: Dict[str, Any], out_dir, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    import json
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases, memory = res["cases"], res["memory"]
    summary = summarise_budget(cases, memory)
    conv = convergence_report(cases, summary)
    files = {"path_budget_cases.csv": cases, "path_budget_memory.csv": memory, "path_budget_summary.csv": summary,
             "path_budget_convergence.csv": conv}
    paths = {}
    for n, df in files.items():
        paths[n] = out / n
        df.to_csv(paths[n], index=False, float_format="%.10g")
    manifest = {"rule": CONVERGENCE_RULE, "levels": {n: (None if s is None else {"scale": s}) for n, s in LEVELS},
                "base_budgets": path_config_info(BASE_BUDGETS), "uncapped_safety_expansions_per_seed": UNCAPPED_SAFETY_EXPANSIONS,
                "edge_limits": list(EDGE_LIMITS), "chosen": choose_level(conv),
                "provenance": "development data only; not a reproduction of any manuscript value",
                "preflight": {"performed": False, "run_class": "diagnostic_sensitivity", "reportable": False, "not_reportable": True,
                              "note": "varies non-frozen path budgets by design; refused at manuscript scale; never a primary configuration"},
                **(extra or {})}
    paths["manifest.json"] = out / "path_budget_manifest.json"
    paths["manifest.json"].write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return paths
