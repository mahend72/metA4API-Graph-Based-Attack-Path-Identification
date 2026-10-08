"""Computational-cost and scalability evaluation (manuscript cost table / scalability discussion).

RESULT INTEGRITY.  Everything here MEASURES the current implementation as it runs.  No runtime or performance value from
manuscript.tex is used, targeted or tuned toward.  Nothing is chosen from test-set performance: every setting (repeats,
budgets, model configurations, the fixed ranker alpha) is fixed in ``ScalabilityConfig`` before any measurement.

What is separated (never only a total runtime):
  * one-time preprocessing   chi_1..chi_20 commuting matrices, MeGiTS adjacency, features, fold preparation
  * algorithmic runtime      per-model training, inference (whole graph and per node), centrality / ranking, path tracing,
                             AttacKG / LADDER fitting and inference
  * model performance        Macro/Micro-F1 of the very runs that were timed, written to a SEPARATE table and never used
                             for any decision
Timing method: ``warmup`` untimed calls, then ``repeats`` timed calls (``time.perf_counter``, ``gc.collect()`` before every
call); median / mean / std / min / max / quartiles are reported and the raw repeats are kept.  Memory is measured in a
SEPARATE extra pass (so tracing never perturbs the timings): peak Python/numpy/scipy allocation (tracemalloc), peak resident
set size sampled from /proc/self/statm (captures torch and C-level allocations), with fallbacks psutil -> resource.ru_maxrss
-> "unavailable" (reported as NaN with the reason).  Limits: the RSS increase reads 0 whenever the allocator re-uses pages
that are already resident (common for small components); torch allocations are visible only through RSS, so the per-component
numbers are lower bounds and ``peak_py_alloc_mb`` is the exact, comparable one for numpy / scipy work.

Graph sizes: structurally comparable synthetic graphs from the existing generator with a fixed seed.  Node-type mix, edge
density (edges per node) and the per-type #Class values of the manuscript profile are kept; only the number of campaigns and
families scales with size (campaign size stays comparable).  Floor: every type has at least ``n_campaigns`` nodes (a generator
requirement), which slightly raises the attack-method share in the 250 / 500 profiles; vulnerabilities absorb the difference.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import threading
import time
import tracemalloc
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from . import cti_baselines as cb
from .ablation import provenance_info
from .datasets import Dataset, load_dataset
from .megits import fold_megits_adjacency, megits_adjacency
from .metagraphs import RelationBlocks, commuting_matrices, default_structures
from .metrics import multilabel_prf
from .model_comparison import MODELS
from .path_tracing import PathConfig, StructureSupport, _neighbours, trace_case
from .ranking import RankerConfig, build_rank_adjacency, centrality, fuse, prioritise, rank_nodes, severity_vector
from .splits import group_stratified_folds
from .supervised_gcn import SupervisedGCNConfig, deterministic, fit_predict, prepare_fold, seed_everything
from .evaluation import gcn_confidence
from .synthetic_tikg import generate, write_dataset
from .synthetic_tikg.profiles import (MANUSCRIPT_EDGE_QUOTAS, MANUSCRIPT_EDGES, MANUSCRIPT_EXPERT_NODES,
                                      MANUSCRIPT_NODE_COUNTS, MANUSCRIPT_NODES, PROFILES, Profile, scale_quotas)
from .tikg_features import FeatureBuilder

DEFAULT_SIZES = (250, 500, 1000, 2000, 3728)
REQUIRED_COMPONENTS = (
    "chi_construction", "megits_adjacency", "megits_similarity_assembly", "megits_fold_adjacency", "features",
    "prepare_fold_gcn", "prepare_fold_hgt", "train_gcn", "train_gat", "train_hgt",
    "inference_gcn", "inference_gat", "inference_hgt", "centrality_ranking", "path_tracing",
    "attackg_fit", "attackg_inference", "ladder_fit", "ladder_inference")
CATEGORY = {"chi_construction": "one_time_preprocessing", "megits_adjacency": "one_time_preprocessing",
            "megits_similarity_assembly": "one_time_preprocessing",
            "megits_fold_adjacency": "one_time_preprocessing", "features": "one_time_preprocessing",
            "prepare_fold_gcn": "one_time_preprocessing", "prepare_fold_hgt": "one_time_preprocessing",
            "path_setup": "one_time_preprocessing"}
TIMING_COLUMNS = ["size", "provenance_source", "component", "category", "repeat", "seconds"]
SUMMARY_COLUMNS = ["size", "component", "category", "n_repeats", "median_s", "mean_s", "std_s", "min_s", "max_s", "q25_s", "q75_s"]


@dataclass(frozen=True)
class ScalabilityConfig:
    sizes: Tuple[int, ...] = DEFAULT_SIZES
    seed: int = 42                              # generator seed (fixed -> identical graphs for identical size)
    fold: int = 0                               # the outer fold used for every measurement (campaign-isolated)
    models: Tuple[str, ...] = ("gcn", "gat", "hgt")
    baselines: Tuple[str, ...] = ("attackg", "ladder")
    warmup: int = 1                             # untimed calls before the fast components
    warmup_slow: int = 1                        # ... before training / baselines (absorbs lazy library initialisation)
    path_cases: Optional[int] = 12              # reference cases traced per size (evenly spaced by sorted id); None = all
    repeats_fast: int = 5                       # chi / MeGiTS / features / inference / ranking
    repeats_slow: int = 3                       # training / path tracing / baselines
    ranker_alpha: float = 0.5                   # fixed (no tuning): alpha only scales the cost-free fusion step
    paths: PathConfig = field(default_factory=PathConfig)
    measure_memory: bool = True
    cache_dir: Optional[str] = None             # where generated datasets are written (default: a temp folder)


# ------------------------------------------------------------------------------------------------ graph-size profiles
def scaled_profile(n_nodes: int) -> Profile:
    """Generator profile with ~n_nodes nodes: manuscript node-type mix and edge density, manuscript #Class per type."""
    if n_nodes == MANUSCRIPT_NODES:
        return PROFILES["manuscript_scale"]
    nodes = scale_quotas(MANUSCRIPT_NODE_COUNTS, n_nodes)
    camps = max(10, round(60 * n_nodes / MANUSCRIPT_NODES))
    for t in list(nodes):
        if nodes[t] < camps:                                         # generator: every campaign needs a node of each type
            nodes["vulnerability"] -= camps - nodes[t]
            nodes[t] = camps
    edges = scale_quotas(MANUSCRIPT_EDGE_QUOTAS, round(MANUSCRIPT_EDGES * n_nodes / MANUSCRIPT_NODES))
    fam = 4 if n_nodes <= 500 else (6 if n_nodes <= 1000 else 8)
    small = {} if n_nodes >= 1000 else {"min_label_support": 2, "min_family_local": 0.85}      # same relaxation as 'dev'
    return Profile(f"scale_{n_nodes}", nodes, edges, n_families=fam, n_campaigns=camps,
                   n_expert=round(MANUSCRIPT_EXPERT_NODES * n_nodes / MANUSCRIPT_NODES), **small)


def make_scaled_dataset(n_nodes: int, seed: int = 42, cache_dir: Optional[str | Path] = None) -> Dataset:
    """Generate (fixed seed) and load the dataset of one size; written under ``cache_dir/<profile>``."""
    import tempfile
    prof = scaled_profile(n_nodes)
    root = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "metA4API_scalability"
    if prof.name == "manuscript_scale":                               # the manuscript profile is named like the on-disk one
        prof = replace(prof, name=f"scale_{n_nodes}")
    ds = generate(prof, seed)
    write_dataset(ds, root / prof.name)
    return load_dataset("synthetic", profile=prof.name, root=root)


def graph_statistics(ds: Dataset, fold=None) -> Dict[str, Any]:
    """Deterministic graph statistics (no timing): identical for an identical (size, seed)."""
    t = ds.tikg
    edges = sorted((t.entities[x.s].id, x.r, t.entities[x.t].id) for x in t.triplets)
    st = {"n_nodes": t.n, "n_edges": len(t.triplets), "edges_per_node": len(t.triplets) / t.n,
          "n_campaigns": int(len({c for c in t.campaigns() if c})), "n_label_columns": int(ds.labels.K),
          "n_labelled": int(ds.labels.labelled.sum()), "n_reference_paths": len(ds.reference_paths),
          "nodes_per_type": {k.value: int(len(v)) for k, v in t.type_nodes.items()},
          "edge_signatures": {s: sum(1 for x in t.triplets if x.sig == s) for s in sorted({x.sig for x in t.triplets})},
          "edge_list_sha256": hashlib.sha256(json.dumps(edges).encode()).hexdigest()}
    if fold is not None:
        st.update(fold_train=int(fold.train.size), fold_val=int(fold.val.size), fold_test=int(fold.test.size))
    return st


def campaign_folds(tikg, labels, preferred: int = 10, val_fraction: float = 0.10):
    """Campaign-isolated folds of the manuscript protocol (10-fold).  The smallest graphs have labels with fewer than 10
    members, which StratifiedGroupKFold rejects; the largest feasible n_splits <= 10 is then used (recorded as
    ``n_splits_used``).  Only the held-out share changes; the isolation logic is the same."""
    err = None
    for n in [preferred, 5, 4, 3, 2]:
        try:
            return group_stratified_folds(tikg, labels, n, 0, val_fraction), n
        except ValueError as e:                       # n_splits larger than the smallest label class
            err = e
    raise err


# ------------------------------------------------------------------------------------------------ timing helpers
def time_call(fn: Callable[[], Any], repeats: int, warmup: int = 1) -> Tuple[List[float], Any]:
    """``warmup`` untimed calls, then ``repeats`` timed calls; returns (seconds per repeat, last result)."""
    out = None
    for _ in range(max(0, warmup)):
        out = fn()
    ts = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return ts, out


def summarise_times(ts: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(ts, dtype=float)
    return {"n_repeats": int(a.size), "median_s": float(np.median(a)), "mean_s": float(a.mean()),
            "std_s": float(a.std(ddof=1)) if a.size > 1 else float("nan"), "min_s": float(a.min()), "max_s": float(a.max()),
            "q25_s": float(np.quantile(a, 0.25)), "q75_s": float(np.quantile(a, 0.75))}


def _rss_proc() -> Optional[float]:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 20
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def _rss_psutil() -> Optional[float]:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:
        return None


def _rss_maxrss() -> Optional[float]:
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return v / 1024.0 if platform.system() != "Darwin" else v / 2 ** 20       # kB on Linux, bytes on macOS
    except Exception:
        return None


RSS_BACKENDS: Dict[str, Callable[[], Optional[float]]] = {"proc_statm": _rss_proc, "psutil": _rss_psutil}


def memory_backend(order: Sequence[str] = ("proc_statm", "psutil")) -> str:
    """First RSS backend that works; 'ru_maxrss' (process high-water mark, a coarser fallback) or 'unavailable'."""
    for name in order:
        fn = RSS_BACKENDS.get(name)
        if fn is not None and fn() is not None:
            return name
    return "ru_maxrss" if _rss_maxrss() is not None else "unavailable"


def measure_memory(fn: Callable[[], Any], backend: Optional[str] = None, interval: float = 0.002) -> Dict[str, Any]:
    """One extra call of ``fn`` under memory observation.  ``peak_py_alloc_mb``: tracemalloc peak (Python / numpy / scipy
    allocations, exact).  ``peak_rss_mb`` / ``rss_delta_mb``: peak resident set size during the call (sampled; includes
    torch and C allocations) and its increase over the start.  Missing backends degrade to NaN with ``memory_backend``."""
    backend = backend or memory_backend()
    read = RSS_BACKENDS.get(backend)
    out: Dict[str, Any] = {"memory_backend": backend, "peak_py_alloc_mb": float("nan"), "peak_rss_mb": float("nan"),
                           "rss_delta_mb": float("nan")}
    gc.collect()
    start = read() if read else (_rss_maxrss() if backend == "ru_maxrss" else None)
    peak = [start if start is not None else float("nan")]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            v = read()
            if v is not None and (np.isnan(peak[0]) or v > peak[0]):
                peak[0] = v
            time.sleep(interval)

    th = threading.Thread(target=sampler, daemon=True) if read else None
    started = tracemalloc.is_tracing()
    if not started:
        tracemalloc.start()
    tracemalloc.reset_peak()
    if th:
        th.start()
    try:
        fn()
    finally:
        _, py_peak = tracemalloc.get_traced_memory()
        if not started:
            tracemalloc.stop()
        if th:
            stop.set()
            th.join()
    out["peak_py_alloc_mb"] = py_peak / 2 ** 20
    if backend == "ru_maxrss":
        end = _rss_maxrss()
        if end is not None and start is not None:
            out.update(peak_rss_mb=end, rss_delta_mb=max(0.0, end - start))
    elif start is not None:
        out.update(peak_rss_mb=float(peak[0]), rss_delta_mb=float(max(0.0, peak[0] - start)))
    return out


# ------------------------------------------------------------------------------------------------ one size
@dataclass
class _Comp:
    name: str
    fn: Callable[[], Any]
    repeats: int
    category: str = "algorithmic_runtime"
    extra: Callable[[Any], Dict[str, Any]] = lambda out: {}
    warmup: Optional[int] = None                # None -> ScalabilityConfig.warmup


def _mb(a) -> float:
    a = sp.csr_matrix(a)
    return float(a.data.nbytes + a.indices.nbytes + a.indptr.nbytes) / 2 ** 20


def measure_size(n_nodes: int, cfg: ScalabilityConfig = ScalabilityConfig(), ds: Optional[Dataset] = None,
                 verbose: bool = False) -> Dict[str, Any]:
    """Every component of one graph size.  Returns raw timings, per-chi table, memory, performance and graph statistics."""
    from .protocol import enforce_launch_guard
    enforce_launch_guard(what="scalability.measure_size", n_nodes=n_nodes, source="synthetic")
    t_gen = time.perf_counter()
    ds = ds or make_scaled_dataset(n_nodes, cfg.seed, cfg.cache_dir)
    t_gen = time.perf_counter() - t_gen
    prov = provenance_info(ds)
    tikg, labels = ds.tikg, ds.labels
    folds, n_splits = campaign_folds(tikg, labels)
    f = folds[min(cfg.fold, len(folds) - 1)]
    test_nodes = np.asarray(f.test_nodes_mask, dtype=bool)
    train_idx = np.flatnonzero(~test_nodes & labels.labelled)
    structures = default_structures()
    rf, rs = cfg.repeats_fast, cfg.repeats_slow
    state: Dict[str, Any] = {}

    # ---- one-time preprocessing -------------------------------------------------------------------------------
    chi_cache: Dict[str, Any] = {}

    def f_chi():
        chi_cache["C"] = commuting_matrices(tikg, structures)
        return chi_cache["C"]

    comps: List[_Comp] = [
        _Comp("chi_construction", f_chi, rf, "one_time_preprocessing",
              lambda C: {"chi_nnz_total": int(sum(m.nnz for m in C.values())), "n_structures": len(C)}),
        _Comp("megits_adjacency", lambda: megits_adjacency(tikg), rf, "one_time_preprocessing",
              lambda r: {"adj_nnz": int(r.adj.nnz), "adj_mb": _mb(r.adj)}),
        _Comp("megits_similarity_assembly", lambda: megits_adjacency(tikg, commuting=chi_cache["C"]), rf, "one_time_preprocessing",
              lambda r: {"adj_nnz": int(r.adj.nnz)}),
        _Comp("megits_fold_adjacency", lambda: fold_megits_adjacency(tikg, test_nodes), rf, "one_time_preprocessing",
              lambda r: {"adj_nnz": int(r.adj.nnz)}),
        _Comp("features", lambda: FeatureBuilder().fit_transform(tikg, np.flatnonzero(f.train)), rf, "one_time_preprocessing",
              lambda x: {"n_features": int(x.shape[1])}),
    ]
    mcfg = {m: MODELS[m].config(SupervisedGCNConfig()) for m in cfg.models}
    ctxs: Dict[str, Any] = {}
    for m in cfg.models:
        kind = "hgt" if MODELS[m].model == "hgt" else "gcn"
        key = f"prepare_fold_{kind}"
        if key not in {c.name for c in comps}:
            comps.append(_Comp(key, (lambda k=kind: ctxs.__setitem__(k, prepare_fold(tikg, labels, f, None, model=k)) or ctxs[k]),
                               rs, "one_time_preprocessing", lambda c: {"in_dim": int(c.x.shape[1])}))
    comps.append(_Comp("path_setup", lambda: (StructureSupport(), _neighbours(tikg)), rf, "one_time_preprocessing"))

    rows: List[Dict[str, Any]] = []
    extras: Dict[str, Dict[str, Any]] = {}
    mem_rows: List[Dict[str, Any]] = []
    fns: Dict[str, Callable] = {}

    def run(c: _Comp):
        ts, out = time_call(c.fn, c.repeats, cfg.warmup if c.warmup is None else c.warmup)
        rows.extend({"size": n_nodes, "provenance_source": ds.source, "component": c.name, "category": c.category,
                     "repeat": i, "seconds": t} for i, t in enumerate(ts))
        extras[c.name] = c.extra(out) if out is not None else {}
        fns[c.name] = c.fn
        if verbose:
            print(f"[{n_nodes}] {c.name}: median {np.median(ts):.4f}s", flush=True)
        return out

    for c in comps:
        run(c)

    # ---- algorithmic runtime: training / inference -----------------------------------------------------------
    fits: Dict[str, Any] = {}
    perf: List[Dict[str, Any]] = []
    for m in cfg.models:
        kind = "hgt" if MODELS[m].model == "hgt" else "gcn"
        ctx = ctxs[kind]

        def train(m=m, ctx=ctx):
            seed_everything(0)
            return fit_predict(ctx, labels, mcfg[m], 0)

        out = run(_Comp(f"train_{m}", train, rs, "algorithmic_runtime",
                        lambda o: {"n_parameters": int(o[0].n_parameters), "epochs_run": int(o[0].epochs_run),
                                   "best_epoch": int(o[0].best_epoch),
                                   "train_time_per_epoch_s": float(np.mean(o[0].epoch_times))}, cfg.warmup_slow))
        fits[m] = out
        res = out[0]
        o2 = run(_Comp(f"inference_{m}", lambda res=res, ctx=ctx: res.predict(ctx.x, ctx.infer_graph), rf))
        med = float(np.median([r["seconds"] for r in rows if r["component"] == f"inference_{m}"]))
        extras[f"inference_{m}"] = {"latency_per_node_us": med / tikg.n * 1e6,
                                    "latency_per_test_node_us": med / max(1, int(f.test.size)) * 1e6,
                                    "mode": "whole-graph forward pass, amortised per node"}
        pred = out[1] > mcfg[m].threshold
        perf.append({"size": n_nodes, "method": m, "provenance_source": ds.source, **{
            k: v for k, v in multilabel_prf(labels.Y, pred, labels.valid, f.test).items() if k in ("macro_f1", "micro_f1")}})

    # ---- ranking ----------------------------------------------------------------------------------------------
    infer_adj = ctxs["gcn"].infer_adj if "gcn" in ctxs else fold_megits_adjacency(tikg, test_nodes).adj
    rcfg = RankerConfig(alpha=cfg.ranker_alpha)
    camp = tikg.campaigns()
    camps = sorted({c for c in camp if c})
    run(_Comp("centrality_ranking", lambda: prioritise(tikg, infer_adj, ds.severity, ds.severity_known, rcfg, alpha=cfg.ranker_alpha),
              rf, "algorithmic_runtime", lambda r: {"a_rank_nnz": int(r.a_rank.nnz)}))
    a_rank = build_rank_adjacency(infer_adj, rcfg.tau)
    run(_Comp("ranking_build_a_rank", lambda: build_rank_adjacency(infer_adj, rcfg.tau), rf))
    run(_Comp("ranking_eigenvector_centrality", lambda: centrality(tikg, a_rank, rcfg.ec_scope, rcfg.ec_norm), rf))
    scores = prioritise(tikg, infer_adj, ds.severity, ds.severity_known, rcfg, alpha=cfg.ranker_alpha)
    run(_Comp("ranking_sort_all_campaigns", lambda: [rank_nodes(tikg, scores.risk, np.flatnonzero(camp == c)) for c in camps], rf))

    # ---- path tracing (timing only; reference paths contribute only the campaigns to trace) ------------------
    support, nbrs = StructureSupport(), _neighbours(tikg)
    if "gcn" in fits:
        conf = gcn_confidence(fits["gcn"][1], labels.valid)
        refs = sorted(ds.reference_paths, key=lambda r: r.case_id)
        if cfg.path_cases and cfg.path_cases < len(refs):
            refs = [refs[i] for i in np.unique(np.linspace(0, len(refs) - 1, cfg.path_cases).round().astype(int))]

        def trace_all():
            return [trace_case(tikg, r.campaign, conf, scores.risk, infer_adj, None, cfg.paths, support, nbrs) for r in refs]

        def trace_info(trs):
            cap = cfg.paths.max_candidates
            return {"n_cases": len(trs), "n_candidates_total": int(sum(len(t.candidates) for t in trs)),
                    "n_seeds_total": int(sum(len(t.seeds) for t in trs)),
                    "n_cases_truncated": int(sum(bool(t.truncated) for t in trs)),
                    "n_cases_candidate_cap_hit": int(sum(t.candidate_cap_hit for t in trs)),
                    "n_cases_search_budget_hit": int(sum(t.search_budget_hit for t in trs)),
                    "n_found_total": int(sum(t.n_found for t in trs)), "n_expansions_total": int(sum(t.n_expansions for t in trs)),
                    "any_budget_hit": bool(any(t.truncated for t in trs)),
                    "budgets": {"max_paths_per_seed": cfg.paths.max_paths_per_seed,
                                "max_expansions_per_seed": cfg.paths.max_expansions_per_seed, "max_candidates": cap}}

        run(_Comp("path_tracing", trace_all, rs, "algorithmic_runtime", trace_info, 0))
        pt = float(np.median([r["seconds"] for r in rows if r["component"] == "path_tracing"]))
        extras["path_tracing"]["per_case_s"] = pt / max(1, len(refs))

    # ---- external baselines -----------------------------------------------------------------------------------
    masks = ctxs["gcn"].masks if "gcn" in ctxs else None
    if masks is not None:
        from .supervised_gcn import visible_labels
        Yv = visible_labels(labels, masks)
        adj_u = cb.undirected_adjacency(tikg)
        for b in cfg.baselines:
            runner = ((lambda: cb.run_attackg(tikg, labels, Yv, masks, test_nodes, adj_u, cb.BASELINES["attackg"].config))
                      if b == "attackg" else (lambda: cb.run_ladder(tikg, labels, Yv, masks, cb.BASELINES["ladder"].config)))
            fit_t, inf_t = [], []
            out = None
            for i in range(cfg.warmup_slow + rs):
                out = runner()
                if i >= cfg.warmup_slow:
                    fit_t.append(out.fit_time_s)
                    inf_t.append(out.inference_time_s)
            for nm, ts in ((f"{b}_fit", fit_t), (f"{b}_inference", inf_t)):
                rows.extend({"size": n_nodes, "provenance_source": ds.source, "component": nm,
                             "category": "algorithmic_runtime", "repeat": i, "seconds": t} for i, t in enumerate(ts))
                extras[nm] = dict(out.diagnostics)
            extras[f"{b}_inference"]["latency_per_node_us"] = float(np.median(inf_t)) / tikg.n * 1e6
            fns[f"{b}_total"] = runner
            pred = out.pred
            perf.append({"size": n_nodes, "method": b, "provenance_source": ds.source, **{
                k: v for k, v in multilabel_prf(labels.Y, pred, labels.valid, f.test).items() if k in ("macro_f1", "micro_f1")}})

    # ---- per-chi construction table ---------------------------------------------------------------------------
    chi_rows = []
    blocks = RelationBlocks(tikg)
    t_blocks, _ = time_call(lambda: [getattr(RelationBlocks(tikg), f"T{i}") for i in range(1, 12)], rf, cfg.warmup)
    for s in structures:
        ts, M = time_call(lambda s=s: s.builder(blocks).tocsr(), rf, cfg.warmup)
        chi_rows.append({"size": n_nodes, "chi": s.id, "kind": s.kind, "median_s": float(np.median(ts)), "nnz": int(M.nnz),
                         "shape": int(M.shape[0]), "density": float(M.nnz / max(1, M.shape[0] ** 2))})
    chi_rows.append({"size": n_nodes, "chi": "relation_blocks_T1_T11", "kind": "shared", "median_s": float(np.median(t_blocks)),
                     "nnz": int(sum(getattr(blocks, f"T{i}").nnz for i in range(1, 12))), "shape": tikg.n, "density": float("nan")})

    # ---- memory (separate pass) -------------------------------------------------------------------------------
    if cfg.measure_memory:
        backend = memory_backend()
        for name, fn in fns.items():
            mem_rows.append({"size": n_nodes, "component": name, **measure_memory(fn, backend)})

    # ---- graph statistics -------------------------------------------------------------------------------------
    full = fold_megits_adjacency(tikg, np.zeros(tikg.n, dtype=bool)).adj
    stats = graph_statistics(ds, f)
    stats.update(megits_nnz=int(full.nnz), megits_sparsity=float(1 - full.nnz / tikg.n ** 2),
                 megits_mean_degree=float(full.nnz / tikg.n), megits_adj_mb=_mb(full),
                 chi_nnz={f"chi{s.id}": int(chi_cache["C"][s.id].nnz) for s in structures} if "C" in chi_cache else {},
                 train_adj_nnz=int(ctxs["gcn"].train_adj.nnz) if "gcn" in ctxs else None,
                 n_splits_used=n_splits, generation_s=t_gen, generator_seed=cfg.seed, profile=scaled_profile(n_nodes).name)
    return {"size": n_nodes, "rows": rows, "extras": extras, "chi": chi_rows, "memory": mem_rows, "performance": perf,
            "stats": stats, "provenance": prov, "source": ds.source}


# ------------------------------------------------------------------------------------------------ whole experiment
@dataclass
class ScalabilityResult:
    timings: pd.DataFrame           # raw: one row per (size, component, repeat)
    summary: pd.DataFrame           # per (size, component): median / mean / std / quartiles + extras
    chi: pd.DataFrame
    memory: pd.DataFrame
    performance: pd.DataFrame       # model performance of the timed runs (NOT used for any decision)
    graphs: pd.DataFrame
    scaling: pd.DataFrame           # log-log slope of median runtime vs nodes per component
    manifest: Dict[str, Any]


def scaling_exponents(summary: pd.DataFrame) -> pd.DataFrame:
    """Empirical exponent b of median_s ~ n^b (least squares on log-log; needs >= 3 sizes and positive times)."""
    rows = []
    for comp, g in summary.groupby("component"):
        g = g[g.median_s > 0].sort_values("size")
        if g["size"].nunique() >= 3:
            b, a = np.polyfit(np.log(g["size"].to_numpy(float)), np.log(g.median_s.to_numpy(float)), 1)
            rows.append({"component": comp, "category": g.category.iloc[0], "exponent": float(b),
                         "n_sizes": int(g["size"].nunique()), "t_at_max_size_s": float(g.median_s.iloc[-1])})
    return pd.DataFrame(rows, columns=["component", "category", "exponent", "n_sizes", "t_at_max_size_s"])


def run_scalability(cfg: ScalabilityConfig = ScalabilityConfig(), verbose: bool = False) -> ScalabilityResult:
    from .protocol import PreflightRequiredError, active_preflight, enforce_launch_guard, exhaustive_path_config
    # A manuscript-scale cost measurement (>= 3,000 nodes) is a manuscript-scale experiment: it needs a passing preflight
    # session, and inside one it must use the frozen exhaustive path search (the 'legacy' truncating budgets are refused).
    enforce_launch_guard(what="run_scalability", n_nodes=max(cfg.sizes) if cfg.sizes else None, source="synthetic")
    if active_preflight() is not None and cfg.paths != exhaustive_path_config():
        raise PreflightRequiredError("run_scalability: refused - the path configuration differs from the preflighted frozen search")
    results = [measure_size(n, cfg, verbose=verbose) for n in cfg.sizes]
    timings = pd.DataFrame([r for x in results for r in x["rows"]], columns=TIMING_COLUMNS)
    summ = []
    for (size, comp, cat), g in timings.groupby(["size", "component", "category"], sort=False):
        ex = next(x["extras"] for x in results if x["size"] == size).get(comp, {})
        summ.append({"size": size, "component": comp, "category": cat, **summarise_times(g.seconds.to_numpy()),
                     **{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in ex.items()}})
    summary = pd.DataFrame(summ)
    chi = pd.DataFrame([r for x in results for r in x["chi"]])
    memory = pd.DataFrame([r for x in results for r in x["memory"]])
    perf = pd.DataFrame([r for x in results for r in x["performance"]])
    graphs = pd.DataFrame([{k: (json.dumps(v, sort_keys=True) if isinstance(v, dict) else v) for k, v in x["stats"].items()}
                           for x in results])
    scaling = scaling_exponents(summary) if len(summary) else pd.DataFrame()
    prov = results[0]["provenance"] if results else {}
    det = {"kind": "scalability", "sizes": list(cfg.sizes), "config": json.loads(json.dumps(asdict(cfg), default=str)),
           "profiles": {str(n): json.loads(json.dumps(asdict(scaled_profile(n)), default=str)) for n in cfg.sizes},
           "graph_statistics": {str(x["size"]): {k: v for k, v in x["stats"].items() if k not in ("generation_s",)} for x in results},
           "provenance": prov,
           "components": sorted(set(timings.component)) if len(timings) else [],
           "timing_method": {"clock": "time.perf_counter", "warmup_untimed_calls": {"fast": cfg.warmup, "training_and_baselines": cfg.warmup_slow, "path_tracing": 0},
                             "repeats_fast": cfg.repeats_fast, "repeats_slow": cfg.repeats_slow, "gc_before_each_call": True,
                             "statistics": "median (primary), mean, std (ddof=1), min, max, quartiles; raw repeats kept"},
           "memory_method": {"backend": memory_backend() if cfg.measure_memory else "disabled",
                             "peak_py_alloc": "tracemalloc peak during one extra call (python/numpy/scipy allocations)",
                             "peak_rss": "RSS sampled every 2 ms during the same extra call (includes torch / C allocations)",
                             "limits": "RSS growth is 0 when already-resident pages are re-used; torch memory only via RSS (lower bound)",
                             "fallbacks": ["psutil", "resource.ru_maxrss (high-water mark)", "unavailable -> NaN"]},
           "separation": {"one_time_preprocessing": sorted(k for k, v in CATEGORY.items() if v == "one_time_preprocessing"),
                          "performance_table": "separate; never used for any decision"},
           "manuscript_values": "not used, tuned against or compared by this code"}
    content = hashlib.sha256(json.dumps(det, sort_keys=True, default=str).encode()).hexdigest()
    from .protocol import attach_preflight
    manifest = {**det, "content_sha256": content, "preflight": attach_preflight({})["preflight"],
                "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                                "platform": platform.platform(), "cpu_count": os.cpu_count(),
                                "torch_threads": torch.get_num_threads(), "processor": platform.processor()}}
    return ScalabilityResult(timings, summary, chi, memory, perf, graphs, scaling, manifest)


def write_scalability(out_dir: str | Path, res: ScalabilityResult) -> Dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {"scalability_timings_raw.csv": res.timings, "scalability_summary.csv": res.summary,
             "scalability_chi.csv": res.chi, "scalability_memory.csv": res.memory,
             "scalability_performance_not_for_selection.csv": res.performance,
             "scalability_graphs.csv": res.graphs, "scalability_scaling.csv": res.scaling}
    paths = {}
    for n, df in files.items():
        paths[n] = out / n
        df.to_csv(paths[n], index=False, float_format="%.10g")
    paths["manifest.json"] = out / "manifest.json"
    paths["manifest.json"].write_text(json.dumps(res.manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return paths
