"""Cost / scalability framework: schema, size profiles, determinism, components, memory fallback, numerical preservation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from iocevaluator import scalability as sc
from iocevaluator.megits import fold_megits_adjacency, megits_adjacency
from iocevaluator.metagraphs import commuting_matrices, default_structures
from iocevaluator.scalability import (REQUIRED_COMPONENTS, SUMMARY_COLUMNS, TIMING_COLUMNS, ScalabilityConfig, graph_statistics,
                                      make_scaled_dataset, measure_memory, memory_backend, scaled_profile, scaling_exponents,
                                      summarise_times, time_call, write_scalability)
from iocevaluator.synthetic_tikg.profiles import (MANUSCRIPT_CLASS_COUNTS, MANUSCRIPT_EDGES, MANUSCRIPT_NODES)

SMALL = ScalabilityConfig(sizes=(250,), repeats_fast=2, repeats_slow=1, warmup=0, warmup_slow=0, path_cases=3)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    cfg = ScalabilityConfig(sizes=SMALL.sizes, repeats_fast=2, repeats_slow=1, warmup=0, warmup_slow=0, path_cases=3,
                            cache_dir=str(tmp_path_factory.mktemp("sc_cache")))
    return sc.run_scalability(cfg), cfg


# ------------------------------------------------------------------------------------------------ size profiles
def test_profiles_have_requested_size_and_comparable_structure():
    sizes = (250, 500, 1000, 2000, 3728)
    profs = [scaled_profile(n) for n in sizes]
    assert [p.n_nodes for p in profs] == list(sizes)
    assert profs[-1].n_edges == MANUSCRIPT_EDGES and profs[-1].n_nodes == MANUSCRIPT_NODES
    density = [p.n_edges / p.n_nodes for p in profs]
    assert max(density) - min(density) < 0.02 * np.mean(density)            # edge density is NOT changed with size
    for p in profs:
        assert p.class_counts == MANUSCRIPT_CLASS_COUNTS                      # label space is not changed with size
        assert all(v >= p.n_campaigns for v in p.node_counts.values())        # generator requirement
    camps = [p.n_campaigns for p in profs]
    assert camps == sorted(camps) and camps[0] >= 10
    # type mix is the manuscript mix, except for the documented floor in the smallest profiles
    big = profs[-1].node_counts
    for p in profs[2:]:
        for t, v in p.node_counts.items():
            assert abs(v / p.n_nodes - big[t] / MANUSCRIPT_NODES) < 0.01


@pytest.mark.parametrize("n", [250, 500])
def test_increasing_size_generation(n, tmp_path):
    ds = make_scaled_dataset(n, 42, tmp_path)
    assert ds.tikg.n == n and ds.source == "synthetic"
    assert len(ds.tikg.triplets) == scaled_profile(n).n_edges
    assert ds.labels.K == sum(MANUSCRIPT_CLASS_COUNTS.values()) or ds.labels.K > 0


def test_generated_sizes_increase(tmp_path):
    sizes = [make_scaled_dataset(n, 42, tmp_path).tikg.n for n in (250, 500, 1000)]
    assert sizes == [250, 500, 1000]


def test_graph_statistics_are_deterministic(tmp_path):
    a = graph_statistics(make_scaled_dataset(250, 42, tmp_path / "a"))
    b = graph_statistics(make_scaled_dataset(250, 42, tmp_path / "b"))
    c = graph_statistics(make_scaled_dataset(250, 7, tmp_path / "c"))
    assert a == b
    assert a["edge_list_sha256"] != c["edge_list_sha256"]                    # the seed matters, the run does not
    assert a["n_nodes"] == 250 and a["edges_per_node"] > 2


# ------------------------------------------------------------------------------------------------ timing schema / components
def test_timing_output_schema(result):
    res, cfg = result
    assert list(res.timings.columns) == TIMING_COLUMNS
    assert set(SUMMARY_COLUMNS) <= set(res.summary.columns)
    assert np.isfinite(res.timings.seconds).all() and (res.timings.seconds > 0).all()
    assert (res.timings.provenance_source == "synthetic").all()
    assert set(res.timings.category) <= {"one_time_preprocessing", "algorithmic_runtime"}
    n = res.timings.groupby("component").repeat.nunique()
    assert n["chi_construction"] == cfg.repeats_fast and n["train_gcn"] == cfg.repeats_slow
    s = res.summary.set_index("component")
    assert (s.min_s <= s.median_s).all() and (s.median_s <= s.max_s).all() and (s.q25_s <= s.q75_s).all()


def test_every_required_component_is_timed(result):
    res, _ = result
    missing = set(REQUIRED_COMPONENTS) - set(res.timings.component)
    assert not missing, missing
    for extra in ("ranking_eigenvector_centrality", "ranking_build_a_rank", "inference_gcn"):
        assert extra in set(res.timings.component)
    # preprocessing is separated from algorithmic runtime
    cat = res.summary.drop_duplicates("component").set_index("component").category
    for c in ("chi_construction", "megits_adjacency", "features", "prepare_fold_gcn"):
        assert cat[c] == "one_time_preprocessing"
    for c in ("train_gcn", "train_gat", "train_hgt", "path_tracing", "attackg_fit", "ladder_inference"):
        assert cat[c] == "algorithmic_runtime"
    assert len(res.chi[res.chi.chi != "relation_blocks_T1_T11"]) == 20          # one row per chi structure


def test_model_details_latency_and_budget_flags(result):
    res, cfg = result
    s = res.summary.set_index("component")
    for m in ("gcn", "gat", "hgt"):
        assert s.loc[f"train_{m}", "n_parameters"] > 0 and s.loc[f"train_{m}", "epochs_run"] >= 1
        assert s.loc[f"inference_{m}", "latency_per_node_us"] > 0
    assert s.loc["attackg_fit", "n_templates"] > 0
    pt = s.loc["path_tracing"]
    assert pt.n_cases == cfg.path_cases and pt.n_cases_truncated >= 0 and pt.n_cases_candidate_cap_hit >= 0
    assert pt.any_budget_hit in (True, False, np.True_, np.False_)
    assert json.loads(pt.budgets)["max_candidates"] == cfg.paths.max_candidates


def test_performance_is_kept_separate_from_runtime(result):
    res, _ = result
    assert set(res.performance.method) == {"gcn", "gat", "hgt", "attackg", "ladder"}
    assert {"macro_f1", "micro_f1"} <= set(res.performance.columns)
    assert not {"macro_f1", "micro_f1"} & set(res.timings.columns) | set(res.summary.columns) & {"macro_f1"}


def test_graph_table_and_scaling_fields(result):
    res, _ = result
    g = res.graphs.iloc[0]
    for k in ("n_nodes", "n_edges", "megits_nnz", "megits_sparsity", "train_adj_nnz", "n_splits_used", "profile"):
        assert k in g.index
    assert g.n_nodes == 250 and 0 < g.megits_sparsity < 1


def test_memory_table_and_manifest(result):
    res, cfg = result
    assert set(res.memory.component) >= {"chi_construction", "train_gcn", "path_tracing"}
    assert (res.memory.peak_py_alloc_mb > 0).all()
    assert res.memory.memory_backend.isin(["proc_statm", "psutil", "ru_maxrss", "unavailable"]).all()
    m = res.manifest
    assert m["kind"] == "scalability" and m["manuscript_values"].startswith("not used")
    assert "development/testing" in m["provenance"]["status"]
    assert m["separation"]["performance_table"].startswith("separate")
    assert "environment" in m and "content_sha256" in m


def test_written_outputs(result, tmp_path):
    res, _ = result
    paths = write_scalability(tmp_path, res)
    for name in ("scalability_timings_raw.csv", "scalability_summary.csv", "scalability_chi.csv", "scalability_memory.csv",
                 "scalability_performance_not_for_selection.csv", "scalability_graphs.csv", "manifest.json"):
        assert paths[name].exists()
    assert list(pd.read_csv(paths["scalability_timings_raw.csv"]).columns) == TIMING_COLUMNS


def test_manifest_hash_excludes_timings(result, tmp_path):
    res, _ = result
    again = json.loads(json.dumps(res.manifest, default=str))
    again.pop("content_sha256"); again.pop("environment"); again.pop("preflight")      # run-specific records live outside the hash
    import hashlib
    assert hashlib.sha256(json.dumps(again, sort_keys=True, default=str).encode()).hexdigest() == res.manifest["content_sha256"]


# ------------------------------------------------------------------------------------------------ helpers
def test_time_call_counts_and_summary_statistics():
    calls = []
    ts, out = time_call(lambda: calls.append(1) or len(calls), repeats=4, warmup=2)
    assert len(calls) == 6 and len(ts) == 4 and out == 6 and all(t >= 0 for t in ts)
    s = summarise_times([1.0, 2.0, 3.0, 4.0])
    assert s["median_s"] == 2.5 and s["mean_s"] == 2.5 and s["min_s"] == 1.0 and s["max_s"] == 4.0
    assert s["std_s"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1)) and s["n_repeats"] == 4
    assert np.isnan(summarise_times([1.0])["std_s"])


def test_scaling_exponent_recovers_known_power_law():
    rows = [{"size": n, "component": "quad", "category": "x", "median_s": 1e-6 * n ** 2} for n in (250, 500, 1000, 2000)]
    rows += [{"size": n, "component": "lin", "category": "x", "median_s": 1e-4 * n} for n in (250, 500, 1000, 2000)]
    ex = scaling_exponents(pd.DataFrame(rows)).set_index("component").exponent
    assert ex["quad"] == pytest.approx(2.0, abs=1e-6) and ex["lin"] == pytest.approx(1.0, abs=1e-6)
    assert scaling_exponents(pd.DataFrame(rows[:2])).empty                     # < 3 sizes: no exponent


# ------------------------------------------------------------------------------------------------ memory fallback
def test_memory_measures_a_known_allocation():
    r = measure_memory(lambda: np.ones(8_000_000))                             # 64 MB
    assert r["peak_py_alloc_mb"] >= 60 and r["peak_py_alloc_mb"] < 200
    if r["memory_backend"] != "unavailable":
        assert r["peak_rss_mb"] > 0 and r["rss_delta_mb"] >= 0


def test_memory_fallback_chain(monkeypatch):
    monkeypatch.setitem(sc.RSS_BACKENDS, "proc_statm", lambda: None)
    monkeypatch.setitem(sc.RSS_BACKENDS, "psutil", lambda: None)
    assert memory_backend() in ("ru_maxrss", "unavailable")
    r = measure_memory(lambda: np.ones(1_000_000))
    assert r["memory_backend"] in ("ru_maxrss", "unavailable") and r["peak_py_alloc_mb"] > 0
    monkeypatch.setattr(sc, "_rss_maxrss", lambda: None)
    assert memory_backend() == "unavailable"
    r = measure_memory(lambda: np.ones(1_000_000))
    assert r["memory_backend"] == "unavailable" and np.isnan(r["peak_rss_mb"]) and np.isnan(r["rss_delta_mb"])
    assert r["peak_py_alloc_mb"] > 0                                           # tracemalloc still works


def test_psutil_backend_used_when_proc_is_missing(monkeypatch):
    monkeypatch.setitem(sc.RSS_BACKENDS, "proc_statm", lambda: None)
    monkeypatch.setitem(sc.RSS_BACKENDS, "psutil", lambda: 123.0)
    assert memory_backend() == "psutil"
    assert memory_backend(order=("nonexistent",)) in ("ru_maxrss", "unavailable")


# ------------------------------------------------------------------------------------------------ numerical preservation
def _same(a: sp.spmatrix, b: sp.spmatrix) -> bool:
    return a.shape == b.shape and (a != b).nnz == 0


def test_precomputed_commuting_matrices_give_identical_megits(tmp_path):
    ds = make_scaled_dataset(250, 42, tmp_path)
    t = ds.tikg
    C = commuting_matrices(t, default_structures())
    assert _same(megits_adjacency(t).adj, megits_adjacency(t, commuting=C).adj)
    test_nodes = np.asarray(group_first_fold_mask(ds))
    Cx = commuting_matrices(t, default_structures(), exclude=test_nodes)
    assert _same(megits_adjacency(t, exclude=test_nodes).adj, megits_adjacency(t, exclude=test_nodes, commuting=Cx).adj)
    assert _same(megits_adjacency(t, binary=True).adj, megits_adjacency(t, binary=True, commuting=C).adj)


def group_first_fold_mask(ds):
    folds, _ = sc.campaign_folds(ds.tikg, ds.labels)
    return folds[0].test_nodes_mask


def test_measurement_does_not_change_the_measured_outputs(tmp_path):
    """Timing / memory wrappers only call the function: the result is bit-identical to a direct call."""
    ds = make_scaled_dataset(250, 42, tmp_path)
    direct = fold_megits_adjacency(ds.tikg, np.zeros(ds.tikg.n, bool)).adj
    holder = {}
    time_call(lambda: holder.__setitem__("a", fold_megits_adjacency(ds.tikg, np.zeros(ds.tikg.n, bool)).adj), 2, 1)
    measure_memory(lambda: holder.__setitem__("b", fold_megits_adjacency(ds.tikg, np.zeros(ds.tikg.n, bool)).adj))
    assert _same(direct, holder["a"]) and _same(direct, holder["b"])


def test_no_manuscript_runtime_or_performance_values_in_code():
    src = Path(sc.__file__).read_text(encoding="utf-8")
    for v in ("2.31 s", "5.84 min", "18.6 ms", "2.10 GB", "1.94 s", "4.91 min", "15.2 ms", "1.82 GB", "0.7551", "0.7696"):
        assert v not in src


def test_small_end_to_end_smoke(result):
    res, cfg = result
    assert len(res.timings) > 20 and len(res.graphs) == 1 and len(res.chi) == 21
    assert res.scaling.empty                                                   # one size -> no exponent, no failure
