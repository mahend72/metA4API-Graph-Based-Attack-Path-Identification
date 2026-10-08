"""Output equivalence of the optimised path tracer against the frozen original (tests/reference_path_tracing.py), and the
budget-sensitivity framework.  Equivalence is exact: same candidates, same order, same float scores (bit-identical),
same Exact-Path Hit@5 and Edge-F1.  Synthetic data only; nothing here is a manuscript result."""
import math
import sys

import numpy as np
import pandas as pd
import pytest

from iocevaluator import path_tracing as N
from iocevaluator.datasets import load_dataset
from iocevaluator.path_budget import (CONVERGENCE_RULE, LEVELS, FoldPathInputs, choose_level, convergence_report,
                                      level_config, run_budget_sensitivity, summarise_budget, write_budget)
from tests import reference_path_tracing as R
from tests.test_path_tracing import ENTS, TRIPS, toy


# ------------------------------------------------------------------------------------------------ helpers
def _cands(res):
    return [(p.nodes, p.relations, p.directions, p.structures, p.score, p.components) for p in res.candidates]


def _same(a, b):
    assert a.truncated == b.truncated
    assert _cands(a) == _cands(b)                                    # exact float equality, same order
    assert list(a.seeds) == list(b.seeds)


def _both(g, campaign, conf, risk, kw, adj=None):
    ra = R.trace_case(g, campaign, conf, risk, adj, None, R.PathConfig(**kw))
    na = N.trace_case(g, campaign, conf, risk, adj, None, N.PathConfig(**kw))
    return ra, na


@pytest.fixture(scope="module")
def dev():
    ds = load_dataset("synthetic", profile="dev")
    rng = np.random.RandomState(7)
    n = ds.tikg.n
    return {"ds": ds, "conf64": rng.uniform(0.3, 1.0, n), "conf32": rng.uniform(0.3, 1.0, n).astype(np.float32),
            "risk": rng.uniform(0.0, 1.0, n), "risk_ties": rng.randint(0, 4, n) / 4.0,
            "risk_nan": np.where(rng.rand(n) < 0.1, np.nan, rng.uniform(0, 1, n))}


CAMPAIGNS = None


def _campaigns(ds, k=3):
    return sorted({c for c in ds.tikg.campaigns() if c})[:k]


UNCAPPED = dict(max_paths_per_seed=10 ** 9, max_expansions_per_seed=10 ** 9, max_candidates=10 ** 9)


# ------------------------------------------------------------------------------------------------ toy graphs
@pytest.mark.parametrize("kw", [dict(max_edges=2), dict(max_edges=4, **UNCAPPED), dict(max_edges=3, support="template"),
                                dict(max_edges=3, max_candidates=2), dict(max_edges=4, max_expansions_per_seed=7)])
def test_toy_graph_identical(kw):
    g = toy()
    conf = np.full(g.n, 0.9)
    risk = np.linspace(0.1, 0.9, g.n)
    a, b = _both(g, "k", conf, risk, kw)
    _same(a, b)


def test_toy_hit_and_edge_f1_identical():
    from iocevaluator.labels import ReferencePath
    g = toy()
    conf, risk = np.full(g.n, 0.9), np.linspace(0.1, 0.9, g.n)
    ref = ReferencePath("c1", "k", ["ta1", "v1", "p1"], ["exploits", "affected_by"])
    a, b = _both(g, "k", conf, risk, dict(max_edges=3, **UNCAPPED))
    ea, eb = R.evaluate_case(g, ref, a, R.PathConfig(max_edges=3)), N.evaluate_case(g, ref, b, N.PathConfig(max_edges=3))
    assert all(ea[k] == eb[k] or (ea[k] != ea[k] and eb[k] != eb[k]) for k in ea)


# ------------------------------------------------------------------------------------------------ dev graph
@pytest.mark.parametrize("kw", [
    dict(max_edges=4, **UNCAPPED),                                                       # uncapped search
    dict(max_edges=3, support="template", **UNCAPPED),
    dict(),                                                                              # default budgets
    dict(max_paths_per_seed=25, max_expansions_per_seed=400, max_candidates=40),         # truncated search: accounting
    dict(max_edges=4, priority_top_n=6, weights=(("priority", 1.0), ("confidence", 2.0), ("structure_weight", 0.5),
                                                 ("coherence", 1.0)), **UNCAPPED),
])
@pytest.mark.parametrize("which", ["conf64", "conf32"])
def test_dev_candidates_scores_ranking_identical(dev, kw, which):
    ds = dev["ds"]
    from iocevaluator.megits import megits_adjacency
    adj = megits_adjacency(ds.tikg).adj if "weights" in kw else None
    for c in _campaigns(ds, 2):
        a, b = _both(ds.tikg, c, dev[which], dev["risk"], kw, adj)
        _same(a, b)


@pytest.mark.parametrize("risk", ["risk_ties", "risk_nan"])
def test_dev_ties_and_nan_risk_identical(dev, risk):
    ds = dev["ds"]
    for c in _campaigns(ds, 2):
        a, b = _both(ds.tikg, c, dev["conf64"], dev[risk], dict(max_edges=4, max_candidates=300, **{k: v for k, v in UNCAPPED.items() if k != "max_candidates"}))
        _same(a, b)


def test_dev_hit5_and_edge_f1_identical(dev):
    ds = dev["ds"]
    camps = set(_campaigns(ds, 4))
    refs = [r for r in ds.reference_paths if r.campaign in camps]
    assert refs
    for kw in (dict(max_edges=5, **UNCAPPED), dict()):
        for r in refs:
            a, b = _both(ds.tikg, r.campaign, dev["conf64"], dev["risk"], kw)
            ea, eb = R.evaluate_case(ds.tikg, r, a, R.PathConfig(**kw)), N.evaluate_case(ds.tikg, r, b, N.PathConfig(**kw))
            assert set(ea) <= set(eb) and set(eb) - set(ea) == {"n_found", "search_budget_hit", "candidate_cap_hit"}
            for k in ea:
                assert ea[k] == eb[k] or (ea[k] != ea[k] and eb[k] != eb[k]), k


def test_candidate_cap_never_changes_top_ranks(dev):
    """The per-case cap truncates after ranking: the first ``max_candidates`` candidates are the same list."""
    ds = dev["ds"]
    c = _campaigns(ds, 1)[0]
    full = N.trace_case(ds.tikg, c, dev["conf64"], dev["risk"], None, None, N.PathConfig(max_edges=4, **UNCAPPED))
    cap = N.trace_case(ds.tikg, c, dev["conf64"], dev["risk"], None, None, N.PathConfig(max_edges=4, **{**UNCAPPED, "max_candidates": 50}))
    assert full.n_found == cap.n_found > 50 and cap.candidate_cap_hit and not full.candidate_cap_hit
    assert _cands(cap) == _cands(full)[:50]


def test_search_diagnostics(dev):
    ds = dev["ds"]
    c = _campaigns(ds, 1)[0]
    tight = N.trace_case(ds.tikg, c, dev["conf64"], dev["risk"], None, None, N.PathConfig(max_paths_per_seed=3))
    free = N.trace_case(ds.tikg, c, dev["conf64"], dev["risk"], None, None, N.PathConfig(max_edges=4, **UNCAPPED))
    assert tight.search_budget_hit and tight.n_seeds_path_budget_hit > 0
    assert not free.search_budget_hit and free.n_expansions > 0 and free.n_found == len(free.candidates)


# ------------------------------------------------------------------------------------------------ arithmetic building blocks
def test_py_sum_matches_builtin_sum_bitwise():
    rng = np.random.RandomState(0)
    terms = [rng.rand(5000) * s for s in (1.0, 3.0, 0.31, 0.0)]
    got = N._py_sum(terms)
    want = np.array([sum(float(t[i]) for t in terms) for i in range(5000)])
    assert (got == want).all()


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("n_cols", [1, 2, 3, 5, 7])
def test_seq_mean_matches_np_mean_bitwise(dtype, n_cols):
    rng = np.random.RandomState(n_cols)
    M = rng.rand(3000, n_cols).astype(dtype)
    got = N._seq_mean([M[:, j] for j in range(n_cols)])
    want = np.array([np.mean(M[i]) for i in range(len(M))])
    assert (got.astype(np.float64) == want.astype(np.float64)).all()


# ------------------------------------------------------------------------------------------------ budget framework
def _inputs(ds):
    rng = np.random.RandomState(3)
    n = ds.tikg.n
    camps = sorted({c for c in ds.tikg.campaigns() if c})
    return [FoldPathInputs(0, camps, rng.uniform(0.3, 1.0, n), rng.uniform(0, 1, n), None)]


def test_level_configs_scale_together():
    base = N.PathConfig()
    x2 = level_config("x2", 5)
    assert (x2.max_candidates, x2.max_paths_per_seed, x2.max_expansions_per_seed, x2.max_edges) == (
        2 * base.max_candidates, 2 * base.max_paths_per_seed, 2 * base.max_expansions_per_seed, 5)
    assert level_config("x1", 6).max_candidates == 5000 and level_config("x10", 4).max_candidates == 50000
    assert level_config("uncapped", 4).max_candidates >= 10 ** 9
    assert [n for n, _ in LEVELS] == ["x1", "x2", "x5", "x10", "uncapped"]


@pytest.fixture(scope="module")
def tiny_budget(dev):
    ds = dev["ds"]
    camps = _campaigns(ds, 2)
    ids = [r.case_id for r in ds.reference_paths if r.campaign in camps][:6]
    res = run_budget_sensitivity(ds, _inputs(ds), ("x1", "uncapped"), (3, 4), memory_cases=2, case_ids=ids)
    return ds, res


def test_budget_outputs_schema_and_levels(tiny_budget):
    ds, res = tiny_budget
    cases, mem = res["cases"], res["memory"]
    assert set(cases.level) == {"x1", "uncapped"} and set(cases.max_edges) == {3, 4}
    need = {"n_found", "n_candidates", "truncated", "search_budget_hit", "candidate_cap_hit", "ref_in_candidates",
            "exact_hit", "edge_f1", "seconds", "top5_paths", "ref_within_max_edges"}
    assert need <= set(cases.columns) and len(cases) == 2 * 2 * len(set(cases.case_id))
    assert (mem.peak_py_alloc_mb > 0).all()
    s = summarise_budget(cases, mem)
    assert {"pct_truncated_any", "ref_recall_in_candidates", "exact_hit5", "edge_f1", "mean_candidates",
            "median_seconds_per_case", "median_peak_py_alloc_mb"} <= set(s.columns) and len(s) == 4


def test_uncapped_level_is_not_truncated_and_a_superset(tiny_budget):
    _, res = tiny_budget
    c = res["cases"]
    unc, x1 = c[c.level == "uncapped"], c[c.level == "x1"]
    assert not unc.truncated.any() and not unc.search_budget_hit.any()
    assert (unc.n_found.values >= x1.n_found.values).all()


def test_convergence_report_uses_changes_not_levels():
    """Rule input is the change vs the most complete level: shifting every metric by the same amount changes nothing."""
    def frame(h):
        rows = []
        for lv in ("x1", "uncapped"):
            for i in range(10):
                rows.append({"dataset": "d", "max_edges": 4, "level": lv, "case_id": f"c{i}", "top5_paths": "p", "top1_path": "p",
                             "exact_hit": h, "edge_f1": h})
        return pd.DataFrame(rows)
    summ = pd.DataFrame([{"dataset": "d", "max_edges": 4, "level": lv, "exhaustive": False, "median_seconds_per_case": 1.0}
                         for lv in ("x1", "uncapped")])
    for h in (0.1, 0.9):
        conv = convergence_report(frame(h), summ)
        assert conv.converged.all() and conv.meets_rule.all()
        assert choose_level(conv)["level"] == "x1"


def test_choose_level_falls_back_when_nothing_converges():
    conv = pd.DataFrame([{"level": lv, "meets_rule": lv == "uncapped"} for lv in ("x1", "x2", "x5", "x10", "uncapped")])
    assert choose_level(conv)["level"] == "uncapped"


def test_write_budget_files_and_manifest(tiny_budget, tmp_path):
    import json
    _, res = tiny_budget
    paths = write_budget(res, tmp_path, {"source": "synthetic"})
    for n in ("path_budget_cases.csv", "path_budget_summary.csv", "path_budget_convergence.csv", "path_budget_memory.csv"):
        assert (tmp_path / n).exists()
    m = json.loads(paths["manifest.json"].read_text())
    assert m["rule"]["min_top5_identical"] == CONVERGENCE_RULE["min_top5_identical"]
    assert "development data only" in m["provenance"] and m["chosen"]["level"] in dict(LEVELS)


def test_no_reference_in_tracer_signature():
    import inspect
    assert "ref" not in " ".join(inspect.signature(N.trace_paths).parameters)


def test_frozen_config_is_exhaustive_and_budget_independent(dev):
    from iocevaluator.path_budget import FROZEN_MAX_EDGES, UNCAPPED_SAFETY_EXPANSIONS, exhaustive_path_config
    cfg = exhaustive_path_config()
    assert cfg.max_edges == FROZEN_MAX_EDGES and cfg.max_paths_per_seed >= 10 ** 9
    assert cfg.max_expansions_per_seed == UNCAPPED_SAFETY_EXPANSIONS
    ds = dev["ds"]
    c = _campaigns(ds, 1)[0]
    tr = N.trace_case(ds.tikg, c, dev["conf64"], dev["risk"], None, None, cfg)
    assert not tr.search_budget_hit and tr.n_expansions < UNCAPPED_SAFETY_EXPANSIONS
    ref = next(r for r in ds.reference_paths if r.campaign == c)
    ev = N.evaluate_case(ds.tikg, ref, tr, cfg)
    assert ev["search_budget_hit"] is False and ev["n_found"] == tr.n_found


@pytest.mark.parametrize("kw", [dict(max_edges=4, **UNCAPPED), dict(max_edges=4, max_paths_per_seed=6, max_expansions_per_seed=10 ** 9),
                                dict(max_edges=3, max_expansions_per_seed=15)])
def test_parallel_identical_neighbour_entries_identical(kw):
    """``TIKG`` removes duplicate triplets, but the tracer accepts any neighbour table.  Identical parallel entries make
    duplicate paths: the original rejected them with a dict while still charging the budget; the optimised tracer must too."""
    g = toy()
    nb = N._neighbours(g)
    for u in list(nb)[:4]:
        if nb[u]:
            nb[u] = [nb[u][0]] + nb[u]                                   # first entry twice, adjacent (sort order kept)
    conf, risk = np.full(g.n, 0.9), np.linspace(0.1, 0.9, g.n)
    a = R.trace_case(g, "k", conf, risk, None, None, R.PathConfig(**kw), nb=nb)
    b = N.trace_case(g, "k", conf, risk, None, None, N.PathConfig(**kw), nb=nb)
    assert a.truncated == b.truncated
    assert _cands(a) == _cands(b) and len(a.candidates) > 0
