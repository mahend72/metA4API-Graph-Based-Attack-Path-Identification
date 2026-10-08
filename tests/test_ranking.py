"""Threat prioritisation / ranking stage (Sec. 3.6): A_rank, eigenvector centrality (weighted / binarised, scopes),
normalisation, severity handling, fusion R = alpha*EC + (1-alpha)*Severity, thresholding, ties, determinism.
Synthetic / semi-synthetic data only."""
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from iocevaluator.datasets import load_dataset
from iocevaluator.megits import megits_adjacency
from iocevaluator.ranking import (RankerConfig, build_rank_adjacency, campaign_rankings, centrality, fuse, prioritise,
                                  rank_nodes, ranked_by_type, severity_vector)
from iocevaluator.synthetic_tikg import generate, write_dataset
from iocevaluator.tikg import TIKG, Entity, EntityType as ET

FIXTURE = Path(__file__).parent / "fixtures" / "cve_pool_sample.json"


@pytest.fixture(scope="module", params=["synthetic", "semi_synthetic"])
def ds(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(request.param)
    d = generate("dev", 42) if request.param == "synthetic" else \
        generate("dev", 42, vulnerability_source="real", cve_pool=FIXTURE)
    write_dataset(d, root / "dev")
    return load_dataset(request.param, profile="dev", root=root)


def small_graph(order=None):
    """3 vulnerabilities + 2 devices + 1 actor in one campaign, ids chosen to exercise id tie-breaking."""
    ents = [Entity("v1", ET.VULNERABILITY, campaign="k"), Entity("v2", ET.VULNERABILITY, campaign="k"),
            Entity("v3", ET.VULNERABILITY, campaign="k"), Entity("d1", ET.DEVICE, campaign="k"),
            Entity("d2", ET.DEVICE, campaign="k"), Entity("ta", ET.THREAT_ACTOR, campaign="k"),
            Entity("w1", ET.VULNERABILITY, campaign="other")]
    if order:
        ents = [ents[i] for i in order]
    return TIKG(ents, [])


def mat(g, edges):
    a = sp.lil_matrix((g.n, g.n))
    for s, t, w in edges:
        a[g.index[s], g.index[t]] = a[g.index[t], g.index[s]] = w
    return a.tocsr()


# ------------------------------------------------------------------------------------------- A_rank
def test_a_rank_is_the_megits_adjacency_without_self_loops(ds):
    adj = megits_adjacency(ds.tikg).adj
    a = build_rank_adjacency(adj)
    assert a.diagonal().sum() == 0
    off = adj - sp.diags(adj.diagonal())
    off.eliminate_zeros()
    assert abs(a - off).sum() < 1e-6 and a.nnz == off.nnz               # weights preserved exactly
    with_loops = adj + sp.eye(adj.shape[0]) * 5
    assert abs(build_rank_adjacency(with_loops) - a).sum() < 1e-6        # self-loops are removed
    assert abs(a - a.T).sum() < 1e-6 and a.data.min() > 0


def test_binarised_a_rank_uses_threshold_tau():
    a = sp.csr_matrix(np.array([[1, 0.9, 0.1], [0.9, 1, 0.5], [0.1, 0.5, 1]]))
    b = build_rank_adjacency(a, tau=0.5)
    assert b.diagonal().sum() == 0 and set(b.data) == {1.0} and b.nnz == 4          # >= tau keeps the 0.5 edge
    assert build_rank_adjacency(a, tau=0.95).nnz == 0
    counts = [build_rank_adjacency(a, tau=t).nnz for t in (0.0, 0.2, 0.6, 1.0)]
    assert counts == sorted(counts, reverse=True)                                    # monotone in tau


# ------------------------------------------------------------------------------------------- centrality
def test_centrality_hand_computed_path_graph_and_binarisation():
    g = small_graph()
    ids = ["v1", "v2", "v3"]
    # weighted path v1 -a- v2 -b- v3: principal eigenvector (a, sqrt(a^2+b^2), b); max-normalised
    a_, b_ = 1.0, 0.3
    A = mat(g, [("v1", "v2", a_), ("v2", "v3", b_)])
    ec = centrality(g, build_rank_adjacency(A), scope="global")
    s = np.hypot(a_, b_)
    np.testing.assert_allclose([ec[g.index[i]] for i in ids], [a_ / s, 1.0, b_ / s], atol=1e-6)
    assert ec[g.index["d1"]] == 0                                                   # isolated node
    # binarised with tau = 0.5 drops the weak edge: single edge v1-v2
    ecb = centrality(g, build_rank_adjacency(A, tau=0.5), scope="global")
    np.testing.assert_allclose([ecb[g.index[i]] for i in ids], [1.0, 1.0, 0.0], atol=1e-6)
    # unit weights: (1, sqrt 2, 1)/sqrt 2
    ecu = centrality(g, build_rank_adjacency(mat(g, [("v1", "v2", 1), ("v2", "v3", 1)])), scope="global")
    np.testing.assert_allclose([ecu[g.index[i]] for i in ids], [1 / np.sqrt(2), 1, 1 / np.sqrt(2)], atol=1e-6)


def test_centrality_matches_dense_eigendecomposition(ds):
    g = ds.tikg
    A = build_rank_adjacency(megits_adjacency(g).adj)
    w, v = np.linalg.eigh(A.toarray())
    ref = np.abs(v[:, -1])
    ref /= ref.max()
    ec = centrality(g, A, scope="global")
    np.testing.assert_allclose(ec, ref, atol=1e-5)
    np.testing.assert_allclose(A @ ec, w[-1] * ec, atol=1e-4)                        # x = (1/lambda) A x


def test_ec_scope_literal_global_is_degenerate_type_scope_is_not(ds):
    """A_rank is block-diagonal by entity type (every chi_k is single-type): the literal global principal eigenvector
    lives on one block only, so the default scope solves the same equation per entity-type block."""
    g = ds.tikg
    A = build_rank_adjacency(megits_adjacency(g).adj)
    glob = centrality(g, A, "global")
    per_type_max = {t: glob[g.type_nodes[t]].max() for t in ET}
    assert sum(m > 1e-6 for m in per_type_max.values()) == 1                         # a single dominant block
    typ = centrality(g, A, "type")
    for t in ET:
        sub = A[g.type_nodes[t]][:, g.type_nodes[t]]
        assert sub.nnz == 0 or typ[g.type_nodes[t]].max() == pytest.approx(1.0)
    comp = centrality(g, A, "component")
    assert (comp > 1e-9).sum() >= (typ > 1e-9).sum()
    assert RankerConfig().ec_scope == "type"


def test_ec_normalisation_ranges(ds):
    g = ds.tikg
    A = build_rank_adjacency(megits_adjacency(g).adj)
    for scope in ("global", "type", "component"):
        for norm in ("max", "minmax"):
            ec = centrality(g, A, scope, norm)
            assert ec.min() >= 0 and ec.max() <= 1 + 1e-9 and np.isfinite(ec).all()
    mm = centrality(g, A, "type", "minmax")
    for t in ET:
        sub = A[g.type_nodes[t]][:, g.type_nodes[t]]
        if sub.nnz:
            x = mm[g.type_nodes[t]]
            assert x.min() == pytest.approx(0.0) and x.max() == pytest.approx(1.0)


# ------------------------------------------------------------------------------------------- severity
def test_severity_only_for_vulnerabilities_and_never_invented():
    g = small_graph()
    sev_raw = np.full(g.n, 0.9)                         # even if a caller supplies values for every node ...
    known = np.ones(g.n, dtype=bool)
    known[g.index["v3"]] = False                        # ... and v3 has no CVSS
    sev, avail = severity_vector(g, sev_raw, known)
    for i, e in enumerate(g.entities):
        if e.type == ET.VULNERABILITY and e.id != "v3":
            assert avail[i] and sev[i] == 0.9
        else:
            assert not avail[i] and np.isnan(sev[i])      # devices / actors / unknown-CVSS vulnerability: unavailable
    s0, a0 = severity_vector(g, None, None)
    assert np.isnan(s0).all() and not a0.any()
    with pytest.raises(ValueError):
        severity_vector(g, np.full(g.n, 7.5), known)      # raw CVSS 0-10 must be divided by 10 first


def test_real_cvss_values_flow_from_semi_synthetic_nodes(tmp_path_factory):
    root = tmp_path_factory.mktemp("semi")
    write_dataset(generate("dev", 42, vulnerability_source="real", cve_pool=FIXTURE), root / "dev")
    d = load_dataset("semi_synthetic", profile="dev", root=root)
    nodes = json.loads((root / "dev" / "nodes.json").read_text())
    raw = {n["id"]: n["attrs"]["cvss_base_score"] for n in nodes if n["attrs"].get("cvss_base_score") is not None}
    assert raw
    sev, avail = severity_vector(d.tikg, d.severity, d.severity_known)
    assert avail.sum() == len(raw) and set(d.tikg.types[avail]) == {"vulnerability"}
    for eid, score in raw.items():
        assert sev[d.tikg.index[eid]] == pytest.approx(float(score) / 10.0)         # real CVSS base score / 10
    assert np.nanmin(sev) >= 0 and np.nanmax(sev) <= 1


# ------------------------------------------------------------------------------------------- fusion
def test_fusion_equation_and_alpha_extremes():
    ec = np.array([0.2, 0.9, 0.5, 0.4])
    sev = np.array([1.0, 0.0, np.nan, 0.5])
    av = np.array([True, True, False, True])
    r = fuse(ec, sev, av, 0.5)
    np.testing.assert_allclose(r, [0.6, 0.45, 0.5, 0.45])                         # node 2: no severity -> EC only
    np.testing.assert_allclose(fuse(ec, sev, av, 1.0), ec)                        # alpha = 1: pure EC
    r0 = fuse(ec, sev, av, 0.0)
    np.testing.assert_allclose(r0, [1.0, 0.0, 0.5, 0.5])                          # alpha = 0: pure severity where known
    r_ex = fuse(ec, sev, av, 0.5, "exclude")
    assert np.isnan(r_ex[2]) and not np.isnan(r_ex[[0, 1, 3]]).any()
    r_z = fuse(ec, sev, av, 0.5, "zero")
    assert r_z[2] == pytest.approx(0.25)                                          # explicit imputation: Severity := 0
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            fuse(ec, sev, av, bad)
    assert 0.0 <= r.max() <= 1.0


def test_prioritise_alpha_extremes_on_graph_and_ranking_order():
    g = small_graph()
    A = mat(g, [("v1", "v2", 1.0), ("v2", "v3", 1.0)])                           # EC: v2 > v1 = v3
    sev = np.zeros(g.n)
    known = np.zeros(g.n, dtype=bool)
    for vid, s in (("v1", 0.9), ("v2", 0.1), ("v3", 0.5)):
        sev[g.index[vid]], known[g.index[vid]] = s, True
    vulns = [g.index[i] for i in ("v1", "v2", "v3")]
    ec_only = prioritise(g, A, sev, known, RankerConfig(alpha=1.0))
    assert [g.entities[i].id for i in rank_nodes(g, ec_only.risk, vulns)] == ["v2", "v1", "v3"]   # v1/v3 tie -> id
    sev_only = prioritise(g, A, sev, known, RankerConfig(alpha=0.0))
    assert [g.entities[i].id for i in rank_nodes(g, sev_only.risk, vulns)] == ["v1", "v3", "v2"]  # by CVSS
    np.testing.assert_allclose(sev_only.risk[vulns], [0.9, 0.1, 0.5])
    mid = prioritise(g, A, sev, known, RankerConfig(alpha=0.5))
    np.testing.assert_allclose(mid.risk[vulns], 0.5 * mid.ec[vulns] + 0.5 * np.array([0.9, 0.1, 0.5]))
    # nodes without severity keep R = EC under the default policy and are NaN under "exclude"
    assert mid.risk[g.index["d1"]] == mid.ec[g.index["d1"]]
    ex = prioritise(g, A, sev, known, RankerConfig(alpha=0.5, missing_severity="exclude"))
    assert np.isnan(ex.risk[g.index["d1"]]) and not np.isnan(ex.risk[vulns]).any()
    with pytest.raises(ValueError):
        prioritise(g, A, sev, known, RankerConfig(alpha=None))                    # alpha must be supplied / tuned
    with pytest.raises(ValueError):
        RankerConfig(alpha=2.0)


def test_tau_changes_the_ranking_through_the_binarised_adjacency():
    g = small_graph()
    A = mat(g, [("v1", "v2", 1.0), ("v2", "v3", 0.3)])
    w = prioritise(g, A, cfg=RankerConfig(alpha=1.0))
    b = prioritise(g, A, cfg=RankerConfig(alpha=1.0, tau=0.5))
    v = [g.index[i] for i in ("v1", "v2", "v3")]
    assert w.ec[v[2]] > 0 and b.ec[v[2]] == 0 and b.a_rank.nnz == 2                  # weak edge removed by tau
    empty = prioritise(g, A, cfg=RankerConfig(alpha=1.0, tau=5.0))
    assert empty.ec.sum() == 0 and empty.a_rank.nnz == 0


# ------------------------------------------------------------------------------------------- ties / determinism / listings
def test_ties_are_broken_by_entity_id_and_nan_is_not_ranked():
    g = small_graph()
    score = np.zeros(g.n)
    assert [g.entities[i].id for i in rank_nodes(g, score)] == sorted(e.id for e in g.entities)
    score[g.index["v2"]] = 0.7
    score[g.index["v1"]] = 0.7 + 1e-13                                               # numerical noise must not reorder ties
    score[g.index["d2"]] = np.nan
    ranked = [g.entities[i].id for i in rank_nodes(g, score)]
    assert ranked[:2] == ["v1", "v2"] and "d2" not in ranked and len(ranked) == g.n - 1
    assert [g.entities[i].id for i in rank_nodes(g, score, top=1)] == ["v1"]


def test_rankings_are_independent_of_node_order_and_repeatable():
    edges = [("v1", "v2", 1.0), ("v2", "v3", 1.0), ("d1", "d2", 1.0)]
    outs = []
    for order in (None, [6, 5, 4, 3, 2, 1, 0], [3, 0, 5, 1, 6, 2, 4]):
        g = small_graph(order)
        r = prioritise(g, mat(g, edges), cfg=RankerConfig(alpha=1.0))
        outs.append([(g.entities[i].id, round(float(r.risk[i]), 8)) for i in rank_nodes(g, r.risk)])
    assert outs[0] == outs[1] == outs[2]
    g = small_graph()
    a, b = (prioritise(g, mat(g, edges), cfg=RankerConfig(alpha=1.0)).risk for _ in range(2))
    np.testing.assert_array_equal(a, b)


def test_per_campaign_and_per_type_rankings(ds):
    g = ds.tikg
    adj = megits_adjacency(g).adj
    r = prioritise(g, adj, ds.severity, ds.severity_known, RankerConfig(alpha=0.5))
    camp = g.campaigns()
    ranks = campaign_rankings(g, r.risk)
    assert set(ranks) == {c for c in camp if c}
    for c, lst in ranks.items():
        assert {i for i, _ in lst} <= {e.id for e, cc in zip(g.entities, camp) if cc == c}
        s = [x for _, x in lst]
        assert [round(x, 10) for x in s] == sorted([round(x, 10) for x in s], reverse=True)
    top3 = campaign_rankings(g, r.risk, entity_types=["vulnerability"], top=3)
    assert all(len(v) <= 3 for v in top3.values())
    assert all(g.entities[g.index[i]].type == ET.VULNERABILITY for v in top3.values() for i, _ in v)
    again = campaign_rankings(g, prioritise(g, adj, ds.severity, ds.severity_known, RankerConfig(alpha=0.5)).risk)
    assert again == ranks                                                           # deterministic
    by_type = ranked_by_type(g, r.risk, top=3)
    assert set(by_type) == {t.value for t in ET} and all(len(v) <= 3 for v in by_type.values())
    assert r.severity_available.sum() == ds.severity_known.sum() and set(g.types[r.severity_available]) == {"vulnerability"}
    assert 0 <= np.nanmin(r.risk) and np.nanmax(r.risk) <= 1 + 1e-9
