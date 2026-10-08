import numpy as np
import pytest
import scipy.sparse as sp

from iocevaluator.attack_paths import PathConfig, evaluate_reference_paths, trace_candidate_paths
from iocevaluator.labels import ReferencePath
from iocevaluator.megits import megits_adjacency
from iocevaluator.prioritisation import (eigenvector_centrality, explain_ioc, fused_risk, rank_adjacency,
                                         select_tau, tune_alpha)
from iocevaluator.tikg import TIKG, Entity, EntityType as ET
from .synthetic import make_synthetic


def test_ec_matches_dense_eigendecomposition():
    rng = np.random.RandomState(0)
    a = rng.rand(30, 30) * (rng.rand(30, 30) > 0.7)
    a = np.triu(a, 1)
    a = a + a.T + np.diag(np.ones(30) * 5)                                  # diagonal must be dropped by rank_adjacency
    A = rank_adjacency(sp.csr_matrix(a))
    w, v = np.linalg.eigh(A.toarray())
    ref = np.abs(v[:, -1])
    ref /= ref.max()
    np.testing.assert_allclose(eigenvector_centrality(A), ref, atol=1e-6)
    # EC satisfies x = (1/lambda) A x
    x = eigenvector_centrality(A)
    np.testing.assert_allclose(A @ x, w[-1] * x, atol=1e-5)


def test_ec_star_vs_chain_prefers_influential_neighbours():
    # IOC A: 5 low-influence leaves ; IOC B: 2 neighbours that sit in a dense core (manuscript Sec. 3.6 example)
    n = 14
    a = np.zeros((n, n))
    for leaf in range(2, 7):
        a[0, leaf] = a[leaf, 0] = 1                                        # node0 = A, leaves 2..6
    core = [7, 8, 9, 10, 11]
    for i in core:
        for j in core:
            if i < j:
                a[i, j] = a[j, i] = 1
    a[1, 7] = a[7, 1] = a[1, 8] = a[8, 1] = 1                              # node1 = B -> two core nodes
    ec = eigenvector_centrality(sp.csr_matrix(a))
    assert ec[1] > ec[0]


def test_binarised_variant_and_selfloops_and_empty():
    a = sp.csr_matrix(np.array([[1, 0.9, 0.1], [0.9, 1, 0.5], [0.1, 0.5, 1]]))
    b = rank_adjacency(a, tau=0.5)
    assert b.diagonal().sum() == 0 and set(b.data) == {1.0} and b.nnz == 4
    assert eigenvector_centrality(sp.csr_matrix((4, 4))).sum() == 0


def test_per_component_option():
    a = sp.block_diag([np.ones((3, 3)) - np.eye(3), (np.ones((2, 2)) - np.eye(2)) * 0.1]).tocsr()
    glob = eigenvector_centrality(a)
    comp = eigenvector_centrality(a, per_component=True)
    assert glob[3] < 1e-6 and comp[3] > 0.5


def test_fusion_no_imputation_and_tuning():
    ec = np.array([0.2, 0.9, 0.5])
    sev = np.array([1.0, 0.0, 0.0])
    known = np.array([True, True, False])
    r = fused_risk(ec, sev, known, 0.5)
    np.testing.assert_allclose(r, [0.6, 0.45, 0.5])                        # unknown severity -> EC only
    np.testing.assert_allclose(fused_risk(ec, None, None, 0.3), ec)
    with pytest.raises(ValueError):
        fused_risk(ec, sev, known, 1.5)
    # alpha tuned on a validation metric: node 0 is the confirmed high-risk IOC -> severity must dominate
    a = tune_alpha(ec, sev, known, lambda R: float(np.argmax(R) == 0))
    assert a < 0.5
    adj = sp.csr_matrix(np.array([[0, 1, 0.2], [1, 0, 0.2], [0.2, 0.2, 0]]))
    assert isinstance(select_tau(adj, lambda e: float(e.sum())), float)


def chain_graph():
    ents = [Entity("ta1", ET.THREAT_ACTOR, campaign="k"), Entity("ta2", ET.THREAT_ACTOR, campaign="k"),
            Entity("d1", ET.DEVICE, campaign="k"), Entity("d2", ET.DEVICE, campaign="k"),
            Entity("v1", ET.VULNERABILITY, campaign="k"), Entity("p1", ET.PLATFORM, campaign="k"),
            Entity("m1", ET.ATTACK_METHOD, campaign="k")]
    t = [("ta1", "unauthorised_access", "d1"), ("ta2", "unauthorised_access", "d2"),
         ("d1", "affected_by", "v1"), ("d2", "affected_by", "v1"), ("d1", "runs_on", "p1"), ("d2", "runs_on", "p1"),
         ("ta1", "uses", "m1"), ("m1", "exploits", "v1")]
    return TIKG(ents, t)


def test_trace_finds_schema_conforming_paths_only():
    g = chain_graph()
    r = megits_adjacency(g)
    seeds = np.array([g.index["ta1"], g.index["ta2"], g.index["m1"]])
    risk, conf = np.ones(g.n), np.ones(g.n)
    paths = trace_candidate_paths(g, seeds, risk, conf, r.adj, r.weights, PathConfig(min_edges=2, max_edges=4))
    named = {tuple(g.entities[i].id for i in p.nodes): p for p in paths}
    chi19 = ("ta1", "d1", "v1", "d2", "ta2")
    assert chi19 in named or chi19[::-1] in named
    key = chi19 if chi19 in named else chi19[::-1]
    assert 19 in named[key].structures and named[key].n_edges == 4
    assert ("ta1", "d1", "p1", "d2", "ta2") in named or ("ta2", "d2", "p1", "d1", "ta1") in named
    # TA -uses-> M -exploits-> V is NOT a chi-conforming segment ending at a seed: no such path
    assert not any(p.nodes[-1] == g.index["v1"] or p.nodes[0] == g.index["v1"] for p in paths)
    # scores are sorted descending
    assert [p.score for p in paths] == sorted((p.score for p in paths), reverse=True)


def test_scope_and_evaluation_metrics():
    g = chain_graph()
    r = megits_adjacency(g)
    probs = np.ones((g.n, 2))
    risk = np.full(g.n, 0.1)
    risk[[g.index[i] for i in ("ta1", "ta2", "d1", "d2", "v1")]] = 1.0      # EC-style priority on the real chain
    ref = ReferencePath("case1", ["ta1", "d1", "v1", "d2", "ta2"], [], campaign="k")
    bad = ReferencePath("case2", ["ta1", "m1", "v1"], [], campaign="k")
    res = evaluate_reference_paths(g, [ref, bad], probs, risk, r.adj, r.weights, PathConfig(max_edges=4))
    assert res[">=4_edges"]["exact_path_hit"] == 1.0                     # reference within top-5
    assert 0.5 <= res[">=4_edges"]["edge_f1"] <= 1.0                     # top-1 path overlaps the reference
    assert res["2_edges"]["n"] == 1 and res["overall"]["n"] == 2
    assert res["2_edges"]["exact_path_hit"] == 0.0                          # non-conforming reference is not recovered
    # no seeds -> no candidates -> zero scores, no crash
    zero = evaluate_reference_paths(g, [ref], np.zeros((g.n, 2)), risk, r.adj, r.weights)
    assert zero["overall"]["exact_path_hit"] == 0.0 and zero["cases"][0]["n_candidates"] == 0


def test_explanation_has_four_elements():
    g, _ = make_synthetic(4)
    r = megits_adjacency(g, keep_components=True)
    ec = eigenvector_centrality(rank_adjacency(r.adj))
    node = g.index["dev0_0"]
    ex = explain_ioc(g, node, r.components, r.adj, ec)
    assert ex["contributing_structures"] and ex["top_neighbours"] and "eigenvector_centrality" in ex
    assert ex["recommended_action"] and ex["severity"] is None
