"""chi_1..chi_20 (Fig. 3): registry, hand-computed instance counts, independent dense re-implementation on both
synthetic datasets, matrix properties, weighting, determinism.
Datasets: fully synthetic and semi-synthetic (small, generated in memory / from the test fixture)."""
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from iocevaluator.datasets import load_dataset
from iocevaluator.megits import megits_adjacency, normalise_weights
from iocevaluator.metagraphs import (STRUCTURES, commuting_matrices, default_structures, get_structure, global_matrix,
                                     select_structures, structure_report, structure_templates)
from iocevaluator.synthetic_tikg import generate, write_dataset
from iocevaluator.tikg import TIKG, TRIPLET_SIGNATURES, Entity, EntityType as ET

FIXTURE = Path(__file__).parent / "fixtures" / "cve_pool_sample.json"
ALL = list(range(1, 21))
# Fig. 3 / Table 13: endpoint type and constituents
NODE_TYPE = {1: "TA", 2: "D", 3: "P", 4: "F", 5: "AT", 6: "V", 7: "D", 8: "TA", 9: "TA", 10: "M", 11: "TA", 12: "TA", 13: "D",
             14: "TA", 15: "D", 16: "TA", 17: "TA", 18: "TA", 19: "TA", 20: "TA"}
ABBR_TYPE = {"TA": "threat_actor", "D": "device", "P": "platform", "F": "file", "AT": "attack_type", "V": "vulnerability",
             "M": "attack_method"}
TABLE13 = {12: (1, 8), 13: (2, 7), 14: (8, 11), 15: (3, 7), 16: (2, 8), 17: (7, 8), 18: (10, 11),
           19: (2, 7, 8), 20: (2, 7, 8, 10, 11)}


@pytest.fixture(scope="module", params=["synthetic", "semi_synthetic"])
def graph(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(request.param)
    ds = generate("dev", 42) if request.param == "synthetic" else \
        generate("dev", 42, vulnerability_source="real", cve_pool=FIXTURE)
    write_dataset(ds, root / "dev")
    return load_dataset(request.param, profile="dev", root=root).tikg


# ------------------------------------------------------------------------------------------- hand-built graph
ENT = {"ta": (ET.THREAT_ACTOR, ["ta1", "ta2", "ta3"]), "d": (ET.DEVICE, ["d1", "d2", "d3"]),
       "v": (ET.VULNERABILITY, ["v1", "v2", "v3"]), "p": (ET.PLATFORM, ["p1", "p2"]), "f": (ET.FILE, ["f1", "f2"]),
       "at": (ET.ATTACK_TYPE, ["at1", "at2"]), "m": (ET.ATTACK_METHOD, ["m1", "m2"])}
EDGES = {  # Table 2 triplets, stored direction
    "T1": [("ta1", "v1"), ("ta2", "v1"), ("ta3", "v2")],
    "T2": [("d1", "v1"), ("d2", "v1"), ("d3", "v2")],
    "T3": [("p1", "v1"), ("p2", "v1")],
    "T4": [("f1", "v1"), ("f2", "v1")],
    "T5": [("at1", "v1"), ("at2", "v2")],
    "T6": [("v1", "v3"), ("v2", "v3")],
    "T7": [("d1", "p1"), ("d2", "p2"), ("d3", "p2")],
    "T8": [("ta1", "d1"), ("ta1", "d2"), ("ta2", "d2"), ("ta3", "d3")],
    "T9": [("ta1", "ta3"), ("ta2", "ta3")],
    "T10": [("m1", "v1"), ("m2", "v1"), ("m2", "v2")],
    "T11": [("ta1", "m1"), ("ta2", "m1"), ("ta2", "m2"), ("ta3", "m2")],
}


def hand():
    ents = [Entity(i, t) for t, ids in ENT.values() for i in ids]
    trips = [(s, TRIPLET_SIGNATURES[k][1], t) for k, es in EDGES.items() for s, t in es]
    return TIKG(ents, trips)


def _brute(k):
    """Instance counts straight from the Fig. 3 topology (explicit enumeration of instances, no matrix algebra)."""
    E = {k_: set(v) for k_, v in EDGES.items()}
    ta, d, v, p, f, at, m = (ENT[x][1] for x in ("ta", "d", "v", "p", "f", "at", "m"))
    def shared(T, nodes):   # a -T-> x <-T- b
        return {(a, b): sum(((a, x) in E[T] and (b, x) in E[T]) for x in {y for _, y in E[T]}) for a in nodes for b in nodes}
    base = {1: ("T1", ta), 2: ("T2", d), 3: ("T3", p), 4: ("T4", f), 5: ("T5", at), 6: ("T6", v), 7: ("T7", d),
            8: ("T8", ta), 9: ("T9", ta), 10: ("T10", m), 11: ("T11", ta)}
    if k in base:
        return base[k][1], shared(*base[k])
    if k == 12:   # a, b share a vulnerability (T1) AND a device (T8)
        return ta, {(a, b): sum(((a, x) in E["T1"] and (b, x) in E["T1"]) for x in v) *
                            sum(((a, y) in E["T8"] and (b, y) in E["T8"]) for y in d) for a, b in product(ta, ta)}
    if k == 13:
        return d, {(a, b): sum(((a, x) in E["T2"] and (b, x) in E["T2"]) for x in v) *
                           sum(((a, y) in E["T7"] and (b, y) in E["T7"]) for y in p) for a, b in product(d, d)}
    if k == 14:
        return ta, {(a, b): sum(((a, x) in E["T11"] and (b, x) in E["T11"]) for x in m) *
                            sum(((a, y) in E["T8"] and (b, y) in E["T8"]) for y in d) for a, b in product(ta, ta)}
    if k == 15:   # D -T7-> P -T3-> V <-T3- P <-T7- D
        return d, {(a, b): sum(((a, x) in E["T7"] and (x, y) in E["T3"] and (z, y) in E["T3"] and (b, z) in E["T7"])
                               for x, y, z in product(p, v, p)) for a, b in product(d, d)}
    if k in (16, 17):   # TA -T8-> D -T2|T7-> X <- D <-T8- TA
        T, X = ("T2", v) if k == 16 else ("T7", p)
        return ta, {(a, b): sum(((a, x) in E["T8"] and (x, y) in E[T] and (z, y) in E[T] and (b, z) in E["T8"])
                                for x, y, z in product(d, X, d)) for a, b in product(ta, ta)}
    if k == 18:   # TA -T11-> M -T10-> V <-T10- M <-T11- TA
        return ta, {(a, b): sum(((a, x) in E["T11"] and (x, y) in E["T10"] and (z, y) in E["T10"] and (b, z) in E["T11"])
                                for x, y, z in product(m, v, m)) for a, b in product(ta, ta)}
    if k == 19:   # TA -T8-> D => {V,P} <= D <-T8- TA ; branches share both devices
        return ta, {(a, b): sum(((a, x) in E["T8"] and (b, z) in E["T8"]) *
                                sum(((x, y) in E["T2"] and (z, y) in E["T2"]) for y in v) *
                                sum(((x, y) in E["T7"] and (z, y) in E["T7"]) for y in p)
                                for x, z in product(d, d)) for a, b in product(ta, ta)}
    if k == 20:   # TA => {M-branch, D-branch} => TA ; the V of the M-branch and the V of the D-branch are distinct nodes
        c18, c19 = _brute(18)[1], _brute(19)[1]
        return ta, {ab: c18[ab] * c19[ab] for ab in c18}
    raise KeyError(k)


# ------------------------------------------------------------------------------------------- registry
def test_registry_is_complete_and_matches_fig3():
    assert [s.id for s in STRUCTURES] == ALL and [s.id for s in default_structures()] == ALL
    assert [s.id for s in STRUCTURES if s.kind == "path"] == list(range(1, 12))
    assert [s.id for s in STRUCTURES if s.kind == "graph"] == list(range(12, 21))
    for s in STRUCTURES:
        assert s.node_type.value == ABBR_TYPE[NODE_TYPE[s.id]], s.id
        assert s.formula and s.schema and s.provenance == "manuscript_fig3"
        assert not hasattr(s, "status")
    assert {k: get_structure(k).uses for k in TABLE13} == TABLE13
    # every relation used in the templates is a Table 2 triplet signature, in its stored direction
    sigs = set(TRIPLET_SIGNATURES.values())
    for s in STRUCTURES:
        for tpl in structure_templates(s.id):
            for a, r, b, dr in tpl:
                assert ((a, r, b) if dr == "fwd" else (b, r, a)) in sigs, (s.id, a, r, b, dr)
            assert tpl[0][0] == s.node_type and tpl[-1][2] == s.node_type
    assert len(select_structures(kinds=["graph"])) == 9 and len(select_structures(ids=[3, 1])) == 2


# ------------------------------------------------------------------------------------------- definitions
@pytest.mark.parametrize("k", ALL)
def test_each_chi_matches_figure_instance_counts_on_hand_graph(k):
    g = hand()
    ids, expected = _brute(k)
    C = commuting_matrices(g, [get_structure(k)])[k].toarray()
    local = {i: j for j, i in enumerate(sorted(ids, key=lambda i: g.index[i]))}
    assert C.shape == (len(ids), len(ids))
    for (a, b), cnt in expected.items():
        assert C[local[a], local[b]] == cnt, (k, a, b)
    assert C.sum() == sum(expected.values())
    assert C.sum() > 0


def test_hand_graph_spot_values():
    C = commuting_matrices(hand())
    # chi2 (D,V,D): d1,d2 share v1
    np.testing.assert_array_equal(C[2].toarray(), [[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    # chi7 (D,P,D): d2,d3 share p2
    np.testing.assert_array_equal(C[7].toarray(), [[1, 0, 0], [0, 1, 1], [0, 1, 1]])
    # chi8 = TA-D-TA: ta1,ta2 share d2 ; ta1 has two devices
    np.testing.assert_array_equal(C[8].toarray(), [[2, 1, 0], [1, 1, 0], [0, 0, 1]])
    # chi13 = C2 (.) C7: only the diagonal and (d1,d2)? d1,d2 share v1 but not a platform -> identity
    np.testing.assert_array_equal(C[13].toarray(), C[2].multiply(C[7]).toarray())
    np.testing.assert_array_equal(C[13].toarray(), np.eye(3))
    # chi9 (TA,TA,TA): ta1,ta2 both assist ta3
    np.testing.assert_array_equal(C[9].toarray(), [[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    # chi6 (V,V,V): v1,v2 evolve to the same v3
    np.testing.assert_array_equal(C[6].toarray(), [[1, 1, 0], [1, 1, 0], [0, 0, 0]])


def test_composition_identities_on_hand_graph():
    g = hand()
    C = commuting_matrices(g)
    Q = lambda n: g.Q(*TRIPLET_SIGNATURES[n])
    for k in range(1, 12):
        q = Q(f"T{k}")
        np.testing.assert_array_equal(C[k].toarray(), (q @ q.T).toarray())
    had = lambda a, b: C[a].multiply(C[b]).toarray()
    np.testing.assert_array_equal(C[12].toarray(), had(1, 8))
    np.testing.assert_array_equal(C[13].toarray(), had(2, 7))
    np.testing.assert_array_equal(C[14].toarray(), had(11, 8))
    np.testing.assert_array_equal(C[15].toarray(), (Q("T7") @ C[3] @ Q("T7").T).toarray())
    np.testing.assert_array_equal(C[16].toarray(), (Q("T8") @ C[2] @ Q("T8").T).toarray())
    np.testing.assert_array_equal(C[17].toarray(), (Q("T8") @ C[7] @ Q("T8").T).toarray())
    np.testing.assert_array_equal(C[18].toarray(), (Q("T11") @ C[10] @ Q("T11").T).toarray())
    np.testing.assert_array_equal(C[19].toarray(), (Q("T8") @ C[13] @ Q("T8").T).toarray())   # Algorithm 1
    np.testing.assert_array_equal(C[20].toarray(), had(18, 19))


def test_algorithm1_chi19_differs_from_product_of_chi16_chi17():
    """chi_19 forces BOTH branches through the same device pair, so it is a strict sub-count of C16 (.) C17."""
    C = commuting_matrices(hand())
    assert (C[19].toarray() <= C[16].multiply(C[17]).toarray()).all()


@pytest.mark.parametrize("k", ALL)
def test_each_chi_matches_dense_reimplementation_on_synthetic_datasets(graph, k):
    """Independent einsum implementation straight from the Fig. 3 topology (dense arrays)."""
    def q(n):
        return graph.Q(*TRIPLET_SIGNATURES[n]).toarray().astype(np.float64)
    base = {1: "T1", 2: "T2", 3: "T3", 4: "T4", 5: "T5", 6: "T6", 7: "T7", 8: "T8", 9: "T9", 10: "T10", 11: "T11"}
    g = lambda n: np.einsum("ax,bx->ab", q(n), q(n))
    if k in base:
        exp = g(base[k])
    elif k == 12: exp = g("T1") * g("T8")
    elif k == 13: exp = g("T2") * g("T7")
    elif k == 14: exp = g("T11") * g("T8")
    elif k == 15: exp = np.einsum("dp,py,zy,ez->de", q("T7"), q("T3"), q("T3"), q("T7"))
    elif k == 16: exp = np.einsum("ad,dv,ev,be->ab", q("T8"), q("T2"), q("T2"), q("T8"))
    elif k == 17: exp = np.einsum("ad,dp,ep,be->ab", q("T8"), q("T7"), q("T7"), q("T8"))
    elif k == 18: exp = np.einsum("am,mv,nv,bn->ab", q("T11"), q("T10"), q("T10"), q("T11"))
    elif k == 19: exp = np.einsum("ad,be,dv,ev,dp,ep->ab", q("T8"), q("T8"), q("T2"), q("T2"), q("T7"), q("T7"))
    else:
        c18 = np.einsum("am,mv,nv,bn->ab", q("T11"), q("T10"), q("T10"), q("T11"))
        c19 = np.einsum("ad,be,dv,ev,dp,ep->ab", q("T8"), q("T8"), q("T2"), q("T2"), q("T7"), q("T7"))
        exp = c18 * c19
    got = commuting_matrices(graph, [get_structure(k)])[k].toarray()
    np.testing.assert_allclose(got, exp, rtol=1e-5, atol=1e-5)


def test_matrix_dimensions_symmetry_nonzero_and_node_mapping(graph):
    C = commuting_matrices(graph)
    assert sorted(C) == ALL
    for s in STRUCTURES:
        n = len(graph.type_nodes[s.node_type])
        M = C[s.id]
        assert M.shape == (n, n) and sp.issparse(M)
        assert abs(M - M.T).sum() == 0 and (M.data >= 0).all()
        assert M.nnz > 0 and (M.diagonal() > 0).any()                    # non-zero structure on both datasets
        off = M - sp.diags(M.diagonal())
        off.eliminate_zeros()
        assert off.nnz > 0, s.id                                          # some real pairwise evidence
        G = global_matrix(graph, s, M)
        assert G.shape == (graph.n, graph.n) and G.nnz == M.nnz
        rows = np.unique(G.nonzero()[0])
        assert set(graph.types[rows]) == {s.node_type.value}              # indices map to nodes of the right type
    # Hadamard meta-graphs are sub-structures of their constituents
    for k, (a, b) in {12: (1, 8), 13: (2, 7), 14: (11, 8)}.items():
        assert ((C[k] > 0).astype(int) - (C[a] > 0).astype(int)).max() <= 0
        assert ((C[k] > 0).astype(int) - (C[b] > 0).astype(int)).max() <= 0
    for a in (18, 19):
        assert ((C[20] > 0).astype(int) - (C[a] > 0).astype(int)).max() <= 0


def test_templates_for_all_structures():
    for k in ALL:
        assert structure_templates(k)
    assert {len(t) for t in structure_templates(13)} == {2}
    assert [len(t) for t in structure_templates(19)] == [4, 4] and [len(t) for t in structure_templates(15)] == [4]
    assert len(structure_templates(20)) == 3


def test_x1_extension_does_not_change_any_structure():
    a = generate("dev", 42)
    b = generate("dev", 42, include_x1=True)
    assert [e for e in a.edges] != [e for e in b.edges]
    from iocevaluator.tikg import tikg_from_dict
    mk = lambda d: tikg_from_dict({"entities": d.nodes, "triplets": [{"s": e["source"], "r": e["relation"], "t": e["target"]} for e in d.edges]})
    ca, cb = commuting_matrices(mk(a)), commuting_matrices(mk(b))
    for k in ALL:
        assert abs(ca[k] - cb[k]).sum() == 0                              # no chi uses the non-Table-2 file triplet


# ------------------------------------------------------------------------------------------- MeGiTS weighting & determinism
def test_weighted_combination(graph):
    structs = default_structures()
    u = megits_adjacency(graph, keep_components=True)
    assert len(u.weights) == 20 and all(abs(w - 1 / 20) < 1e-12 for w in u.weights.values())
    w = {k: 0.02 for k in ALL}
    w[19] = 0.4
    w = {k: v / sum(w.values()) for k, v in w.items()}                    # non-uniform, sums to 1
    r = megits_adjacency(graph, structs, w, keep_components=True)
    assert abs(sum(r.components.values()) - r.adj).sum() < 1e-5
    only = megits_adjacency(graph, select_structures(ids=[19]), {19: 1.0}).adj
    ta = np.flatnonzero(graph.types == "threat_actor")
    assert only.nnz > 0 and set(only.nonzero()[0]) <= set(ta)
    with pytest.raises(ValueError):
        normalise_weights(structs, {k: 0.1 for k in ALL})                 # sums to 2.0
    sub = megits_adjacency(graph, select_structures(ids=[1, 9]))
    assert len(sub.weights) == 2 and sub.adj.nnz > 0
    assert u.adj.nnz >= max(only.nnz, sub.adj.nnz)


def test_deterministic(graph):
    a, b = commuting_matrices(graph), commuting_matrices(graph)
    for k in a:
        assert (a[k] != b[k]).nnz == 0
    r1, r2 = megits_adjacency(graph).adj, megits_adjacency(graph).adj
    assert (r1 != r2).nnz == 0


def test_structure_report(graph):
    rows = {r["id"]: r for r in structure_report(graph)}
    assert len(rows) == 20 and all(rows[k]["nnz"] > 0 and rows[k]["offdiag_nnz"] > 0 for k in ALL)
    assert all("status" not in r for r in rows.values())
