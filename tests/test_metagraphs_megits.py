import numpy as np
import pytest
import scipy.sparse as sp

from iocevaluator.megits import (fold_megits_adjacency, megits_adjacency, normalise_weights,
                                 structure_similarity)
from iocevaluator.metagraphs import (STRUCTURES, commuting_matrices, default_structures, get_structure,
                                     select_structures, structure_templates)
from iocevaluator.tikg import TIKG, Entity, EntityType as ET
from .synthetic import make_synthetic


def hand_graph():
    ents = [Entity("ta1", ET.THREAT_ACTOR), Entity("ta2", ET.THREAT_ACTOR), Entity("d1", ET.DEVICE),
            Entity("d2", ET.DEVICE), Entity("v1", ET.VULNERABILITY), Entity("p1", ET.PLATFORM)]
    t = [("ta1", "unauthorised_access", "d1"), ("ta2", "unauthorised_access", "d2"),
         ("d1", "affected_by", "v1"), ("d2", "affected_by", "v1"), ("d1", "runs_on", "p1"), ("d2", "runs_on", "p1")]
    return TIKG(ents, t)


def test_twenty_structures_registry_all_symmetric():
    assert [s.id for s in STRUCTURES] == list(range(1, 21))
    assert {s.kind for s in STRUCTURES[:11]} == {"path"} and {s.kind for s in STRUCTURES[11:]} == {"graph"}
    assert [s.id for s in default_structures()] == list(range(1, 21))
    g, _ = make_synthetic(4)
    C = commuting_matrices(g)                                  # default = all 20 structures
    assert sorted(C) == list(range(1, 21))
    for s in default_structures():
        n = len(g.type_nodes[s.node_type])
        assert C[s.id].shape == (n, n)
        assert abs(C[s.id] - C[s.id].T).sum() == 0          # symmetric
        assert (C[s.id].data >= 0).all()
    assert all(structure_templates(s.id) for s in default_structures())


def test_algorithm1_chi19_on_hand_graph():
    g = hand_graph()
    C = commuting_matrices(g, select_structures(ids=[2, 7, 19]))
    np.testing.assert_array_equal(C[2].toarray(), np.ones((2, 2)))     # D-V-D
    np.testing.assert_array_equal(C[7].toarray(), np.ones((2, 2)))     # D-P-D
    np.testing.assert_array_equal(C[19].toarray(), np.ones((2, 2)))    # Q_TAD (C2 (.) C7) Q_TAD^T
    # break the shared platform: Hadamard product must remove the actor link
    g2 = TIKG(g.entities + [Entity("p2", ET.PLATFORM)], [(g.entities[t.s].id, t.r, g.entities[t.t].id) for t in g.triplets
                                                          if not (t.r == "runs_on" and g.entities[t.s].id == "d2")]
              + [("d2", "runs_on", "p2")])
    C2 = commuting_matrices(g2, select_structures(ids=[19]))[19].toarray()
    assert C2[0, 1] == 0 and C2[0, 0] == 1


def test_megits_formula_hand_example():
    # C: nodes 0,1,2 ; C00=4,C11=2,C01=2 -> 2*2/(4+2)=2/3 ; zero-denominator pair -> 0
    C = sp.csr_matrix(np.array([[4, 2, 0], [2, 2, 0], [0, 0, 0]], dtype=float))
    S = structure_similarity(C).toarray()
    assert S[0, 1] == pytest.approx(2 / 3) and S[0, 0] == pytest.approx(1.0) and S[2, 2] == 0 and S[0, 2] == 0
    assert (S <= 1 + 1e-12).all()


def test_weights_validation_and_uniform_default():
    s = select_structures(ids=[1, 2])
    assert all(abs(v - 1 / 20) < 1e-12 for v in normalise_weights(STRUCTURES).values())
    assert all(abs(v - 1 / 20) < 1e-12 for v in normalise_weights(default_structures()).values())
    with pytest.raises(ValueError):
        normalise_weights(s, {1: 0.7, 2: 0.7})
    with pytest.raises(ValueError):
        normalise_weights(s, {1: -0.5, 2: 1.5})


def test_megits_adjacency_properties_and_weighting():
    g, _ = make_synthetic(4)
    r = megits_adjacency(g, keep_components=True)
    A = r.adj
    assert abs(A - A.T).sum() < 1e-6 and A.diagonal().sum() == 0 and A.min() >= 0
    assert A.max() <= 1.0 + 1e-6                               # weights sum to 1, each term <= 1
    comp_sum = sum(r.components.values())
    assert abs(comp_sum - A).sum() < 1e-5                      # adjacency == sum_k w_k S_k
    only = megits_adjacency(g, select_structures(ids=[2]), {2: 1.0}).adj
    assert only.nnz > 0 and only.nnz <= A.nnz
    b = megits_adjacency(g, binary=True).adj
    assert set(np.unique(b.data)) <= {1.0} and (b.nnz >= (A > 0).sum())


def test_fold_adjacency_is_independent_of_test_node_structure():
    g, _ = make_synthetic(6)
    test = np.array([e.campaign in {"c0", "c1"} for e in g.entities])
    base = fold_megits_adjacency(g, test).adj
    # perturb: add structure that touches only test nodes -> train-train entries must not change
    ents = g.entities
    extra = [(ents[a].id, "affected_by", ents[b].id) for a, b in
             [(g.index["dev0_0"], g.index["cve1_1"]), (g.index["dev1_0"], g.index["cve0_1"])]]
    g2 = type(g)(ents, [(ents[t.s].id, t.r, ents[t.t].id) for t in g.triplets] + extra)
    pert = fold_megits_adjacency(g2, test).adj
    tr = np.flatnonzero(~test)
    assert abs(base[tr][:, tr] - pert[tr][:, tr]).sum() == 0
    # ...whereas the leaky full-graph adjacency does change somewhere
    assert abs(megits_adjacency(g).adj - megits_adjacency(g2).adj).sum() > 0
    # entries involving test nodes are still present (test nodes enter at inference)
    assert base[np.flatnonzero(test)].nnz > 0
