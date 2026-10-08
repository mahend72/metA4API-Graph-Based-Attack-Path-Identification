import numpy as np
import pytest
import scipy.sparse as sp
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from iocevaluator.metrics import (auc_scores, edge_f1, exact_path_hit_at_k, mean_std, multilabel_prf,
                                  paired_wilcoxon, topk_hit_rate)
from iocevaluator.models import (GCN, TrainConfig, fit, make_graph_data, masked_bce, normalise_adjacency,
                                 predict_proba)


def ring(n=12):
    a = sp.lil_matrix((n, n))
    for i in range(n):
        a[i, (i + 1) % n] = a[(i + 1) % n, i] = 1
    return a.tocsr()


def test_normalised_adjacency_matches_formula():
    A = ring(6).toarray() * 0.5
    At = A + np.eye(6)
    D = np.diag(At.sum(1) ** -0.5)
    np.testing.assert_allclose(normalise_adjacency(sp.csr_matrix(A)).toarray(), D @ At @ D, atol=1e-6)


def test_gcn_forward_matches_eq4():
    n, d, h, k = 12, 5, 7, 3
    A = ring(n)
    g = make_graph_data(A, np.zeros(n, int))
    torch.manual_seed(0)
    m = GCN(d, h, k, layers=2, dropout=0.5).eval()
    x = torch.randn(n, d)
    Ah = torch.as_tensor(normalise_adjacency(A).toarray(), dtype=torch.float32)
    z = torch.sigmoid(Ah @ torch.relu(Ah @ x @ m.W[0]) @ m.W[1])
    np.testing.assert_allclose(torch.sigmoid(m(x, g)).detach().numpy(), z.detach().numpy(), atol=1e-5)
    assert len(m.W) == 2 and m.dropout == 0.5
    for L in (1, 3, 4):
        assert GCN(d, h, k, layers=L)(x, g).shape == (n, k)


def test_masked_bce_ignores_invalid_labels():
    logits = torch.zeros(2, 2)
    Y = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    valid = torch.tensor([[True, False], [True, True]])
    manual = np.log(2)
    assert masked_bce(logits, Y, valid, torch.tensor([0, 1])).item() == pytest.approx(manual, rel=1e-5)


def _toy(n=60, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, n)
    x = np.c_[y + 0.3 * rng.randn(n), rng.randn(n)]
    Y = np.c_[y, 1 - y].astype(np.float32)
    A = sp.lil_matrix((n, n))
    for c in (0, 1):                              # homophilous graph: ring over the nodes of each class
        ids = np.flatnonzero(y == c)
        for a, b in zip(ids, np.roll(ids, 1)):
            A[a, b] = A[b, a] = 1
    return x.astype(np.float32), Y, A.tocsr()


@pytest.mark.parametrize("kind", ["gcn", "gat", "hgt"])
def test_models_learn_and_early_stop(kind):
    x, Y, A = _toy()
    valid = np.ones_like(Y, bool)
    types = np.zeros(len(x), int)
    g = make_graph_data(A, types, {"r": A} if kind == "hgt" else None)
    idx = np.arange(len(x))
    cfg = TrainConfig(max_epochs=200, patience=60, lr=2e-2, hidden=16, dropout=0.2)
    res = fit(kind, x, g, Y, valid, idx[:40], idx[40:50], cfg, seed=0)
    pred = predict_proba(res.model, x, g) > 0.5
    assert (pred[50:] == (Y[50:] > 0)).mean() > 0.7
    assert res.loss_history[-1] < res.loss_history[0] and len(res.epoch_times) == res.epochs_run
    assert res.best_epoch <= res.epochs_run


def test_early_stopping_patience():
    x, Y, A = _toy()
    g = make_graph_data(A, np.zeros(len(x), int))
    cfg = TrainConfig(max_epochs=500, patience=3, lr=1e-2)
    idx = np.arange(len(x))
    res = fit("gcn", x, g, Y, np.ones_like(Y, bool), idx[:40], idx[40:], cfg, 0)
    assert res.epochs_run < 500 and res.epochs_run - res.best_epoch == 3


def test_fit_is_seed_deterministic():
    x, Y, A = _toy()
    g = make_graph_data(A, np.zeros(len(x), int))
    idx = np.arange(len(x))
    cfg = TrainConfig(max_epochs=20, lr=1e-2)
    p1 = predict_proba(fit("gcn", x, g, Y, np.ones_like(Y, bool), idx[:40], idx[40:], cfg, 3).model, x, g)
    p2 = predict_proba(fit("gcn", x, g, Y, np.ones_like(Y, bool), idx[:40], idx[40:], cfg, 3).model, x, g)
    np.testing.assert_allclose(p1, p2)


def test_metrics_against_sklearn():
    rng = np.random.RandomState(0)
    Y = (rng.rand(80, 4) > 0.6).astype(np.float32)
    prob = np.clip(Y * 0.5 + rng.rand(80, 4) * 0.6, 0, 1)
    valid = np.ones_like(Y, bool)
    idx = np.arange(80)
    m = multilabel_prf(Y, prob > 0.5, valid, idx)
    assert m["macro_f1"] == pytest.approx(f1_score(Y, prob > 0.5, average="macro", zero_division=0))
    assert m["micro_f1"] == pytest.approx(f1_score(Y, prob > 0.5, average="micro", zero_division=0))
    a = auc_scores(Y, prob, valid, idx)
    assert a["roc_auc"] == pytest.approx(np.mean([roc_auc_score(Y[:, k], prob[:, k]) for k in range(4)]))
    assert a["pr_auc"] == pytest.approx(np.mean([average_precision_score(Y[:, k], prob[:, k]) for k in range(4)]))


def test_invalid_labels_not_scored():
    Y = np.array([[1, 1], [0, 1]], dtype=np.float32)
    valid = np.array([[True, False], [True, False]])
    pred = np.array([[True, True], [False, True]])
    m = multilabel_prf(Y, pred, valid, np.arange(2))
    assert m["micro_f1"] == 1.0 and m["n_labels_scored"] == 1


def test_topk_and_path_metrics_and_stats():
    cases = [(["a", "b", "c"], {"c"}), (["x", "y", "z"], {"q"})]
    assert topk_hit_rate(cases, 3) == 0.5 and topk_hit_rate(cases, 2) == 0.0
    ref = ["a", "b", "c"]
    assert exact_path_hit_at_k([["a", "x", "c"], ["c", "b", "a"]], ref, 5)
    assert not exact_path_hit_at_k([["a", "x", "c"], ["c", "b", "a"]], ref, 1)
    assert edge_f1(["a", "b", "c"], None, ref) == 1.0
    assert edge_f1(["c", "b", "a"], None, ref) == 1.0                       # undirected edge overlap
    assert edge_f1(["a", "b", "d"], None, ref) == pytest.approx(0.5)
    assert edge_f1(["a", "b", "c"], ["r1", "r9"], ref, ["r1", "r2"]) == pytest.approx(0.5)
    assert edge_f1(None, None, ref) == 0.0
    assert mean_std([1, 3]) == (2.0, 1.0)
    w = paired_wilcoxon(np.arange(10) + 1.0, np.arange(10) * 1.0)
    assert w["p_value"] < 0.05 and w["significant"]
