"""Supervised multi-label GCN (manuscript Eq. 4-5, Sec. 4.2-4.3): architecture, heterogeneous label heads, leakage
guards, early stopping, determinism, and small training runs on the synthetic and semi-synthetic datasets.
NO manuscript results are produced here - these are pipeline-validation tests on synthetic data."""
import copy
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from iocevaluator.datasets import load_dataset
from iocevaluator.labels import build_label_space
from iocevaluator.megits import fold_megits_adjacency
from iocevaluator.models import make_graph_data, normalise_adjacency
from iocevaluator.splits import group_stratified_folds
from iocevaluator.supervised_gcn import (LabelLeakageError, MultiLabelGCN, SupervisedGCNConfig, dimension_report,
                                         fit_supervised, inner_validation_split, make_masks, run_supervised_fold,
                                         type_heads, visible_labels)
from iocevaluator.synthetic_tikg import generate, write_dataset
from iocevaluator.tikg import EntityType
from .synthetic import make_synthetic

FIXTURE = Path(__file__).parent / "fixtures" / "cve_pool_sample.json"
FAST = dict(max_epochs=40, lr=1e-2, patience=20)


# ------------------------------------------------------------------------------------------- toy graph helpers
def toy_graph(n_campaigns=9):
    g, lab = make_synthetic(n_campaigns)
    ls = build_label_space(g, lab)
    return g, ls


def toy_setup(seed=0, cfg=None):
    g, ls = toy_graph()
    folds = group_stratified_folds(g, ls, 3, seed)
    f = folds[0]
    return g, ls, f


# ------------------------------------------------------------------------------------------- configuration
def test_defaults_are_the_manuscript_configuration():
    c = SupervisedGCNConfig()
    assert (c.layers, c.hidden, c.dropout, c.lr, c.weight_decay, c.patience, c.threshold) == (2, 64, 0.5, 1e-3, 5e-4, 20, 0.5)
    for bad in (0, 5):
        with pytest.raises(ValueError):
            SupervisedGCNConfig(layers=bad)


# ------------------------------------------------------------------------------------------- architecture
def ring(n=12):
    a = sp.lil_matrix((n, n))
    for i in range(n):
        a[i, (i + 1) % n] = a[(i + 1) % n, i] = 1
    return a.tocsr()


def test_forward_is_eq4_for_two_layers_and_valid_for_depths_1_to_4():
    n, d, k = 12, 5, 4
    A = ring(n) * 0.5
    g = make_graph_data(A, np.zeros(n, int))
    valid = np.ones((n, k), bool)
    torch.manual_seed(0)
    m = MultiLabelGCN(d, k, valid, SupervisedGCNConfig(hidden=7)).eval()
    x = torch.randn(n, d)
    Ah = torch.as_tensor(normalise_adjacency(A).toarray(), dtype=torch.float32)
    z = torch.sigmoid(Ah @ torch.relu(Ah @ x @ m.W[0]) @ m.W[1])
    np.testing.assert_allclose(m.proba(x, g).detach().numpy(), z.detach().numpy(), atol=1e-5)
    assert [tuple(w.shape) for w in m.W] == [(5, 7), (7, 4)] and m.dropout == 0.5
    assert not any("bias" in name for name, _ in m.named_parameters())            # Eq. 4 has no bias terms
    for L in (1, 2, 3, 4):
        mm = MultiLabelGCN(d, k, valid, SupervisedGCNConfig(layers=L, hidden=7)).eval()
        assert len(mm.W) == L and mm.proba(x, g).shape == (n, k)
    assert [tuple(w.shape) for w in MultiLabelGCN(d, k, valid, SupervisedGCNConfig(layers=4, hidden=7)).W] == \
        [(5, 7), (7, 7), (7, 7), (7, 4)]


def test_dropout_only_active_in_training_mode():
    n, d, k = 12, 5, 3
    g = make_graph_data(ring(n), np.zeros(n, int))
    m = MultiLabelGCN(d, k, np.ones((n, k), bool))
    x = torch.randn(n, d)
    m.eval()
    assert torch.equal(m.proba(x, g), m.proba(x, g))
    m.train()
    assert not torch.equal(m.proba(x, g), m.proba(x, g))


# ------------------------------------------------------------------------------------------- heterogeneous label spaces
def test_label_heads_differ_by_entity_type_and_partition_the_output():
    g, ls = toy_graph()
    heads = type_heads(g, ls)
    cols = np.concatenate([h.columns for h in heads.values()])
    assert sorted(cols) == list(range(ls.K))                       # disjoint cover of the K output columns
    assert {t: h.out_dim for t, h in heads.items()}[EntityType.THREAT_ACTOR] == 3   # fam0..fam2 in the toy data
    for t, h in heads.items():
        assert len(h.nodes) == len(g.type_nodes[t])
        # a node only owns its own type's columns
        assert ls.valid[np.ix_(h.nodes, h.columns)].all()
        other = np.setdiff1d(np.arange(ls.K), h.columns)
        assert not ls.valid[np.ix_(h.nodes, other)].any()


def test_predictions_are_zero_outside_the_nodes_own_label_space():
    g, ls, f = toy_setup()
    masks = make_masks(ls, f.train, f.val, f.test)
    n = g.n
    graph = make_graph_data(fold_megits_adjacency(g, f.test_nodes_mask).adj, np.zeros(n, int))
    x = np.random.RandomState(0).randn(n, 6).astype(np.float32)
    res = fit_supervised(x, graph, visible_labels(ls, masks), ls.valid, masks, SupervisedGCNConfig(max_epochs=5), 0)
    p = res.predict(x, graph)
    assert p.shape == (n, ls.K) and (p >= 0).all() and (p <= 1).all()
    assert (p[~ls.valid] == 0).all()
    by_type = res.predict_by_type(x, graph, type_heads(g, ls))
    for t, tp in by_type.items():
        assert tp.prob.shape == (len(g.type_nodes[t]), len(ls.classes[t])) and len(tp.classes) == tp.prob.shape[1]


# ------------------------------------------------------------------------------------------- masks / leakage
def test_mask_validation_rejects_overlap_unlabelled_and_missing_validation():
    g, ls, f = toy_setup()
    make_masks(ls, f.train, f.val, f.test)
    with pytest.raises(LabelLeakageError):
        make_masks(ls, f.train, f.val, np.concatenate([f.test, f.val[:1]]))        # val/test overlap
    with pytest.raises(LabelLeakageError):
        make_masks(ls, np.concatenate([f.train, f.test[:1]]), f.val, f.test)       # train/test overlap
    with pytest.raises(LabelLeakageError):
        make_masks(ls, f.train, f.val[:0], f.test)                                 # no inner validation split
    unl = np.flatnonzero(~ls.labelled)
    if len(unl):
        with pytest.raises(LabelLeakageError):
            make_masks(ls, f.train, np.concatenate([f.val, unl[:1]]), f.test)


def test_trainer_refuses_labels_outside_train_and_val():
    g, ls, f = toy_setup()
    masks = make_masks(ls, f.train, f.val, f.test)
    y = visible_labels(ls, masks)
    assert not y[masks.test].any() and np.array_equal(y[masks.train], ls.Y[masks.train])
    graph = make_graph_data(sp.eye(g.n, format="csr"), np.zeros(g.n, int))
    x = np.zeros((g.n, 3), np.float32)
    with pytest.raises(LabelLeakageError):
        fit_supervised(x, graph, ls.Y, ls.valid, masks, SupervisedGCNConfig(max_epochs=2))   # full Y incl. test rows


def test_test_labels_do_not_influence_training():
    """Changing test-node labels must leave the trained weights bit-identical (no label leakage)."""
    g, ls, f = toy_setup()
    masks = make_masks(ls, f.train, f.val, f.test)
    graph = make_graph_data(fold_megits_adjacency(g, f.test_nodes_mask).adj, np.zeros(g.n, int))
    x = np.random.RandomState(1).randn(g.n, 5).astype(np.float32)
    cfg = SupervisedGCNConfig(**FAST)
    a = fit_supervised(x, graph, visible_labels(ls, masks), ls.valid, masks, cfg, 3)
    ls2 = copy.deepcopy(ls)                                            # same label space, scrambled TEST labels
    ls2.Y[masks.test] = (1.0 - ls2.Y[masks.test]) * ls2.valid[masks.test]
    assert not np.array_equal(ls2.Y[masks.test], ls.Y[masks.test])
    b = fit_supervised(x, graph, visible_labels(ls2, masks), ls2.valid, masks, cfg, 3)
    for wa, wb in zip(a.model.W, b.model.W):
        assert torch.equal(wa, wb)
    np.testing.assert_array_equal(a.predict(x, graph), b.predict(x, graph))


def test_inner_validation_is_carved_from_outer_train_only_and_campaign_isolated():
    g, ls, f = toy_setup()
    tr, va = inner_validation_split(g, f.train, 0.2, seed=0)
    assert set(tr) | set(va) == set(f.train) and not set(tr) & set(va) and len(va) > 0
    assert not (set(va) | set(tr)) & set(f.test)
    camp = g.campaigns()
    assert not {camp[i] for i in tr} & {camp[i] for i in va}


# ------------------------------------------------------------------------------------------- training behaviour
def test_loss_decreases_on_a_learnable_task():
    g, ls, f = toy_setup()
    masks = make_masks(ls, f.train, f.val, f.test)
    graph = make_graph_data(fold_megits_adjacency(g, f.test_nodes_mask).adj, np.zeros(g.n, int))
    rng = np.random.RandomState(0)
    x = np.c_[ls.Y, 0.3 * rng.randn(g.n, 4)].astype(np.float32)           # label-informative features (toy only)
    res = fit_supervised(x, graph, visible_labels(ls, masks), ls.valid, masks,
                         SupervisedGCNConfig(lr=2e-2, max_epochs=100, dropout=0.2), 0)
    assert res.loss_history[-1] < 0.8 * res.loss_history[0]
    assert len(res.loss_history) == len(res.val_macro_f1_history) == len(res.val_loss_history) == len(res.epoch_times) == res.epochs_run
    p = res.predict(x, graph)
    assert ((p[masks.val] > 0.5) == (ls.Y[masks.val] > 0))[ls.valid[masks.val]].mean() > 0.9


def test_early_stopping_patience():
    g, ls, f = toy_setup()
    masks = make_masks(ls, f.train, f.val, f.test)
    graph = make_graph_data(fold_megits_adjacency(g, f.test_nodes_mask).adj, np.zeros(g.n, int))
    x = np.random.RandomState(0).randn(g.n, 5).astype(np.float32)         # uninformative -> validation stops improving
    res = fit_supervised(x, graph, visible_labels(ls, masks), ls.valid, masks,
                         SupervisedGCNConfig(lr=1e-2, patience=3, max_epochs=500), 0)
    assert res.stopped_early and res.epochs_run < 500 and res.epochs_run - res.best_epoch == 3
    # the returned model is the best-validation checkpoint, not the last epoch
    assert res.best_epoch <= res.epochs_run


def test_depth_variants_train_and_default_is_two_layers():
    g, ls, f = toy_setup()
    masks = make_masks(ls, f.train, f.val, f.test)
    graph = make_graph_data(fold_megits_adjacency(g, f.test_nodes_mask).adj, np.zeros(g.n, int))
    x = np.random.RandomState(0).randn(g.n, 5).astype(np.float32)
    assert SupervisedGCNConfig().layers == 2
    for L in (1, 2, 3, 4):
        res = fit_supervised(x, graph, visible_labels(ls, masks), ls.valid, masks,
                             SupervisedGCNConfig(layers=L, max_epochs=8), 0)
        assert len(res.model.W) == L and np.isfinite(res.loss_history).all()


def test_seed_determinism():
    g, ls, f = toy_setup()
    masks = make_masks(ls, f.train, f.val, f.test)
    graph = make_graph_data(fold_megits_adjacency(g, f.test_nodes_mask).adj, np.zeros(g.n, int))
    x = np.random.RandomState(0).randn(g.n, 5).astype(np.float32)
    run = lambda s: fit_supervised(x, graph, visible_labels(ls, masks), ls.valid, masks, SupervisedGCNConfig(**FAST), s)
    a, b, c = run(7), run(7), run(8)
    np.testing.assert_array_equal(a.predict(x, graph), b.predict(x, graph))
    assert a.loss_history == b.loss_history and a.best_epoch == b.best_epoch
    assert not np.array_equal(a.predict(x, graph), c.predict(x, graph))


# ------------------------------------------------------------------------------------------- full pipeline, both datasets
@pytest.fixture(scope="module", params=["synthetic", "semi_synthetic"])
def dataset(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(request.param)
    ds = generate("dev", 42) if request.param == "synthetic" else \
        generate("dev", 42, vulnerability_source="real", cve_pool=FIXTURE)
    write_dataset(ds, root / "dev")
    return load_dataset(request.param, profile="dev", root=root)


def test_fold_pipeline_on_synthetic_and_semi_synthetic(dataset):
    g, ls = dataset.tikg, dataset.labels
    assert dataset.is_synthetic
    fold = group_stratified_folds(g, ls, 3, 0)[0]
    cfg = SupervisedGCNConfig(max_epochs=60, lr=1e-2, patience=20)
    run = run_supervised_fold(g, ls, fold, cfg, seed=0)
    f = run.fit
    assert f.loss_history[-1] < f.loss_history[0]                              # loss decreases
    assert run.prob.shape == (g.n, ls.K) and np.isfinite(run.prob).all()       # valid dimensions / range
    assert run.prob.min() >= 0.0 and run.prob.max() <= 1.0
    assert (run.prob[~ls.valid] == 0).all()                                    # nothing predicted outside a type's classes
    assert len(f.model.W) == 2 and f.model.W[0].shape == (run.in_dim, 64) and f.model.W[1].shape == (64, ls.K)
    for k in ("macro_f1", "micro_f1", "macro_precision", "micro_recall"):
        assert 0.0 <= run.test_metrics[k] <= 1.0
    assert set(run.test_metrics_by_type) <= {t.value for t in EntityType}
    # heterogeneous label spaces: class counts per type are not all equal (Table 7 pattern)
    dims = {r["entity_type"]: r["output_dim"] for r in dimension_report(g, ls, run.in_dim, cfg)}
    assert len(set(dims.values())) > 1 and sum(dims.values()) == ls.K
    assert dims == {t.value: len(ls.classes[t]) for t in EntityType}
    # seed determinism through the whole pipeline
    again = run_supervised_fold(g, ls, fold, cfg, seed=0)
    np.testing.assert_array_equal(run.prob, again.prob)


def test_fold_pipeline_depth_variants_and_never_scores_labels_it_trained_on(dataset):
    g, ls = dataset.tikg, dataset.labels
    fold = group_stratified_folds(g, ls, 3, 0)[0]
    assert not set(fold.train) & set(fold.test) and not set(fold.val) & set(fold.test)
    for L in (1, 2, 3, 4):
        run = run_supervised_fold(g, ls, fold, SupervisedGCNConfig(layers=L, max_epochs=6, lr=1e-2), seed=1)
        assert len(run.fit.model.W) == L and run.prob.shape == (g.n, ls.K)
