"""GAT / HGT baselines under the leakage-safe protocol: forward / training, heterogeneous typing, label masking, campaign
isolation, seed determinism, identical folds across models, parameter counts, paired comparisons (with Holm), dev run.
Synthetic data only - no manuscript value is used or compared."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from iocevaluator.ablation import paired_comparison
from iocevaluator.datasets import load_dataset
from iocevaluator.evaluation import EvalConfig, verify_fold
from iocevaluator.labels import build_label_space
from iocevaluator.megits import fold_megits_adjacency
from iocevaluator.metrics import holm_adjust
from iocevaluator.model_comparison import (MODELS, REFERENCE_MODEL, parameter_table, run_model_comparison_dataset,
                                           select_models, write_model_comparison)
from iocevaluator.models import make_graph_data
from iocevaluator.splits import group_stratified_folds
from iocevaluator.supervised_gcn import (SupervisedGCNConfig, build_supervised_model, count_parameters, dimension_report,
                                         fit_supervised, parameter_breakdown, prepare_fold, type_heads, visible_labels)
from iocevaluator.synthetic_tikg import generate, write_dataset
from iocevaluator.tikg import EntityType as ET
from .synthetic import make_synthetic

FIXTURE = Path(__file__).parent / "fixtures" / "cve_pool_sample.json"
KINDS = ("gcn", "gat", "hgt")


def cfg_for(kind, **kw):
    return SupervisedGCNConfig(model=kind, hidden=16 if kind == "hgt" else 64, **kw)


def toy(kind="gcn"):
    g, lab = make_synthetic(9)
    ls = build_label_space(g, lab)
    f = group_stratified_folds(g, ls, 3, 0)[0]
    ctx = prepare_fold(g, ls, f, model=kind)
    x = np.c_[ls.Y, 0.3 * np.random.RandomState(0).randn(g.n, 4)].astype(np.float32)     # learnable toy features
    return g, ls, f, ctx, x


# ------------------------------------------------------------------------------------------- config
def test_config_validation_and_defaults():
    assert SupervisedGCNConfig().model == "gcn" and SupervisedGCNConfig().heads == 4
    with pytest.raises(ValueError):
        SupervisedGCNConfig(model="sage")
    with pytest.raises(ValueError):
        SupervisedGCNConfig(model="gat", hidden=30)                              # not divisible by heads
    assert [m.name for m in select_models(["gat"])] == ["gcn", "gat"]            # reference always present
    assert MODELS["hgt"].hidden == 16 and MODELS["hgt_h64"].hidden == 64 and MODELS["gcn"].hidden == 64
    with pytest.raises(KeyError):
        select_models(["gnn"])


# ------------------------------------------------------------------------------------------- graph representation
def test_gat_uses_the_same_graph_as_the_gcn():
    g, ls, f, c_gcn, x = toy("gcn")
    c_gat = prepare_fold(g, ls, f, model="gat")
    np.testing.assert_array_equal(c_gcn.x, c_gat.x)                                # same features
    assert abs(c_gcn.train_adj - c_gat.train_adj).sum() == 0 and abs(c_gcn.infer_adj - c_gat.infer_adj).sum() == 0
    for a, b in ((c_gcn.train_graph, c_gat.train_graph), (c_gcn.infer_graph, c_gat.infer_graph)):
        assert torch.equal(a.src, b.src) and torch.equal(a.dst, b.dst)
    # edge set = support of the MeGiTS adjacency + self-loops (attention replaces the weights)
    n = g.n
    assert c_gat.infer_graph.src.shape[0] == c_gat.infer_adj.nnz + n
    assert c_gat.infer_graph.n_rel == 1


def test_hgt_uses_native_entity_and_relation_types():
    g, ls, f, ctx, _ = toy("hgt")
    gr = ctx.infer_graph
    order = {t.value: i for i, t in enumerate(ET)}
    assert gr.n_type == 7 and gr.node_type.tolist() == [order[t] for t in g.types]
    sigs = sorted(g.relation_adjacencies())
    assert gr.n_rel == 2 * len(sigs) + 1
    from iocevaluator.tikg import TRIPLET_SIGNATURES
    for r, sig in enumerate(sigs):
        s_t, _, t_t = TRIPLET_SIGNATURES[sig]
        for rid, (a, b) in ((2 * r, (s_t, t_t)), (2 * r + 1, (t_t, s_t))):           # forward / reverse relation
            m = gr.rel == rid
            assert m.any()
            assert set(gr.node_type[gr.src[m]].tolist()) == {order[a.value]} and set(gr.node_type[gr.dst[m]].tolist()) == {order[b.value]}
    loop = gr.rel == 2 * len(sigs)
    assert torch.equal(gr.src[loop], gr.dst[loop]) and int(loop.sum()) == g.n        # self-loop relation
    # no MeGiTS edges, relations are the observed triplets: edges = 2 * triplets + n self loops
    assert gr.src.shape[0] == 2 * len(g.triplets) + g.n
    # the training graph has no relation of any held-out node (except their self-loops)
    h = torch.as_tensor(f.test_nodes_mask)
    tg = ctx.train_graph
    real = tg.rel != 2 * len(sigs)
    assert not h[tg.src[real]].any() and not h[tg.dst[real]].any()
    assert (ctx.infer_graph.rel != 2 * len(sigs)).sum() > real.sum()                 # held-out relations return at inference


# ------------------------------------------------------------------------------------------- forward / dimensions / masking
@pytest.mark.parametrize("kind", KINDS)
def test_forward_dimensions_range_and_label_masking(kind):
    g, ls, f, ctx, x = toy(kind)
    cfg = cfg_for(kind)
    torch.manual_seed(0)
    m = build_supervised_model(cfg, x.shape[1], ls.K, ls.valid, ctx.infer_graph).eval()
    xt = torch.as_tensor(x)
    p = m.proba(xt, ctx.infer_graph)
    assert p.shape == (g.n, ls.K) and (p >= 0).all() and (p <= 1).all()
    assert (p[torch.as_tensor(~ls.valid)] == 0).all()                                # other types' labels are exactly 0
    assert m.forward(xt, ctx.infer_graph).shape == (g.n, ls.K)
    m.train()
    assert not torch.equal(m.proba(xt, ctx.infer_graph), m.proba(xt, ctx.infer_graph))   # dropout only when training


@pytest.mark.parametrize("kind", KINDS)
def test_output_dimensions_by_entity_type_and_masked_loss(kind):
    g, ls, f, ctx, x = toy(kind)
    cfg = cfg_for(kind, max_epochs=6, lr=1e-2)
    res = fit_supervised(x, ctx.train_graph, ctx.Y_visible, ls.valid, ctx.masks, cfg, 0)
    by_type = res.predict_by_type(x, ctx.infer_graph, type_heads(g, ls))
    for t, tp in by_type.items():
        assert tp.prob.shape == (len(g.type_nodes[t]), len(ls.classes[t]))
    assert sum(r["output_dim"] for r in dimension_report(g, ls, x.shape[1], cfg)) == ls.K
    # labels outside a node's type do not influence training: flip them in Y -> identical weights
    y2 = ctx.Y_visible.copy()
    flip = ~ls.valid
    flip[~ctx.masks.visible] = False                                                # hidden rows must stay label-free
    y2[flip] = 1.0
    res2 = fit_supervised(x, ctx.train_graph, y2, ls.valid, ctx.masks, cfg, 0)
    assert all(torch.equal(a, b) for a, b in zip(res.model.parameters(), res2.model.parameters()))


# ------------------------------------------------------------------------------------------- training / determinism / leakage
@pytest.mark.parametrize("kind", KINDS)
def test_training_reduces_loss_and_is_seed_deterministic(kind):
    g, ls, f, ctx, x = toy(kind)
    cfg = cfg_for(kind, lr=2e-2, max_epochs=40, dropout=0.2)
    a = fit_supervised(x, ctx.train_graph, ctx.Y_visible, ls.valid, ctx.masks, cfg, 3)
    b = fit_supervised(x, ctx.train_graph, ctx.Y_visible, ls.valid, ctx.masks, cfg, 3)
    c = fit_supervised(x, ctx.train_graph, ctx.Y_visible, ls.valid, ctx.masks, cfg, 4)
    assert a.loss_history[-1] < 0.9 * a.loss_history[0] and a.n_parameters == count_parameters(a.model) > 0
    np.testing.assert_array_equal(a.predict(x, ctx.infer_graph), b.predict(x, ctx.infer_graph))
    assert a.loss_history == b.loss_history and a.best_epoch == b.best_epoch
    assert not np.array_equal(a.predict(x, ctx.infer_graph), c.predict(x, ctx.infer_graph))


@pytest.mark.parametrize("kind", KINDS)
def test_early_stopping_uses_validation_only(kind):
    g, ls, f, ctx, _ = toy(kind)
    x = np.random.RandomState(0).randn(g.n, 5).astype(np.float32)                    # uninformative -> stops early
    res = fit_supervised(x, ctx.train_graph, ctx.Y_visible, ls.valid, ctx.masks,
                         cfg_for(kind, lr=1e-2, patience=3, max_epochs=300), 0)
    assert res.stopped_early and res.epochs_run - res.best_epoch == 3
    assert not ctx.Y_visible[ctx.masks.test].any()                                   # the trainer never saw test labels


@pytest.mark.parametrize("kind", KINDS)
def test_held_out_campaigns_cannot_influence_training(kind):
    """Scramble test labels and delete all held-out relations: trained weights stay bit-identical for every model."""
    import copy
    from iocevaluator.tikg import TIKG
    g, ls, f, ctx, x = toy(kind)
    h = f.test_nodes_mask
    kept = [(g.entities[t.s].id, t.r, g.entities[t.t].id) for t in g.triplets if not (h[t.s] or h[t.t])]
    g2 = TIKG(copy.deepcopy(g.entities), kept)
    ls2 = copy.deepcopy(ls)
    ls2.Y[f.test] = (1.0 - ls2.Y[f.test]) * ls2.valid[f.test]
    ctx2 = prepare_fold(g2, ls2, f, model=kind)
    cfg = cfg_for(kind, max_epochs=8, lr=1e-2)
    a = fit_supervised(x, ctx.train_graph, ctx.Y_visible, ls.valid, ctx.masks, cfg, 1)
    b = fit_supervised(x, ctx2.train_graph, ctx2.Y_visible, ls2.valid, ctx2.masks, cfg, 1)
    assert all(torch.equal(p, q) for p, q in zip(a.model.parameters(), b.model.parameters()))


# ------------------------------------------------------------------------------------------- parameter counts
def test_parameter_counts_match_the_architectures():
    d, K = 17, 40
    valid = np.ones((6, K), bool)
    g = make_graph_data(sp.eye(6, format="csr"), np.zeros(6, int))
    gcn = build_supervised_model(SupervisedGCNConfig(), d, K, valid, g)
    assert count_parameters(gcn) == d * 64 + 64 * K                                 # W0, W1 (no bias)
    gat = build_supervised_model(SupervisedGCNConfig(model="gat"), d, K, valid, g)
    assert count_parameters(gat) == (d * 64 + 2 * 4 * 16) + (64 * K + 2 * K)        # 4 heads x 16 + attention vectors; out layer
    gs = build_supervised_model(SupervisedGCNConfig(model="gat", hidden=32, heads=2), d, K, valid, g)
    assert count_parameters(gs) == (d * 32 + 2 * 2 * 16) + (32 * K + 2 * K)
    hg = make_graph_data(sp.eye(6, format="csr"), np.arange(6) % 3, {"T1": sp.eye(6, format="csr")})
    hgt = build_supervised_model(SupervisedGCNConfig(model="hgt", hidden=16), d, K, valid, hg)
    br = parameter_breakdown(hgt)
    assert sum(br.values()) == count_parameters(hgt) > 0
    n_type, n_rel, h, heads = hg.n_type, hg.n_rel, 16, 4
    layer = 4 * n_type * (h * h + h) + 2 * n_rel * heads * (h // heads) ** 2 + n_rel * heads + n_type    # q,k,v,o + W_att,W_msg + mu + skip
    assert count_parameters(hgt) == (d * h + h) + 2 * layer + (h * K + K)


# ------------------------------------------------------------------------------------------- Holm / paired comparison
def test_holm_adjustment_hand_example():
    adj = holm_adjust([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])                       # step-down with running maximum
    assert (adj >= np.array([0.01, 0.04, 0.03, 0.005])).all() and (adj <= 1).all()
    withnan = holm_adjust([0.01, np.nan, 0.04])
    np.testing.assert_allclose(withnan[[0, 2]], [0.02, 0.04])                       # m counts non-NaN only
    assert np.isnan(withnan[1])
    np.testing.assert_allclose(holm_adjust([0.6, 0.7]), [1.0, 1.0])                 # capped at 1
    assert len(holm_adjust([])) == 0


def test_paired_comparison_reports_raw_and_holm_adjusted_p_values():
    rows = []
    rng = np.random.RandomState(0)
    for fold in range(8):
        for seed in (0, 1):
            ref = 0.7 + 0.01 * fold + 0.001 * rng.randn()
            rows.append({"variant": "gcn", "fold": fold, "seed": seed, "macro_f1": ref})
            rows.append({"variant": "gat", "fold": fold, "seed": seed, "macro_f1": ref - 0.03 - 0.001 * fold})
            rows.append({"variant": "hgt", "fold": fold, "seed": seed, "macro_f1": ref + 0.0005 * rng.randn()})
    p = paired_comparison(pd.DataFrame(rows), "gcn", ["macro_f1"]).set_index("variant")
    assert {"wilcoxon_p", "wilcoxon_p_holm", "significant_holm_0.05"} <= set(p.columns)
    assert (p.wilcoxon_p_holm >= p.wilcoxon_p - 1e-12).all() and (p.wilcoxon_p_holm <= 1).all()
    ps = p.wilcoxon_p.sort_values()
    assert p.wilcoxon_p_holm.loc[ps.index[0]] == pytest.approx(min(1.0, 2 * ps.iloc[0]))   # m = 2 comparisons
    assert p.loc["gat", "significant_holm_0.05"] and p.loc["gat", "folds_worse"] == 8


# ------------------------------------------------------------------------------------------- dev-dataset run
@pytest.fixture(scope="module", params=["synthetic", "semi_synthetic"])
def ds(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(request.param)
    d = generate("dev", 42) if request.param == "synthetic" else \
        generate("dev", 42, vulnerability_source="real", cve_pool=FIXTURE)
    write_dataset(d, root / "dev")
    return load_dataset(request.param, profile="dev", root=root)


CFG = EvalConfig(seeds=(0, 1), max_folds=2, model=SupervisedGCNConfig(max_epochs=6, lr=1e-2))


@pytest.fixture(scope="module")
def comparison(ds):
    return run_model_comparison_dataset(ds, KINDS, CFG)


def test_identical_folds_and_features_across_models(ds):
    g, ls = ds.tikg, ds.labels
    folds = group_stratified_folds(g, ls, 10, 0)
    f = folds[0]
    verify_fold(g, ls, f)
    ctxs = {k: prepare_fold(g, ls, f, model=k) for k in KINDS}
    for k in ("gat", "hgt"):
        np.testing.assert_array_equal(ctxs[k].x, ctxs["gcn"].x)                       # train-only scaler, identical features
        np.testing.assert_array_equal(ctxs[k].masks.train, ctxs["gcn"].masks.train)
        np.testing.assert_array_equal(ctxs[k].masks.val, ctxs["gcn"].masks.val)
        np.testing.assert_array_equal(ctxs[k].masks.test, ctxs["gcn"].masks.test)
        np.testing.assert_array_equal(ctxs[k].Y_visible, ctxs["gcn"].Y_visible)
    camp = g.campaigns()
    assert not {camp[i] for i in f.train} & {camp[i] for i in f.test} and not {camp[i] for i in f.val} & {camp[i] for i in f.test}


def test_dev_model_comparison_run(ds, comparison):
    res, runs = comparison, comparison.runs
    assert set(runs.variant) == set(KINDS) and len(runs) == 3 * 2 * 2
    assert res.reference == REFERENCE_MODEL
    for m in ("macro_f1", "micro_f1", "roc_auc", "pr_auc", "n_parameters", "model"):
        assert m in runs.columns
    assert runs.macro_f1.between(0, 1).all() and (runs.n_parameters > 0).all()
    assert (res.timings.train_time_total_s > 0).all() and (res.timings.inference_time_s > 0).all()
    assert set(res.by_type.variant) == set(KINDS) and (runs.data_provenance == runs.data_provenance.iloc[0]).all()
    # identical (fold, seed) pairs and held-out campaigns for every model
    pairs = {v: set(zip(g.fold, g.seed)) for v, g in runs.groupby("variant")}
    assert len({frozenset(p) for p in pairs.values()}) == 1
    m0 = res.per_variant["gcn"].manifest["folds"]
    assert all(r.manifest["folds"] == m0 for r in res.per_variant.values())
    assert [res.per_variant[k].manifest["model"]["model"] for k in KINDS] == list(KINDS)
    # parameter counts reported per model; per-fold counts are recorded and consistent with the architectures
    pt = parameter_table(runs)
    assert set(pt.variant) == set(KINDS) and pt.set_index("variant").loc["gcn", "ratio_to_reference"] == pytest.approx(1.0)
    pc = pt.set_index("variant")
    assert pc.loc["gat", "params_mean"] > pc.loc["gcn", "params_mean"] and pc.loc["hgt", "ratio_to_reference"] > 1.0
    d_in = ds.tikg and prepare_fold(ds.tikg, ds.labels, res.folds[0]).x.shape[1]
    K = ds.labels.K
    g0 = runs[(runs.variant == "gcn") & (runs.fold == res.folds[0].index)].n_parameters.iloc[0]
    assert g0 == d_in * 64 + 64 * K
    # paired comparison against the GCN: identical pairs, raw + Holm p-values, GCN excluded
    paired = res.paired()
    assert set(paired.variant) == {"gat", "hgt"} and "wilcoxon_p_holm" in paired
    assert (paired[paired.metric == "macro_f1"].n_pairs == 4).all()
    man = res.manifest
    assert man["kind"] == "model_comparison" and man["reference_variant"] == "gcn" and len(man["parameter_counts"]) == 3
    assert {v["name"] for v in man["variants"]} == set(KINDS)
    gat_info = next(v for v in man["variants"] if v["name"] == "gat")
    assert "same MeGiTS adjacency" in gat_info["graph"]
    assert next(v for v in man["variants"] if v["name"] == "hgt")["graph"].startswith("native TIKG")


def test_model_comparison_outputs_and_determinism(ds, comparison, tmp_path):
    out = write_model_comparison(tmp_path, comparison)
    for n in ("ablation_runs.csv", "ablation_paired.csv", "model_parameters.csv", "manifest.json", "ablation_timings.csv"):
        assert out[n].exists()
    assert len(pd.read_csv(out["ablation_runs.csv"])) == len(comparison.runs)
    cfg = EvalConfig(seeds=(0,), max_folds=1, model=SupervisedGCNConfig(max_epochs=4, lr=1e-2))
    a, b = run_model_comparison_dataset(ds, ("gat", "hgt"), cfg), run_model_comparison_dataset(ds, ("gat", "hgt"), cfg)
    pd.testing.assert_frame_equal(a.runs, b.runs)
    assert a.manifest["content_sha256"] == b.manifest["content_sha256"]
