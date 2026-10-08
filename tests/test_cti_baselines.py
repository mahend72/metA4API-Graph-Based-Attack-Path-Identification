"""AttacKG / LADDER external baselines: construction, equations, determinism, isolation, leakage, alignment, metrics."""
import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iocevaluator import cti_baselines as cb
from iocevaluator.cti_baselines import (AttacKGConfig, BASELINES, COMPONENTS, LADDERConfig, UNSUPPORTED_REASONS,
                                        _Template, _bigram_dice, alignment_score, build_templates, evaluate_baseline_cv,
                                        label_term_in_text_rate, metric_support, paired_with_reasons, run_attackg,
                                        run_baseline_comparison_dataset, run_ladder, select_baselines,
                                        undirected_adjacency, write_baseline_comparison)
from iocevaluator.datasets import load_dataset
from iocevaluator.evaluation import EvalConfig
from iocevaluator.splits import group_stratified_folds
from iocevaluator.supervised_gcn import LabelLeakageError, isolate_nodes, make_masks, visible_labels

SMALL = EvalConfig(max_folds=2, seeds=(0, 1))


@pytest.fixture(scope="module")
def ds():
    return load_dataset("synthetic", profile="dev")


@pytest.fixture(scope="module")
def fold_ctx(ds):
    f = group_stratified_folds(ds.tikg, ds.labels, 10, 0, 0.10)[0]
    masks = make_masks(ds.labels, f.train, f.val, f.test)
    return f, masks, visible_labels(ds.labels, masks), np.asarray(f.test_nodes_mask, dtype=bool)


@pytest.fixture(scope="module")
def comparison(ds):
    return cb.run_baseline_comparison(ds.tikg, ds.labels, select_baselines(), SMALL,
                                      provenance={"status": "synthetic"})


# ------------------------------------------------------------------------------------------------ construction
def test_construction_and_configuration():
    b = select_baselines()
    assert [x.name for x in b] == ["attackg", "ladder"]
    assert isinstance(BASELINES["attackg"].config, AttacKGConfig) and isinstance(BASELINES["ladder"].config, LADDERConfig)
    with pytest.raises(dataclasses.FrozenInstanceError):
        BASELINES["attackg"].config.gamma = 0.9
    with pytest.raises(KeyError):
        select_baselines(["attackg", "nope"])
    for x in b:
        info = x.info()
        assert info["config"] and info["n_trainable_parameters"] == 0 and info["seed_invariant"]
        assert set(info["components"]) == {"reproduced_exactly", "adapted_to_tikg", "not_reproducible"}
        assert all(info["components"][k] for k in info["components"])
    assert set(COMPONENTS) == {"attackg", "ladder"}


def test_hyperparameters_are_explicit_and_take_effect(ds, fold_ctx):
    f, masks, Yv, test_nodes = fold_ctx
    adj = undirected_adjacency(ds.tikg)
    a = run_attackg(ds.tikg, ds.labels, Yv, masks, test_nodes, adj, AttacKGConfig())
    b = run_attackg(ds.tikg, ds.labels, Yv, masks, test_nodes, adj, AttacKGConfig(gamma=0.9))
    assert not np.allclose(a.score, b.score)
    l1 = run_ladder(ds.tikg, ds.labels, Yv, masks, LADDERConfig(w_t=0.2))
    l2 = run_ladder(ds.tikg, ds.labels, Yv, masks, LADDERConfig(w_t=0.8))
    assert not np.allclose(l1.score, l2.score)


def test_no_manuscript_values_in_baseline_code():
    src = Path(cb.__file__).read_text(encoding="utf-8")
    for v in ("0.5588", "0.5396", "0.6772", "0.6838", "0.7551", "0.7696"):
        assert v not in src


# ------------------------------------------------------------------------------------------------ equations
def _tpl(sim, occ=(2.0, 1.0), deps=((0, 1, 1.0),)):
    return _Template(["a", "b"], np.array(occ), np.array(sim, dtype=np.float32), list(deps))


def test_attackg_alignment_equations_by_hand():
    cfg = AttacKGConfig(gamma=0.5, node_threshold=0.5)
    # graph nodes: 0 = classified node (type a), 1 = neighbour (type b); N = 2
    tpl = _tpl([[1.0, 0.0], [0.0, 0.0]])
    cats = ["a", "b"]
    hop1 = np.array([[0, 1], [1, 0]], dtype=float)
    # Gamma(0:0) = 0.5 + 0.5*1 = 1.0 ; Gamma(1:1) = 0.5 ; Eq3 = (1*2 + 0.5*1)/3 ; Eq4 = 1.0*0.5/1 * 1 / 1 ; Eq5 = mean
    expect = 0.5 * ((2.0 + 0.5) / 3.0 + 0.5)
    assert alignment_score(tpl, 0, [0, 1], cats, hop1, cfg) == pytest.approx(expect)
    hop2 = hop1 * 2                                           # the dependency is two hops long -> Eq.(4) halves
    assert alignment_score(tpl, 0, [0, 1], cats, hop2, cfg) == pytest.approx(0.5 * ((2.5) / 3.0 + 0.25))
    # a type mismatch gives 0 for that node (Eq.1) and no dependency contribution
    assert alignment_score(tpl, 0, [0, 1], ["a", "c"], hop1, cfg) == pytest.approx(0.5 * (2.0 / 3.0 + 0.0))
    # disconnected nodes: Cmin = infinity -> dependency contributes 0
    inf = np.array([[0, np.inf], [np.inf, 0]])
    assert alignment_score(tpl, 0, [0, 1], cats, inf, cfg) == pytest.approx(0.5 * (2.5 / 3.0))
    # candidate threshold removes weak candidates
    assert alignment_score(tpl, 0, [0, 1], cats, hop1, AttacKGConfig(gamma=0.5, node_threshold=0.9)) == \
        pytest.approx(0.5 * (2.0 / 3.0))
    # a template without dependencies reduces to the node-level score
    assert alignment_score(_tpl([[1.0, 0.0], [0.0, 0.0]], deps=()), 0, [0, 1], cats, hop1, cfg) == pytest.approx(2.5 / 3.0)


def test_char_bigram_dice():
    d = _bigram_dice(["abcd", "abcd", "wxyz", ""], 2 ** 12)
    assert d[0, 1] == pytest.approx(1.0) and d[0, 2] == pytest.approx(0.0) and d[0, 3] == 0.0
    assert np.allclose(d, d.T)


# ------------------------------------------------------------------------------------------------ outputs / dimensions
@pytest.mark.parametrize("name", ["attackg", "ladder"])
def test_output_dimensions_and_validity(ds, fold_ctx, name):
    f, masks, Yv, test_nodes = fold_ctx
    adj = undirected_adjacency(ds.tikg)
    out = (run_attackg(ds.tikg, ds.labels, Yv, masks, test_nodes, adj) if name == "attackg"
           else run_ladder(ds.tikg, ds.labels, Yv, masks))
    N, K = ds.tikg.n, ds.labels.K
    assert out.score.shape == out.pred.shape == (N, K)
    assert out.pred.dtype == bool and np.isfinite(out.score).all()
    assert not out.pred[~ds.labels.valid].any()                   # never predicts a label of another entity type
    assert (out.score[~ds.labels.valid] == 0).all()
    assert out.n_parameters == 0
    if name == "attackg":
        assert out.score.min() >= 0 and out.score.max() <= 1
    else:
        assert out.pred.sum(1).max() <= 1                          # LADDER maps a phrase to ONE technique
        assert out.threshold in LADDERConfig().tau_grid


def test_ladder_weighted_distance_is_linear_in_wt(ds, fold_ctx):
    f, masks, Yv, _ = fold_ctx
    s = {w: run_ladder(ds.tikg, ds.labels, Yv, masks, LADDERConfig(w_t=w)).score for w in (0.0, 0.5, 1.0)}
    v = ds.labels.valid
    assert np.allclose(s[0.5][v], 0.5 * s[0.0][v] + 0.5 * s[1.0][v])        # d = w_t*d_title + (1-w_t)*d_desc


def test_ladder_argmin_and_threshold_rule(ds, fold_ctx):
    f, masks, Yv, _ = fold_ctx
    out = run_ladder(ds.tikg, ds.labels, Yv, masks)
    dist = np.where(ds.labels.valid, 1.0 - out.score, np.inf)
    best = dist.argmin(1)
    for i in range(ds.tikg.n):
        if out.pred[i].any():
            assert out.pred[i].argmax() == best[i] and dist[i, best[i]] < out.threshold
        else:
            assert dist[i, best[i]] >= out.threshold - 1e-12


# ------------------------------------------------------------------------------------------------ determinism
@pytest.mark.parametrize("name", ["attackg", "ladder"])
def test_deterministic_execution(ds, name):
    cfg = EvalConfig(max_folds=1, seeds=(0, 3))
    a = evaluate_baseline_cv(ds.tikg, ds.labels, BASELINES[name], cfg)
    b = evaluate_baseline_cv(ds.tikg, ds.labels, BASELINES[name], cfg)
    cols = [c for c in a.runs.columns if a.runs[c].dtype.kind == "f"]
    pd.testing.assert_frame_equal(a.runs[cols], b.runs[cols])
    assert a.manifest["content_sha256"] == b.manifest["content_sha256"]
    assert a.runs[cols].iloc[0].equals(a.runs[cols].iloc[1])        # seed-invariant: seed rows of a fold are identical


# ------------------------------------------------------------------------------------------------ isolation / leakage
def _scramble_test_text(ds, test_nodes):
    t = copy.deepcopy(ds.tikg)
    for i in np.flatnonzero(test_nodes):
        e = t.entities[i]
        e.name = "ZZZ-" + e.name[::-1]
        e.attrs = {**e.attrs, "text": "qqq www eee " + str(e.attrs.get("text", ""))[::-1]}
    return t


def test_campaign_isolation_templates_ignore_held_out_campaigns(ds, fold_ctx):
    f, masks, Yv, test_nodes = fold_ctx
    assert test_nodes.any() and not (test_nodes & masks.visible).any()
    cfg = AttacKGConfig()
    adj = undirected_adjacency(ds.tikg)
    tadj = isolate_nodes(adj, test_nodes)
    t2 = _scramble_test_text(ds, test_nodes)
    n1, x1 = cb.node_text(ds.tikg)
    n2, x2 = cb.node_text(t2)
    A = build_templates(ds.tikg, ds.labels, Yv, masks, tadj, cfg, _bigram_dice(n1, cfg.bigram_features), _bigram_dice(x1, cfg.bigram_features))
    B = build_templates(t2, ds.labels, Yv, masks, tadj, cfg, _bigram_dice(n2, cfg.bigram_features), _bigram_dice(x2, cfg.bigram_features))
    assert A.keys() == B.keys() and A
    keep = ~test_nodes
    for c in A:
        assert A[c].keys == B[c].keys and A[c].deps == B[c].deps and np.array_equal(A[c].occ, B[c].occ)
        assert np.allclose(A[c].sim[:, keep], B[c].sim[:, keep])    # templates carry no held-out-campaign term


def test_ladder_scores_of_visible_nodes_ignore_held_out_text(ds, fold_ctx):
    f, masks, Yv, test_nodes = fold_ctx
    t2 = _scramble_test_text(ds, test_nodes)
    a = run_ladder(ds.tikg, ds.labels, Yv, masks)
    b = run_ladder(t2, ds.labels, Yv, masks)
    assert np.allclose(a.score[~test_nodes], b.score[~test_nodes])   # tf-idf / descriptions fitted on training nodes only
    assert a.threshold == b.threshold


def test_attackg_scores_of_unconnected_visible_nodes_ignore_held_out_text(ds, fold_ctx):
    f, masks, Yv, test_nodes = fold_ctx
    adj = undirected_adjacency(ds.tikg)
    t2 = _scramble_test_text(ds, test_nodes)
    a = run_attackg(ds.tikg, ds.labels, Yv, masks, test_nodes, adj)
    b = run_attackg(t2, ds.labels, Yv, masks, test_nodes, adj)
    far = [v for v in np.flatnonzero(~test_nodes)
           if not test_nodes[cb._ego(adj, int(v), AttacKGConfig().hops, AttacKGConfig().max_graph_nodes)].any()]
    assert len(far) > 0
    assert np.allclose(a.score[far], b.score[far])


@pytest.mark.parametrize("name", ["attackg", "ladder"])
def test_no_test_label_leakage(ds, fold_ctx, name):
    f, masks, Yv, test_nodes = fold_ctx
    adj = undirected_adjacency(ds.tikg)
    run = (lambda Y: run_attackg(ds.tikg, ds.labels, Y, masks, test_nodes, adj)) if name == "attackg" \
        else (lambda Y: run_ladder(ds.tikg, ds.labels, Y, masks))
    base = run(Yv)
    lab2 = dataclasses.replace(ds.labels, Y=ds.labels.Y.copy())
    rng = np.random.default_rng(0)
    lab2.Y[f.test] = rng.integers(0, 2, size=lab2.Y[f.test].shape)             # scramble every TEST label
    again = run(visible_labels(lab2, masks))
    assert np.array_equal(base.score, again.score) and np.array_equal(base.pred, again.pred)
    leaky = Yv.copy()
    leaky[f.test[0]] = ds.labels.Y[f.test[0]] + 1                               # a test row reaches the baseline
    with pytest.raises(LabelLeakageError):
        run(leaky)


def test_threshold_selected_on_validation_only(ds, fold_ctx):
    f, masks, Yv, _ = fold_ctx
    a = run_ladder(ds.tikg, ds.labels, Yv, masks)
    lab2 = dataclasses.replace(ds.labels, Y=ds.labels.Y.copy())
    lab2.Y[f.test] = 1 - lab2.Y[f.test]
    b = run_ladder(ds.tikg, ds.labels, visible_labels(lab2, masks), masks)
    assert a.threshold == b.threshold and a.val_macro_f1 == b.val_macro_f1


def test_label_term_in_text_diagnostic(ds):
    t = copy.deepcopy(ds.tikg)
    i = int(np.flatnonzero(ds.labels.labelled & (ds.labels.Y * ds.labels.valid).any(1))[0])
    k = int(np.flatnonzero((ds.labels.Y[i] > 0) & ds.labels.valid[i])[0])
    t.entities[i].attrs["text"] = "mentions " + ds.labels.names[k].split(":", 1)[1]
    assert label_term_in_text_rate(t, ds.labels) > label_term_in_text_rate(ds.tikg, ds.labels)


# ------------------------------------------------------------------------------------------------ alignment with the GCN
def test_fold_seed_alignment_with_gcn_and_paired_comparison(ds, comparison):
    r = comparison.runs
    ref = r[r.variant == "gcn"]
    for v in ("attackg", "ladder"):
        cur = r[r.variant == v]
        assert set(zip(cur.fold, cur.seed)) == set(zip(ref.fold, ref.seed)) == {(f, s) for f in (0, 1) for s in (0, 1)}
        m = cur.merge(ref, on=["fold", "seed"], suffixes=("", "_ref"))
        assert (m.n_test == m.n_test_ref).all() and (m.n_val == m.n_val_ref).all() and (m.n_train == m.n_train_ref).all()
    folds = [fi["test_campaigns"] if "test_campaigns" in fi else fi for fi in comparison.manifest["folds"]]
    assert [x.index for x in comparison.folds] == [0, 1] and folds
    paired = comparison.paired()
    assert set(paired.variant) == {"attackg", "ladder"}
    sub = paired[paired.metric.isin(["macro_f1", "micro_f1", "roc_auc", "pr_auc"])]
    assert (sub.n_pairs == 4).all() and (sub.n_folds == 2).all()
    for col in ("wilcoxon_p", "wilcoxon_p_holm", "significant_holm_0.05"):
        assert col in sub
    assert (sub.wilcoxon_p_holm >= sub.wilcoxon_p - 1e-12).all()
    for v in ("attackg", "ladder"):                                           # difference = variant - reference, per pair
        d = (r[r.variant == v].set_index(["fold", "seed"]).macro_f1 - ref.set_index(["fold", "seed"]).macro_f1).mean()
        assert sub[(sub.variant == v) & (sub.metric == "macro_f1")].mean_diff.iloc[0] == pytest.approx(d)


def test_paired_comparison_rejects_misaligned_pairs(comparison):
    r = comparison.runs
    bad = r[~((r.variant == "attackg") & (r.fold == 1) & (r.seed == 1))]
    with pytest.raises(ValueError):
        cb.AblationResult(bad, comparison.by_type, comparison.paths, comparison.timings, comparison.folds,
                          comparison.manifest, comparison.variants, {}, "gcn").paired()


# ------------------------------------------------------------------------------------------------ metrics
def test_metric_compatibility_and_unsupported_handling(comparison):
    r = comparison.runs
    for v in ("attackg", "ladder"):
        cur = r[r.variant == v]
        for m in ("macro_f1", "micro_f1", "roc_auc", "pr_auc"):
            assert cur[m].between(0, 1).all() and cur[m].notna().all()
        assert cur.n_parameters.eq(0).all()
        assert cur[["top5_hit_rate", "top10_hit_rate"]].isna().all().all()    # N/A, not manufactured
        assert not {"exact_path_hit5", "path_edge_f1"} & set(cur.columns)
    assert comparison.paths.empty
    bt = comparison.by_type
    for v in ("attackg", "ladder"):
        sub = bt[bt.variant == v]
        assert set(sub.entity_type) == set(bt[bt.variant == "gcn"].entity_type)
        assert {"macro_f1", "micro_f1", "roc_auc", "pr_auc"} <= set(sub.columns)
    tim = comparison.timings[comparison.timings.variant.isin(["attackg", "ladder"])]
    assert (tim.train_time_total_s >= 0).all() and (tim.inference_time_s > 0).all()


def test_metric_support_table_and_reasons(comparison):
    for name in ("attackg", "ladder"):
        s = metric_support(name)
        for m in ("macro_f1", "micro_f1", "roc_auc", "pr_auc", "per_entity_type"):
            assert s[m]["supported"]
        for m in UNSUPPORTED_REASONS:
            assert not s[m]["supported"] and s[m]["reason"]
    assert not metric_support("gcn")["top5_hit_rate"]["supported"]          # not requested here, reason recorded
    assert metric_support("gcn", topk=True, paths=True)["exact_path_hit5"]["supported"]
    p = paired_with_reasons(comparison)
    na = p[p.metric.isin(["top5_hit_rate", "exact_path_hit5"])]
    assert (na.n_pairs.fillna(0) == 0).all() or na.empty
    assert (p.loc[p.n_pairs.fillna(0) == 0, "na_reason"] != "").all()
    assert set(comparison.manifest["metric_support"]) == {"gcn", "attackg", "ladder"}


def test_written_outputs_and_manifest(comparison, tmp_path):
    paths = write_baseline_comparison(tmp_path, comparison)
    for name in ("ablation_runs.csv", "ablation_paired.csv", "metric_support.csv", "model_parameters.csv", "manifest.json"):
        assert paths[name].exists()
    man = json.loads(paths["manifest.json"].read_text())
    assert man["kind"] == "external_baseline_comparison" and man["reference_variant"] == "gcn"
    by = {v["name"]: v for v in man["variants"]}
    assert by["attackg"]["components"]["not_reproducible"] and by["ladder"]["config"]["w_t"] == 0.5
    assert pd.read_csv(paths["metric_support.csv"]).query("method == 'ladder' and metric == 'path_edge_f1'").supported.item() is False \
        or not pd.read_csv(paths["metric_support.csv"]).query("method == 'ladder' and metric == 'path_edge_f1'").supported.item()
    assert "wilcoxon_p_holm" in pd.read_csv(paths["ablation_paired.csv"]).columns


# ------------------------------------------------------------------------------------------------ small dev runs
@pytest.mark.parametrize("source", ["synthetic", "semi_synthetic"])
def test_small_dev_runs_on_synthetic_and_semi_synthetic(source):
    d = load_dataset(source, profile="dev")
    res = run_baseline_comparison_dataset(d, cfg=EvalConfig(max_folds=1, seeds=(0,)))
    r = res.runs
    assert set(r.variant) == {"gcn", "attackg", "ladder"} and len(r) == 3
    assert r[["macro_f1", "micro_f1", "roc_auc", "pr_auc"]].notna().all().all()
    assert (r.data_provenance == res.manifest["provenance"]["status"]).all()
    prov = res.manifest["provenance"]
    assert prov["source"] == source and prov["reportable_in_manuscript"] is False
