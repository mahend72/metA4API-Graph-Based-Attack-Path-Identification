"""Leakage-safe evaluation harness: split isolation / coverage / determinism, train-dependent artefacts recomputed per
fold, hand-computed metrics, aggregation, manifests and a 10-fold run on the dev datasets.
Synthetic data only - nothing here is a manuscript result."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from iocevaluator.datasets import load_dataset
from iocevaluator.evaluation import (EvalConfig, TopKCase, _run_metrics, aggregate_metric, build_manifest, evaluate_cv,
                                     evaluate_dataset, topk_for_fold, verify_fold, write_evaluation)
from iocevaluator.labels import build_label_space
from iocevaluator.splits import Fold, group_stratified_folds
from iocevaluator.supervised_gcn import LabelLeakageError, SupervisedGCNConfig, prepare_fold, type_heads, fit_predict
from iocevaluator.synthetic_tikg import generate, write_dataset
from iocevaluator.tikg import TIKG, Entity, EntityType as ET

FIXTURE = Path(__file__).parent / "fixtures" / "cve_pool_sample.json"
FAST = EvalConfig(seeds=(0, 1), model=SupervisedGCNConfig(max_epochs=12, lr=1e-2))


@pytest.fixture(scope="module", params=["synthetic", "semi_synthetic"])
def ds(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(request.param)
    d = generate("dev", 42) if request.param == "synthetic" else \
        generate("dev", 42, vulnerability_source="real", cve_pool=FIXTURE)
    write_dataset(d, root / "dev")
    return load_dataset(request.param, profile="dev", root=root)


@pytest.fixture(scope="module")
def folds(ds):
    return group_stratified_folds(ds.tikg, ds.labels, 10, 0)


# ------------------------------------------------------------------------------------------- defaults
def test_default_protocol_is_10_folds_x_5_seeds():
    c = EvalConfig()
    assert c.n_splits == 10 and list(c.seeds) == [0, 1, 2, 3, 4] and c.val_fraction == 0.10 and c.max_folds is None
    assert (c.model.layers, c.model.hidden, c.model.dropout, c.model.patience) == (2, 64, 0.5, 20)


# ------------------------------------------------------------------------------------------- split properties
def test_campaign_isolation_no_validation_test_leakage_and_coverage(ds, folds):
    g, ls = ds.tikg, ds.labels
    camp = g.campaigns()
    assert len(folds) == 10
    for f in folds:
        info = verify_fold(g, ls, f)                                       # raises on any overlap
        tr, va, te = set(f.train), set(f.val), set(f.test)
        assert not tr & te and not va & te and not tr & va
        assert va and va <= set(np.flatnonzero(ls.labelled))
        c_tr, c_va, c_te = ({camp[i] for i in s} for s in (tr, va, te))
        assert not c_tr & c_te and not c_va & c_te                         # zero train/test campaign overlap
        assert not c_va & c_tr                                             # validation carved out campaign-isolated
        # every node (labelled or not) of a test campaign is hidden from training
        assert set(camp[f.test_nodes_mask]) == c_te
        assert not f.test_nodes_mask[list(tr | va)].any()
        assert info["train_campaigns"] and info["test_campaigns"] and not set(info["train_campaigns"]) & set(info["test_campaigns"])
    # coverage: each labelled node is in exactly one test fold; each campaign in exactly one test fold
    tests = np.concatenate([f.test for f in folds])
    assert sorted(tests) == sorted(np.flatnonzero(ls.labelled)) and len(tests) == len(set(tests))
    per_fold_campaigns = [set(camp[f.test]) for f in folds]
    all_c = set().union(*per_fold_campaigns)
    assert all(sum(c in s for s in per_fold_campaigns) == 1 for c in all_c)
    assert all_c == {c for c in camp[ls.labelled]}


def test_splits_are_deterministic_and_seed_dependent(ds, folds):
    again = group_stratified_folds(ds.tikg, ds.labels, 10, 0)
    for a, b in zip(folds, again):
        for attr in ("train", "val", "test", "test_nodes_mask"):
            np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))
    # With only 12 dev campaigns over 10 folds the outer partition is nearly forced; the seed still changes the
    # shuffled assignment / inner validation carve-out.
    other = group_stratified_folds(ds.tikg, ds.labels, 10, 7)
    assert any(not np.array_equal(a.val, b.val) or not np.array_equal(a.test, b.test) for a, b in zip(folds, other))


def test_verify_fold_rejects_tampered_splits(ds, folds):
    g, ls, f = ds.tikg, ds.labels, folds[0]
    bad = Fold(f.index, f.train, f.val, np.concatenate([f.test, f.val[:1]]), f.test_nodes_mask)
    with pytest.raises(LabelLeakageError):
        verify_fold(g, ls, bad)
    swap = Fold(f.index, np.setdiff1d(f.train, f.test[:1]), f.val, f.test, f.test_nodes_mask)
    verify_fold(g, ls, swap)                                               # removing a node is harmless
    leak = Fold(f.index, np.concatenate([f.train, f.test[:1]]), f.val, f.test[1:], f.test_nodes_mask)
    with pytest.raises(LabelLeakageError):
        verify_fold(g, ls, leak)                                           # a test-campaign node in train


# ------------------------------------------------------------------------------------------- train-dependent artefacts
def test_test_nodes_cannot_influence_training(ds, folds):
    """Rewrite everything about the test campaigns (attributes, triplets, labels): the fold's features of train/val
    nodes, its training graph and the trained weights must be bit-identical."""
    g, ls, f = ds.tikg, ds.labels, folds[0]
    hidden = f.test_nodes_mask
    ents = copy.deepcopy(g.entities)
    for i in np.flatnonzero(hidden):
        ents[i].attrs = {**ents[i].attrs, "last_seen": "2099-01-01T00:00:00", "update_count": 10 ** 6,
                         "industries": ["zzz"], "source_countries": ["ZZ"], "target_countries": ["ZZ"]}
    trips = [(g.entities[t.s].id, t.r, g.entities[t.t].id) for t in g.triplets]
    kept = [t for t in trips if not (hidden[g.index[t[0]]] or hidden[g.index[t[2]]])]
    g2 = TIKG(ents, kept)
    ls2 = copy.deepcopy(ls)
    ls2.Y[f.test] = (1.0 - ls2.Y[f.test]) * ls2.valid[f.test]
    c1, c2 = prepare_fold(g, ls, f), prepare_fold(g2, ls2, f)
    vis = c1.masks.visible
    np.testing.assert_array_equal(c1.x[vis], c2.x[vis])                    # scaler / vocabularies from train U val only
    assert (c1.train_graph.A_norm.to_dense() == c2.train_graph.A_norm.to_dense()).all()
    # the training graph contains no edge to a test-campaign node: they only keep their self-loop
    dense = c1.train_graph.A_norm.to_dense().numpy()
    off = dense.copy()
    np.fill_diagonal(off, 0)
    assert not off[hidden].any() and not off[:, hidden].any()
    cfg = SupervisedGCNConfig(max_epochs=10, lr=1e-2)
    r1, _ = fit_predict(c1, ls, cfg, 5)
    r2, _ = fit_predict(c2, ls2, cfg, 5)
    import torch
    assert all(torch.equal(a, b) for a, b in zip(r1.model.W, r2.model.W))
    # the inference graph, in contrast, does contain test structure
    assert c1.infer_graph.A_norm._nnz() > c1.train_graph.A_norm._nnz()


def test_feature_scaler_is_fitted_on_training_nodes_only(ds, folds):
    from iocevaluator.tikg_features import FeatureBuilder
    g, ls, f = ds.tikg, ds.labels, folds[1]
    ctx = prepare_fold(g, ls, f)
    fb = FeatureBuilder()
    fb.fit(g, np.flatnonzero(ctx.masks.train))                              # train only - NOT train U validation
    np.testing.assert_allclose(fb.transform(g), ctx.x)
    fb_tv = FeatureBuilder().fit(g, np.flatnonzero(ctx.masks.visible))
    assert not np.allclose(fb_tv._mu, fb._mu)                               # validation nodes would change the scaler
    assert not np.allclose(FeatureBuilder().fit(g, np.arange(g.n))._mu, fb._mu)
    # perturbing ONLY validation (and test) node attributes leaves the features of training nodes untouched
    ents = copy.deepcopy(g.entities)
    for i in np.concatenate([f.val, f.test]):
        ents[i].attrs = {**ents[i].attrs, "last_seen": "2099-01-01T00:00:00", "update_count": 10 ** 6,
                         "industries": ["zzz"], "source_countries": ["ZZ"], "target_countries": ["ZZ"]}
    g2 = TIKG(ents, [(g.entities[t.s].id, t.r, g.entities[t.t].id) for t in g.triplets])
    ctx2 = prepare_fold(g2, ls, f)
    np.testing.assert_array_equal(ctx.x[f.train], ctx2.x[f.train])
    assert not np.array_equal(ctx.x[f.val], ctx2.x[f.val])                  # validation features themselves do change


def test_training_is_inductive_inference_is_transductive_in_heldout_topology(ds, folds):
    """Documented protocol: held-out-campaign topology never reaches training, but is used at inference."""
    g, ls, f = ds.tikg, ds.labels, folds[0]
    ctx = prepare_fold(g, ls, f)
    hidden = f.test_nodes_mask
    # (1) inductive training: the training graph has no edge touching a held-out node, the inference graph does
    assert ctx.train_adj[hidden].nnz == 0 and ctx.train_adj[:, hidden].nnz == 0
    assert ctx.infer_adj[hidden].nnz > 0
    # (2) with the SAME trained weights, held-out predictions differ between the two graphs -> inference uses the
    #     held-out topology (transductive in topology); training-node predictions are unaffected by held-out nodes
    res, prob_full = fit_predict(ctx, ls, SupervisedGCNConfig(max_epochs=15, lr=1e-2), 0)
    prob_isolated = res.predict(ctx.x, ctx.train_graph)
    assert not np.allclose(prob_full[f.test], prob_isolated[f.test])
    # (3) no held-out label is used by either: predictions are identical when held-out labels are scrambled
    ls2 = copy.deepcopy(ls)
    ls2.Y[f.test] = (1.0 - ls2.Y[f.test]) * ls2.valid[f.test]
    _, prob_full2 = fit_predict(prepare_fold(g, ls2, f), ls2, SupervisedGCNConfig(max_epochs=15, lr=1e-2), 0)
    np.testing.assert_array_equal(prob_full, prob_full2)


# ------------------------------------------------------------------------------------------- hand-computed metrics
def hand_case():
    E = [Entity("ta1", ET.THREAT_ACTOR, campaign="c1"), Entity("ta2", ET.THREAT_ACTOR, campaign="c1"),
         Entity("v1", ET.VULNERABILITY, campaign="c1"), Entity("v2", ET.VULNERABILITY, campaign="c1")]
    g = TIKG(E, [])
    ls = build_label_space(g, {"ta1": ["a"], "ta2": ["b"], "v1": ["c"], "v2": []},
                           classes={"threat_actor": ["a", "b", "z"], "vulnerability": ["c"]})
    assert ls.names == ["threat_actor:a", "threat_actor:b", "threat_actor:z", "vulnerability:c"]
    # columns: a, b, z (never positive), c ; entries outside a node's type are deliberately non-zero (must be ignored)
    prob = np.array([[0.9, 0.2, 0.1, 0.95],
                     [0.4, 0.8, 0.1, 0.95],
                     [0.9, 0.9, 0.9, 0.6],
                     [0.1, 0.1, 0.1, 0.7]])
    f = Fold(0, np.array([], int), np.array([], int), np.arange(4), np.ones(4, bool))
    return g, ls, prob, f


def test_hand_computed_metrics_with_type_valid_labels():
    g, ls, prob, f = hand_case()
    cfg = EvalConfig(ks=(1, 3))
    m, typed = _run_metrics(ls, g, prob, f, cfg, type_heads(g, ls), None, None)
    # a: TP1 -> F1 1 ; b: TP1 -> F1 1 ; z: no positives (unsupported) ; c: TP1 FP1 FN0 -> P .5 R 1 F1 2/3
    assert m["macro_f1"] == pytest.approx((1 + 1 + 2 / 3) / 3)             # supported labels only
    assert m["macro_f1_all_labels"] == pytest.approx((1 + 1 + 0 + 2 / 3) / 4)
    assert m["micro_precision"] == pytest.approx(3 / 4) and m["micro_recall"] == pytest.approx(1.0)
    assert m["micro_f1"] == pytest.approx(6 / 7)
    # AUC per label over the nodes that own it: a,b perfectly ranked, c inverted (AUC 0, AP 1/2); z skipped
    assert m["roc_auc"] == pytest.approx((1 + 1 + 0) / 3) and m["pr_auc"] == pytest.approx((1 + 1 + 0.5) / 3)
    by = {r["entity_type"]: r for r in typed}
    assert by["threat_actor"]["macro_f1"] == pytest.approx(1.0) and by["threat_actor"]["micro_f1"] == pytest.approx(1.0)
    assert by["threat_actor"]["roc_auc"] == pytest.approx(1.0) and by["threat_actor"]["n_test_nodes"] == 2
    assert by["vulnerability"]["macro_f1"] == pytest.approx(2 / 3) and by["vulnerability"]["micro_f1"] == pytest.approx(2 / 3)
    assert by["vulnerability"]["roc_auc"] == pytest.approx(0.0) and by["vulnerability"]["pr_auc"] == pytest.approx(0.5)
    assert np.isnan(m["top1_hit_rate"])                                    # no expert cases supplied -> not computed


def test_top_k_hit_rate_hand_example():
    g, ls, prob, f = hand_case()
    score = np.array([0.1, 0.2, 0.9, 0.5])                                  # ranking: v1, v2, ta2, ta1
    cases = [TopKCase("q1", "c1", frozenset({"v1"})), TopKCase("q2", "c1", frozenset({"ta1"})),
             TopKCase("q3", "elsewhere", frozenset({"v1"}))]               # q3: campaign not in the test fold -> ignored
    r = topk_for_fold(g, cases, f, score, (1, 3, 4))
    assert r["topk_n_cases"] == 2
    assert r["top1_hit_rate"] == pytest.approx(0.5)                         # q1 hits at rank 1, q2 does not
    assert r["top3_hit_rate"] == pytest.approx(0.5)                         # ta1 is ranked 4th
    assert r["top4_hit_rate"] == pytest.approx(1.0)
    # ties are broken by entity id (deterministic)
    tie = topk_for_fold(g, [TopKCase("q", "c1", frozenset({"ta1"}))], f, np.zeros(4), (1,))
    assert tie["top1_hit_rate"] == 1.0                                      # "ta1" sorts first
    m, _ = _run_metrics(ls, g, prob, f, EvalConfig(ks=(1,)), type_heads(g, ls), cases[:2], score)
    assert m["top1_hit_rate"] == pytest.approx(0.5) and m["topk_n_cases"] == 2


def test_aggregation_hand_example():
    vals = {0: [.6, .8], 1: [.7, .7], 2: [.5, .9], 3: [.6, .6]}
    df = pd.DataFrame([{"fold": f, "seed": s, "m": v} for f, vs in vals.items() for s, v in enumerate(vs)])
    a = aggregate_metric(df, "m")
    assert a["n_runs"] == 8 and a["n_folds"] == 4
    assert a["mean"] == pytest.approx(0.675) and a["std"] == pytest.approx(np.sqrt(0.115 / 7))     # ddof = 1
    assert a["fold_mean"] == pytest.approx(0.675) and a["fold_std"] == pytest.approx(0.05)
    h = stats.t.ppf(0.975, 3) * 0.05 / 2
    assert a["ci95_folds_t"] == pytest.approx([0.675 - h, 0.675 + h])
    lo, hi = a["ci95_folds_boot"]
    assert 0.6 <= lo <= 0.675 <= hi <= 0.7
    assert a["ci95_runs_t"][0] < a["mean"] < a["ci95_runs_t"][1]
    assert aggregate_metric(df, "m") == a                                  # deterministic bootstrap
    nan = pd.DataFrame({"fold": [0, 1], "seed": [0, 0], "m": [np.nan, 0.5]})
    assert aggregate_metric(nan, "m")["n_runs"] == 1                       # NaN runs (e.g. undefined AUC) are skipped


# ------------------------------------------------------------------------------------------- 10-fold execution + manifests
@pytest.fixture(scope="module")
def result(ds):
    return evaluate_dataset(ds, FAST)


def test_ten_fold_run_on_dev_dataset(ds, result):
    runs, bt = result.runs, result.by_type
    assert len(result.folds) == 10 and len(runs) == 10 * 2
    assert sorted(runs.fold.unique()) == list(range(10)) and sorted(runs.seed.unique()) == [0, 1]
    for m in ("macro_f1", "micro_f1", "roc_auc", "pr_auc", "macro_f1_all_labels"):
        v = runs[m].dropna()
        assert len(v) and ((v >= 0) & (v <= 1)).all(), m
    assert {"top5_hit_rate", "top10_hit_rate"} <= set(runs.columns) and runs.top5_hit_rate.isna().all()
    assert set(bt.entity_type) <= {t.value for t in ET} and len(bt) > 0
    assert (runs.n_test > 0).all() and runs.groupby("fold").n_test.nunique().max() == 1
    s = result.summary()
    assert set(s["by_entity_type"]) <= {t.value for t in ET}
    for m in ("macro_f1", "micro_f1", "roc_auc", "pr_auc"):
        a = s["overall"][m]
        assert a["n_folds"] == 10 and a["n_runs"] <= 20 and a["ci95_folds_t"][0] <= a["fold_mean"] <= a["ci95_folds_t"][1]


def test_outputs_and_manifest_contents(ds, result, tmp_path):
    paths = write_evaluation(tmp_path, result)
    for name in ("results_per_run.csv", "results_per_fold.csv", "results_per_seed.csv", "results_per_type.csv",
                 "timings.csv", "summary.json", "manifest.json"):
        assert paths[name].exists()
    assert len(pd.read_csv(paths["results_per_fold.csv"])) == 10 and len(pd.read_csv(paths["results_per_seed.csv"])) == 2
    man = json.loads(paths["manifest.json"].read_text())
    assert man["dataset"]["profile"] == "dev" and man["dataset"]["is_synthetic"] is True
    assert man["dataset"]["fingerprint_sha256"] and man["protocol"]["seeds"] == [0, 1]
    assert man["protocol"]["n_splits"] == 10 and man["protocol"]["folds_run"] == 10 and man["protocol"]["fold_seed"] == 0
    assert len(man["protocol"]["runs"]) == 20
    assert len(man["folds"]) == 10
    for fi in man["folds"]:
        assert fi["train_campaigns"] and fi["val_campaigns"] and fi["test_campaigns"]
        assert not set(fi["train_campaigns"]) & set(fi["test_campaigns"])
        assert not set(fi["val_campaigns"]) & set(fi["test_campaigns"])
    assert man["model"]["layers"] == 2 and man["model"]["patience"] == 20 and "exclude_fields" in man["features"]
    assert set(man["chi_weights"]) == {f"chi{k}" for k in range(1, 21)}
    assert all(abs(w - 1 / 20) < 1e-12 for w in man["chi_weights"].values()) and len(man["chi_structures"]) == 20
    assert {"git", "python", "torch", "numpy"} <= set(man["environment"]) and "commit" in man["environment"]["git"]


def test_run_is_deterministic_and_manifest_content_is_stable(ds):
    cfg = EvalConfig(seeds=(3,), max_folds=2, model=SupervisedGCNConfig(max_epochs=10, lr=1e-2))
    a, b = evaluate_dataset(ds, cfg), evaluate_dataset(ds, cfg)
    pd.testing.assert_frame_equal(a.runs, b.runs)
    pd.testing.assert_frame_equal(a.by_type, b.by_type)
    assert a.manifest["content_sha256"] == b.manifest["content_sha256"]
    strip = lambda m: {k: v for k, v in m.items() if k != "environment"}
    assert json.dumps(strip(a.manifest), sort_keys=True) == json.dumps(strip(b.manifest), sort_keys=True)
    c = evaluate_dataset(ds, EvalConfig(seeds=(4,), max_folds=2, model=cfg.model))
    assert c.manifest["content_sha256"] != a.manifest["content_sha256"]     # a different seed is a different experiment


def test_expert_cases_flow_through_the_harness(ds):
    """Exercise the top-k path with stand-in cases (test data only: the highest-CVSS vulnerability of each campaign)."""
    g = ds.tikg
    from iocevaluator.ranking import ProbabilityBaselineRanker, RankerConfig, ThreatPrioritisationRanker
    cases = []
    for c in sorted({c for c in g.campaigns() if c}):
        vs = [i for i in g.type_nodes[ET.VULNERABILITY] if g.campaigns()[i] == c and ds.severity_known[i]]
        if vs:
            best = max(vs, key=lambda i: (ds.severity[i], g.entities[i].id))
            cases.append(TopKCase(f"case-{c}", c, frozenset({g.entities[best].id})))
    assert cases
    r = evaluate_dataset(ds, EvalConfig(seeds=(0,), max_folds=3, ks=(1, 5, 10),
                                        model=SupervisedGCNConfig(max_epochs=8, lr=1e-2)), topk_cases=cases)
    h = r.runs[["top1_hit_rate", "top5_hit_rate", "top10_hit_rate", "topk_n_cases"]]
    assert (h.topk_n_cases > 0).all()
    assert ((h.top1_hit_rate <= h.top5_hit_rate) & (h.top5_hit_rate <= h.top10_hit_rate)).all()
    # the default ranker is the manuscript's threat prioritisation, recorded in the manifest
    rk = r.manifest["ranking"]
    assert rk["name"] == "threat_prioritisation" and rk["manuscript_ranker"] is True
    assert rk["alpha"] == 0.5 and rk["tau"] is None and rk["ec_scope"] == "type" and rk["missing_severity"] == "ec_only"
    # explicit baseline placeholder; alpha tuned on validation campaigns only
    cfg = EvalConfig(seeds=(0,), max_folds=3, ks=(1, 5), model=SupervisedGCNConfig(max_epochs=8, lr=1e-2))
    base = evaluate_dataset(ds, cfg, topk_cases=cases, ranker=ProbabilityBaselineRanker())
    assert base.manifest["ranking"] == {"name": "baseline_max_gcn_probability", "manuscript_ranker": False}
    tuned = evaluate_dataset(ds, cfg, topk_cases=cases, ranker=ThreatPrioritisationRanker(RankerConfig(alpha=None)))
    assert tuned.runs.ranker_alpha.notna().any() and ((tuned.runs.ranker_alpha.dropna() >= 0) & (tuned.runs.ranker_alpha.dropna() <= 1)).all()
    assert tuned.manifest["ranking"]["alpha"] is None
    again = evaluate_dataset(ds, cfg, topk_cases=cases, ranker=ThreatPrioritisationRanker(RankerConfig(alpha=None)))
    pd.testing.assert_frame_equal(tuned.runs, again.runs)
