"""Ablation framework: variant definitions, adjacency construction per variant, chi membership, binary vs weighted
semantics, reduced-label sampling, fold/seed pairing, depth variants, paired comparison, provenance, result integrity,
and a complete ablation run on the dev dataset.  No manuscript value is used anywhere (synthetic data only)."""
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from iocevaluator.ablation import (ALL_IDS, GRAPH_IDS, GROUPS, PATH_IDS, REFERENCE, VARIANTS, build_variant_adjacency,
                                   paired_comparison, provenance_info, run_ablation_dataset, select_variants,
                                   write_ablation)
from iocevaluator.datasets import load_dataset
from iocevaluator.evaluation import EvalConfig
from iocevaluator.labels import build_label_space
from iocevaluator.megits import megits_adjacency
from iocevaluator.metagraphs import STRUCTURES, commuting_matrices, get_structure, global_matrix, select_structures
from iocevaluator.megits import structure_similarity
from iocevaluator.path_tracing import PathConfig
from iocevaluator.splits import Fold, group_stratified_folds
from iocevaluator.supervised_gcn import (LabelLeakageError, SupervisedGCNConfig, make_masks, prepare_fold,
                                         reduce_training_labels, visible_labels)
from iocevaluator.synthetic_tikg import generate, write_dataset
from .synthetic import make_synthetic

FIXTURE = Path(__file__).parent / "fixtures" / "cve_pool_sample.json"
CFG = EvalConfig(seeds=(0, 1), max_folds=2, model=SupervisedGCNConfig(max_epochs=6, lr=1e-2),
                 paths=PathConfig(max_edges=3, priority_top_n=5, max_paths_per_seed=200, max_candidates=200))


@pytest.fixture(scope="module", params=["synthetic", "semi_synthetic"])
def ds(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(request.param)
    d = generate("dev", 42) if request.param == "synthetic" else \
        generate("dev", 42, vulnerability_source="real", cve_pool=FIXTURE)
    write_dataset(d, root / "dev")
    return load_dataset(request.param, profile="dev", root=root)


def no_test_fold(tikg):
    return SimpleNamespace(test_nodes_mask=np.zeros(tikg.n, dtype=bool))


# ------------------------------------------------------------------------------------------- definitions / chi membership
def test_variant_registry_matches_the_requested_ablations():
    names = set(VARIANTS)
    assert {"original_adjacency", "binary_semantic", "megits_paths_only", "megits_graphs_only", REFERENCE} <= names
    assert {f"chi{k}" for k in range(1, 21)} <= names
    assert {"megits_full_1layer", "megits_full_3layer", "megits_full_4layer",
            "megits_full_lf25", "megits_full_lf50", "megits_full_lf75"} <= names
    assert len(VARIANTS) == 31
    assert VARIANTS[REFERENCE].structure_ids == tuple(range(1, 21)) and VARIANTS[REFERENCE].layers == 2
    assert VARIANTS[REFERENCE].label_fraction == 1.0 and VARIANTS[REFERENCE].adjacency == "megits"
    assert [VARIANTS[f"megits_full_{L}layer"].layers for L in (1, 3, 4)] == [1, 3, 4]
    assert [VARIANTS[f"megits_full_lf{p}"].label_fraction for p in (25, 50, 75)] == [0.25, 0.5, 0.75]
    assert all(VARIANTS[f"megits_full_lf{p}"].structure_ids == ALL_IDS for p in (25, 50, 75))
    assert VARIANTS["original_adjacency"].structure_ids == () and VARIANTS["original_adjacency"].weights == {}
    for k in range(1, 21):
        assert VARIANTS[f"chi{k}"].structure_ids == (k,) and VARIANTS[f"chi{k}"].weights == {k: 1.0}
    assert VARIANTS["binary_semantic"].adjacency == "binary" and VARIANTS["binary_semantic"].structure_ids == ALL_IDS
    # groups
    assert set(GROUPS["all"]) == names and len(GROUPS["all"]) == 31
    assert [v.name for v in select_variants(["depth"])] == ["megits_full_1layer", REFERENCE, "megits_full_3layer", "megits_full_4layer"]
    assert REFERENCE in [v.name for v in select_variants(["chi3"])]                # reference is always added
    with pytest.raises(KeyError):
        select_variants(["nope"])


def test_meta_path_vs_meta_graph_grouping_follows_figure3():
    """Fig. 3: chi_1..chi_11 are the base symmetric meta-paths, chi_12..chi_20 the composite meta-graphs."""
    assert PATH_IDS == tuple(range(1, 12)) and GRAPH_IDS == tuple(range(12, 21))
    assert VARIANTS["megits_paths_only"].structure_ids == tuple(range(1, 12))
    assert VARIANTS["megits_graphs_only"].structure_ids == tuple(range(12, 21))
    assert {s.id for s in STRUCTURES if s.kind == "path"} == set(range(1, 12))
    assert set(PATH_IDS) | set(GRAPH_IDS) == set(ALL_IDS) and not set(PATH_IDS) & set(GRAPH_IDS)


def test_weights_are_uniform_and_renormalised_over_the_active_subset():
    for name, n in (("megits_paths_only", 11), ("megits_graphs_only", 9), (REFERENCE, 20), ("binary_semantic", 20)):
        w = VARIANTS[name].weights
        assert len(w) == n and sum(w.values()) == pytest.approx(1.0) and all(x == pytest.approx(1 / n) for x in w.values())
    info = VARIANTS["megits_paths_only"].info()
    assert info["active_structures"] == list(range(1, 12)) and len(info["chi_weights"]) == 11 and info["binary"] is False
    assert VARIANTS["binary_semantic"].info()["binary"] is True


# ------------------------------------------------------------------------------------------- adjacency construction
def test_adjacency_for_every_variant_matches_its_definition(ds):
    g = ds.tikg
    fold = no_test_fold(g)                                                     # no held-out nodes: exact definition
    adjs = {n: build_variant_adjacency(v, g, fold) for n, v in VARIANTS.items() if v.layers == 2 and v.label_fraction == 1.0}
    assert len(adjs) == 25
    # original adjacency: observed relations only, symmetric, binary, no self-loops, includes cross-type edges
    orig = adjs["original_adjacency"]
    assert abs(orig - g.observed_adjacency()).sum() == 0 and orig.diagonal().sum() == 0 and set(orig.data) == {1.0}
    assert abs(orig - orig.T).sum() == 0
    coo = orig.tocoo()
    assert (g.types[coo.row] != g.types[coo.col]).any()
    # MeGiTS variants only connect nodes of the endpoint type of an active structure (chi are single-type)
    for name, A in adjs.items():
        if name == "original_adjacency":
            continue
        v = VARIANTS[name]
        coo = A.tocoo()
        assert (g.types[coo.row] == g.types[coo.col]).all() and A.diagonal().sum() == 0
        active_types = {get_structure(k).node_type.value for k in v.structure_ids}
        assert set(g.types[coo.row]) <= active_types
        assert abs(A - A.T).sum() < 1e-6 and A.nnz > 0
    for k in range(1, 21):
        assert set(g.types[adjs[f"chi{k}"].nonzero()[0]]) <= {get_structure(k).node_type.value}
    # weighted full / paths-only / graphs-only equal the independent sum_k w_k S_k over exactly their structures
    C = commuting_matrices(g)
    def expected(ids):
        tot = sp.csr_matrix((g.n, g.n))
        for k in ids:
            S = structure_similarity(C[k])
            S = S - sp.diags(S.diagonal())
            tot = tot + global_matrix(g, get_structure(k), S) / len(ids)
        tot.setdiag(0)
        tot.eliminate_zeros()
        return tot
    for name in ("megits_paths_only", "megits_graphs_only", REFERENCE):
        assert abs(adjs[name] - expected(VARIANTS[name].structure_ids)).max() < 1e-6, name
    # the two groups partition the full set of structures: supports are a union
    sup = lambda A: (A > 0).astype(int)
    union = sup(adjs["megits_paths_only"]) + sup(adjs["megits_graphs_only"])
    assert abs((union > 0).astype(int) - sup(adjs[REFERENCE])).sum() == 0
    assert adjs["megits_paths_only"].nnz < adjs[REFERENCE].nnz or adjs["megits_graphs_only"].nnz < adjs[REFERENCE].nnz


def test_binary_vs_weighted_semantics(ds):
    g = ds.tikg
    fold = no_test_fold(g)
    w = build_variant_adjacency(VARIANTS[REFERENCE], g, fold)
    b = build_variant_adjacency(VARIANTS["binary_semantic"], g, fold)
    assert set(b.data) == {1.0}                                                # unweighted
    assert len(set(np.round(w.data, 6))) > 1 and w.data.max() <= 1.0 + 1e-6    # weighted: graded similarities
    assert abs((w > 0).astype(float) - b).sum() == 0                           # identical connectivity, weights dropped
    # an instance of a single chi_k is enough for the binary link (union of supports)
    c5 = build_variant_adjacency(VARIANTS["chi5"], g, fold)
    assert ((c5 > 0).astype(int) - (b > 0).astype(int)).max() <= 0


def test_held_out_nodes_never_enter_variant_training_graphs(ds):
    g, ls = ds.tikg, ds.labels
    f = group_stratified_folds(g, ls, 10, 0)[0]
    h = f.test_nodes_mask
    for name in ("original_adjacency", "binary_semantic", "megits_paths_only", "chi8", "chi2"):
        ctx = prepare_fold(g, ls, f, adj=build_variant_adjacency(VARIANTS[name], g, f))
        assert ctx.train_adj[h].nnz == 0 and ctx.train_adj[:, h].nnz == 0           # no held-out node while training
        if name in ("original_adjacency", "binary_semantic"):
            assert ctx.infer_adj[h].nnz > 0                                          # but present at inference


# ------------------------------------------------------------------------------------------- reduced-label sampling
def toy_ctx():
    g, lab = make_synthetic(9)
    ls = build_label_space(g, lab)
    f = group_stratified_folds(g, ls, 3, 0)[0]
    return g, ls, f, prepare_fold(g, ls, f)


def test_reduced_label_sampling_is_deterministic_nested_and_train_only():
    g, ls, f, ctx = toy_ctx()
    n = int(ctx.masks.train.sum())
    subsets = {fr: reduce_training_labels(ctx, ls, f, 0, fr) for fr in (0.25, 0.5, 0.75, 1.0)}
    sizes = {fr: int(c.masks.train.sum()) for fr, c in subsets.items()}
    assert sizes == {0.25: round(0.25 * n), 0.5: round(0.5 * n), 0.75: round(0.75 * n), 1.0: n}
    tr = {fr: set(np.flatnonzero(c.masks.train)) for fr, c in subsets.items()}
    assert tr[0.25] < tr[0.5] < tr[0.75] < tr[1.0] and tr[1.0] == set(np.flatnonzero(ctx.masks.train))   # nested
    again = reduce_training_labels(ctx, ls, f, 0, 0.5)
    np.testing.assert_array_equal(again.masks.train, subsets[0.5].masks.train)                          # deterministic
    assert not np.array_equal(reduce_training_labels(ctx, ls, f, 1, 0.5).masks.train, subsets[0.5].masks.train)  # seed
    assert subsets[1.0] is ctx
    other_fold = group_stratified_folds(g, ls, 3, 0)[1]
    assert not np.array_equal(reduce_training_labels(prepare_fold(g, ls, other_fold), ls, other_fold, 0, 0.5).masks.train,
                              subsets[0.5].masks.train)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            reduce_training_labels(ctx, ls, f, 0, bad)


def test_reduced_labels_never_touch_validation_test_or_campaign_isolation():
    g, ls, f, ctx = toy_ctx()
    camp = g.campaigns()
    for fr in (0.25, 0.5, 0.75):
        c = reduce_training_labels(ctx, ls, f, 3, fr)
        # validation / test masks identical; validation labels identical; test labels still hidden from training
        np.testing.assert_array_equal(c.masks.val, ctx.masks.val)
        np.testing.assert_array_equal(c.masks.test, ctx.masks.test)
        np.testing.assert_array_equal(c.Y_visible[c.masks.val], ctx.Y_visible[ctx.masks.val])
        assert not c.Y_visible[c.masks.test].any()
        np.testing.assert_array_equal(c.Y_visible[c.masks.train], ls.Y[c.masks.train])      # kept labels are the true ones
        dropped = ctx.masks.train & ~c.masks.train
        assert dropped.any() and not c.Y_visible[dropped].any()                              # dropped labels are withheld
        assert not (c.masks.train & (c.masks.val | c.masks.test)).any()
        # campaign isolation: the retained training nodes still share no campaign with validation / test
        tc = {camp[i] for i in np.flatnonzero(c.masks.train)}
        assert not tc & {camp[i] for i in np.flatnonzero(c.masks.test)} and not tc & {camp[i] for i in np.flatnonzero(c.masks.val)}
        # everything else about the fold is unchanged: features, graphs, the unmodified ls
        np.testing.assert_array_equal(c.x, ctx.x)
        assert c.train_graph is ctx.train_graph and c.infer_graph is ctx.infer_graph
    make_masks(ls, np.flatnonzero(reduce_training_labels(ctx, ls, f, 0, 0.25).masks.train), f.val, f.test)   # still valid masks


# ------------------------------------------------------------------------------------------- paired comparison
def test_paired_comparison_hand_example_and_alignment():
    rows = []
    for fold in range(4):
        for seed in (0, 1):
            ref = 0.70 + 0.01 * fold
            rows.append({"variant": REFERENCE, "fold": fold, "seed": seed, "macro_f1": ref, "micro_f1": ref})
            rows.append({"variant": "v_better", "fold": fold, "seed": seed, "macro_f1": ref + 0.02 + 0.01 * seed, "micro_f1": ref})
            rows.append({"variant": "v_worse", "fold": fold, "seed": seed, "macro_f1": ref - 0.05, "micro_f1": np.nan})
    runs = pd.DataFrame(rows)
    p = paired_comparison(runs, REFERENCE, ["macro_f1", "micro_f1"]).set_index(["variant", "metric"])
    b = p.loc[("v_better", "macro_f1")]
    assert b.n_pairs == 8 and b.n_folds == 4 and b.mean_diff == pytest.approx(0.025)
    assert b.folds_better == 4 and b.folds_worse == 0 and b.ci95_folds_t_lo == pytest.approx(0.025) == b.ci95_folds_t_hi
    assert p.loc[("v_worse", "macro_f1")].mean_diff == pytest.approx(-0.05) and p.loc[("v_worse", "macro_f1")].folds_worse == 4
    assert p.loc[("v_better", "micro_f1")].mean_diff == pytest.approx(0.0) and p.loc[("v_better", "micro_f1")].folds_tied == 4
    assert p.loc[("v_worse", "micro_f1")].n_pairs == 0                           # undefined metric -> no pairs
    assert REFERENCE not in set(p.reset_index().variant)
    # alignment: a variant that is missing a (fold, seed) pair cannot be compared
    with pytest.raises(ValueError):
        paired_comparison(runs[~((runs.variant == "v_better") & (runs.fold == 3) & (runs.seed == 1))], REFERENCE, ["macro_f1"])
    with pytest.raises(KeyError):
        paired_comparison(runs[runs.variant != REFERENCE], REFERENCE)
    # significance test over per-fold differences is reported
    assert "wilcoxon_p" in p.columns and "significant_0.05" in p.columns


# ------------------------------------------------------------------------------------------- provenance / integrity
def test_dataset_provenance_labels():
    fake = lambda s: SimpleNamespace(source=s, name=f"{s}/x")
    syn = provenance_info(fake("synthetic"))
    assert syn["status"] == "development/testing results only" and syn["reportable_in_manuscript"] is False
    semi = provenance_info(fake("semi_synthetic"))
    assert "semi-synthetic benchmark" in semi["status"] and "NOT a reproduction" in semi["status"]
    assert semi["reportable_in_manuscript"] is False
    real = provenance_info(fake("real"))
    assert "once the real dataset is reconstructed and the protocol is frozen" in real["status"] and not real["reportable_in_manuscript"]
    frozen = provenance_info(fake("real"), protocol_frozen=True)
    assert frozen["reportable_in_manuscript"] is True and "protocol frozen" in frozen["status"]
    assert "historical reference only" in syn["manuscript_values"]


def test_no_manuscript_scores_in_the_code():
    root = Path(__file__).resolve().parents[1] / "iocevaluator"
    reported = ["0.7551", "0.7696", "0.7321", "0.7463", "0.7484", "0.7627", "0.7402", "0.7538", "0.7168", "0.7314",
                "0.7418", "0.7562", "0.7476", "0.7621", "0.7326", "0.7471", "0.8424", "0.8176", "0.7812", "0.8468"]
    for py in root.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert not [x for x in reported if re.search(rf"(?<![\d.]){re.escape(x)}(?!\d)", text)], py.name
    for name in ("ablation.py", "evaluation.py"):
        src = (root / name).read_text(encoding="utf-8")
        assert not re.search(r"open\([^)]*manuscript|read_text\([^)]*manuscript", src)     # never reads manuscript.tex


# ------------------------------------------------------------------------------------------- complete dev-dataset run
@pytest.fixture(scope="module")
def full_run(ds):
    return run_ablation_dataset(ds, ("all",), CFG)


def test_complete_ablation_run_on_dev_dataset(ds, full_run):
    res = full_run
    runs = res.runs
    assert set(runs.variant) == set(VARIANTS) and len(runs) == 31 * 2 * 2          # 31 variants x 2 folds x 2 seeds
    for m in ("macro_f1", "micro_f1", "roc_auc", "pr_auc", "top5_hit_rate", "exact_path_hit5", "path_edge_f1",
              "n_path_cases", "n_train_used"):
        assert m in runs.columns, m
    assert runs.macro_f1.notna().all() and runs.macro_f1.between(0, 1).all()
    assert runs.top5_hit_rate.isna().all()                                         # no expert cases -> not valid, not invented
    assert (res.timings.train_time_total_s > 0).all() and (res.timings.inference_time_s > 0).all()
    assert len(res.timings) == len(runs) and len(res.paths) > 0 and "variant" in res.paths
    assert set(res.by_type.variant) == set(VARIANTS)
    # raw fold x seed rows are kept for every variant and carry the provenance label
    assert (runs.groupby("variant").size() == 4).all() and runs.data_provenance.nunique() == 1
    assert runs.data_provenance.iloc[0] == provenance_info(ds)["status"]
    # reduced-label variants really train on fewer labels; the others on all of them
    rl = {v: float((g.n_train_used / g.n_train).mean()) for v, g in runs.groupby("variant")}
    for p in (25, 50, 75):
        assert rl[f"megits_full_lf{p}"] == pytest.approx(p / 100, abs=0.01)
    assert rl[REFERENCE] == 1.0 and rl["chi1"] == 1.0 and rl["original_adjacency"] == 1.0
    # identical folds and seeds across variants (paired design)
    pairs = {v: set(zip(g.fold, g.seed)) for v, g in runs.groupby("variant")}
    assert len({frozenset(p) for p in pairs.values()}) == 1
    first = next(iter(res.per_variant.values()))
    for r in res.per_variant.values():
        assert [f.index for f in r.folds] == [f.index for f in first.folds]
        for a, b in zip(r.folds, first.folds):
            for attr in ("train", "val", "test"):
                np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))
        assert r.manifest["folds"] == first.manifest["folds"]                       # same campaigns in every variant
        assert r.manifest["protocol"]["seeds"] == [0, 1]
    # depth variants
    for L in (1, 3, 4):
        assert res.per_variant[f"megits_full_{L}layer"].manifest["model"]["layers"] == L
    assert res.per_variant[REFERENCE].manifest["model"]["layers"] == 2


def test_paired_outputs_and_manifest_of_the_complete_run(ds, full_run, tmp_path):
    res = full_run
    paired = res.paired()
    assert set(paired.variant) == set(VARIANTS) - {REFERENCE} and REFERENCE not in set(paired.variant)
    m = paired[paired.metric == "macro_f1"]
    assert len(m) == 30 and (m.n_pairs == 4).all() and (m.n_folds == 2).all()
    # difference to the reference is recomputable from the raw rows
    raw = res.runs.set_index(["variant", "fold", "seed"]).macro_f1
    d = (raw["chi3"] - raw[REFERENCE]).mean()
    assert m.set_index("variant").loc["chi3", "mean_diff"] == pytest.approx(d)
    out = write_ablation(tmp_path, res)
    for n in ("ablation_runs.csv", "ablation_per_type.csv", "ablation_timings.csv", "ablation_paired.csv",
              "ablation_summary.json", "manifest.json", "ablation_path_cases.csv"):
        assert out[n].exists()
    back = pd.read_csv(out["ablation_runs.csv"])
    assert len(back) == len(res.runs) and "variant" in back and "data_provenance" in back
    import json
    man = json.loads(out["manifest.json"].read_text())
    assert man["kind"] == "ablation_study" and man["reference_variant"] == REFERENCE
    assert man["provenance"]["status"] == provenance_info(ds)["status"] and man["provenance"]["reportable_in_manuscript"] is False
    assert man["dataset"]["profile"] == "dev" and man["protocol"]["seeds"] == [0, 1] and len(man["folds"]) == 2
    vs = {v["name"]: v for v in man["variants"]}
    assert set(vs) == set(VARIANTS)
    assert vs["megits_paths_only"]["active_structures"] == list(range(1, 12))
    assert vs["megits_graphs_only"]["active_structures"] == list(range(12, 21))
    assert vs["chi7"]["chi_weights"] == {"chi7": 1.0} and vs["binary_semantic"]["binary"] is True
    assert vs["original_adjacency"]["active_structures"] == [] and vs["megits_full_lf25"]["label_fraction"] == 0.25
    assert vs[REFERENCE]["chi_weights"] and all(abs(w - 1 / 20) < 1e-12 for w in vs[REFERENCE]["chi_weights"].values())
    assert "historical reference only" in man["provenance"]["manuscript_values"]
    assert man["content_sha256"] and {"git", "python"} <= set(man["environment"])
    summ = json.loads(out["ablation_summary.json"].read_text())
    assert set(summ["variants"]) == set(VARIANTS) and summ["provenance"]["status"] == provenance_info(ds)["status"]


def test_ablation_run_is_deterministic(ds):
    cfg = EvalConfig(seeds=(0,), max_folds=1, model=SupervisedGCNConfig(max_epochs=5, lr=1e-2), paths=CFG.paths)
    a = run_ablation_dataset(ds, ("megits_paths_only", "megits_full_lf50"), cfg)
    b = run_ablation_dataset(ds, ("megits_paths_only", "megits_full_lf50"), cfg)
    pd.testing.assert_frame_equal(a.runs, b.runs)
    assert a.manifest["content_sha256"] == b.manifest["content_sha256"]


# ------------------------------------------------------------------------------------------- label-preserving reduced-label sampling
def test_reduced_label_sampling_preserves_label_coverage_and_distribution(ds):
    from iocevaluator.supervised_gcn import label_coverage, label_distribution_tv, label_preserving_order
    g, ls = ds.tikg, ds.labels
    f = group_stratified_folds(g, ls, 10, 0)[0]
    train = np.flatnonzero(prepare_fold(g, ls, f).masks.train)
    order = label_preserving_order(ls, train, [0, f.index, 0, 7919])
    assert sorted(order) == sorted(train)                                           # a permutation of the training nodes
    np.testing.assert_array_equal(order, label_preserving_order(ls, train, [0, f.index, 0, 7919]))   # deterministic
    assert not np.array_equal(order, label_preserving_order(ls, train, [0, f.index, 1, 7919]))
    full = label_coverage(ls, train)["n_labels_with_positive"]
    rng = np.random.RandomState(0)
    for frac in (0.25, 0.5, 0.75):
        k = max(1, round(frac * len(train)))
        sub = order[:k]
        rand = [rng.permutation(train)[:k] for _ in range(30)]
        cov = label_coverage(ls, sub)["n_labels_with_positive"]
        assert cov >= np.mean([label_coverage(ls, r)["n_labels_with_positive"] for r in rand])      # >= random subsets
        assert label_distribution_tv(ls, sub, train) <= np.mean([label_distribution_tv(ls, r, train) for r in rand]) + 1e-9
        if k >= 2 * full:
            assert cov == full                                                      # coverage is complete when feasible
    # nested prefixes of one ordering
    assert set(order[: round(0.25 * len(train))]) < set(order[: round(0.5 * len(train))]) < set(order[: round(0.75 * len(train))])


def test_label_coverage_is_complete_on_a_small_feasible_toy():
    from iocevaluator.supervised_gcn import label_coverage
    g, ls, f, ctx = toy_ctx()
    train = np.flatnonzero(ctx.masks.train)
    full = label_coverage(ls, train)["n_labels_with_positive"]
    for fr in (0.5, 0.75):
        c = reduce_training_labels(ctx, ls, f, 0, fr)
        assert label_coverage(ls, np.flatnonzero(c.masks.train))["n_labels_with_positive"] == full


def test_ablation_paired_table_has_holm_columns(full_run):
    p = full_run.paired(["macro_f1"])
    assert {"wilcoxon_p", "wilcoxon_p_holm", "significant_holm_0.05"} <= set(p.columns)
    ok = p.wilcoxon_p.notna()
    assert (p.loc[ok, "wilcoxon_p_holm"] >= p.loc[ok, "wilcoxon_p"] - 1e-12).all() and (p.wilcoxon_p_holm.dropna() <= 1).all()
    assert (p.wilcoxon_p_holm.isna() == p.wilcoxon_p.isna()).all()
