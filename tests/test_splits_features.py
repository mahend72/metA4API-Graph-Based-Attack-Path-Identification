import numpy as np
import pytest

from iocevaluator.labels import build_label_space
from iocevaluator.splits import group_stratified_folds
from iocevaluator.tikg import EntityType as ET
from iocevaluator.tikg_features import FeatureBuilder, FeatureConfig
from .synthetic import make_synthetic


def setup():
    g, lab = make_synthetic(12)
    return g, build_label_space(g, lab)


def test_campaign_isolation_and_partition():
    g, ls = setup()
    folds = group_stratified_folds(g, ls, n_splits=4, seed=1)
    camp = g.campaigns()
    all_test = np.concatenate([f.test for f in folds])
    assert sorted(all_test) == sorted(np.flatnonzero(ls.labelled))          # each labelled node tested exactly once
    for f in folds:
        tc = set(camp[f.test])
        assert not tc & set(camp[f.train]) and not tc & set(camp[f.val])      # campaign-level isolation
        assert not set(f.train) & set(f.val) and len(f.val) > 0                # val carved from outer-train only
        assert set(camp[np.flatnonzero(f.test_nodes_mask)]) == tc               # whole campaigns hidden from train graph


def test_too_few_groups_and_missing_campaign():
    g, ls = setup()
    with pytest.raises(ValueError):
        group_stratified_folds(g, ls, n_splits=20)
    g.entities[0].campaign = ""
    with pytest.raises(ValueError):
        group_stratified_folds(g, ls, n_splits=4)


def test_features_exclude_fields_and_train_only_stats():
    g, _ = setup()
    tr = np.arange(0, g.n // 2)
    fb = FeatureBuilder(FeatureConfig(exclude_fields=("industries",)))
    X = fb.fit_transform(g, tr)
    assert X.shape[0] == g.n and np.isfinite(X).all()
    assert not any(n.startswith("org:") for n in fb.names)                    # excluded label-source field is absent
    # statistics are train-only: standardised numeric columns have mean ~0 on train, not necessarily on all nodes
    assert np.abs(X[tr, :3].mean(0)).max() < 1e-4
    X2 = FeatureBuilder(FeatureConfig()).fit_transform(g, tr)
    assert any(True for _ in X2) and X2.shape[1] > X.shape[1]
    # changing a TEST node's attributes must not change the fitted statistics
    te = g.n - 1
    base = FeatureBuilder(FeatureConfig()).fit(g, tr)
    g.entities[te].attrs["update_count"] = 10_000
    chg = FeatureBuilder(FeatureConfig()).fit(g, tr)
    np.testing.assert_allclose(base._mu, chg._mu)


def test_text_strip_terms():
    g, _ = setup()
    g.entities[0].attrs["text"] = "contains SECRETLABEL word"
    fb = FeatureBuilder(FeatureConfig(text_backend="tfidf", strip_terms=("SECRETLABEL",)))
    fb.fit(g, np.arange(g.n))
    assert "secretlabel" not in fb._tfidf.vocabulary_
