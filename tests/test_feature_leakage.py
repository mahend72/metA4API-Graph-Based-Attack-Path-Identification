"""Feature-leakage audit as tests: the semi-synthetic feature policy, the allow-list behaviour of the feature builder,
injected forbidden fields, and train-only preprocessing (no validation / test statistics, vocabularies or terms)."""
import copy
from pathlib import Path

import numpy as np
import pytest

from iocevaluator.protocol import PreflightError, final_experiment_preflight, load_feature_policy
from iocevaluator.splits import group_stratified_folds
from iocevaluator.supervised_gcn import prepare_fold
from iocevaluator.tikg_features import FeatureBuilder, FeatureConfig

ROOT = Path(__file__).resolve().parents[1]
POLICY_FILE = ROOT / "configs" / "feature_policy_semi_synthetic.json"
READ_ALLOWED = {"first_seen", "last_seen", "update_count"}          # the only attrs keys the audited builder may read under the policy
FAMILY_PROXIES = {"source_countries", "target_countries", "industries", "family", "campaign", "campaign_id", "family_id", "text",
                  "synthetic", "provenance", "synthetic_fields", "cwe", "cvss_base_score", "cvss_vector", "cvss_version", "severity"}


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    from iocevaluator.datasets import load_dataset
    from iocevaluator.synthetic_tikg import generate, write_dataset
    root = tmp_path_factory.mktemp("semi")
    write_dataset(generate("dev", 42), root / "dev")
    return load_dataset("synthetic", profile="dev", root=root)


@pytest.fixture(scope="module")
def policy():
    return load_feature_policy(POLICY_FILE)


def cfg_for(policy, **kw):
    return FeatureConfig(exclude_fields=tuple(policy.exclude_fields), strip_terms=tuple(policy.strip_terms), **kw)


class LogDict(dict):
    """attrs dict that records every key the feature code asks for."""
    reads: set = set()

    def get(self, k, d=None):
        LogDict.reads.add(k)
        return super().get(k, d)

    def __getitem__(self, k):
        LogDict.reads.add(k)
        return super().__getitem__(k)


def matrix(tikg, cfg, train_idx):
    fb = FeatureBuilder(cfg)
    return fb.fit_transform(tikg, train_idx), fb


# ------------------------------------------------------------------------------------------------ the policy file
def test_semi_synthetic_policy_is_explicit_explained_and_scoped(policy):
    assert policy.specified() and policy.version
    assert FAMILY_PROXIES <= set(policy.exclude_fields)
    assert set(policy.exclude_fields) == set(policy.reasons) and all(len(r) > 20 for r in policy.reasons.values())
    assert not (READ_ALLOWED & set(policy.exclude_fields))                       # the retained fields are not excluded
    assert set(policy.applies_to) == {"semi_synthetic", "synthetic"} and "real" not in policy.applies_to
    assert policy.as_dict()["policy_sha256"]


def test_policy_without_a_reason_or_for_real_data_is_rejected(tmp_path, ds, policy):
    import json
    bad = tmp_path / "p.json"
    bad.write_text(json.dumps({"exclude_fields": ["a", "b"], "reasons": {"a": "x"}, "source": "s"}))
    with pytest.raises(PreflightError, match="without a reason"):
        load_feature_policy(bad)
    from iocevaluator.evaluation import EvalConfig
    real = type("D", (), {"source": "real"})()
    rep = final_experiment_preflight(EvalConfig(features=cfg_for(policy)), None, real, feature_policy=policy, allow_dirty=True,
                                     protocol_file=ROOT / "PROTOCOL_FREEZE.md")
    assert "feature_policy_in_scope" in {c.name for c in rep.failures()}      # the real-data policy is still OPEN
    semi = type("D", (), {"source": "semi_synthetic"})()
    rep = final_experiment_preflight(EvalConfig(features=cfg_for(policy)), None, semi, feature_policy=policy, allow_dirty=True,
                                     protocol_file=ROOT / "PROTOCOL_FREEZE.md")
    assert "feature_policy_in_scope" not in {c.name for c in rep.failures()} and "feature_policy_applied" not in {c.name for c in rep.failures()}


# ------------------------------------------------------------------------------------------------ what the builder reads
@pytest.mark.parametrize("backend", ["none", "tfidf"])
def test_under_the_policy_only_the_audited_attrs_are_ever_read(ds, policy, backend):
    g = copy.deepcopy(ds.tikg)
    for e in g.entities:
        e.attrs = LogDict(e.attrs)
    LogDict.reads = set()
    FeatureBuilder(cfg_for(policy, text_backend=backend, text_dim=4)).fit_transform(g, np.flatnonzero(ds.labels.labelled))
    # the builder asks for excluded keys too (and gets None); what matters is that no excluded value is ever returned
    returned = set()
    fb = FeatureBuilder(cfg_for(policy, text_backend=backend, text_dim=4))
    for e in g.entities:
        for k in e.attrs:
            if fb._attr(e, k) is not None:
                returned.add(k)
    assert returned <= READ_ALLOWED | (set(ds.tikg.entities[0].attrs) - set(policy.exclude_fields))
    assert {k for k in LogDict.reads if k not in policy.exclude_fields} <= READ_ALLOWED


def test_without_the_policy_the_builder_does_read_family_proxies(ds):
    g = copy.deepcopy(ds.tikg)
    for e in g.entities:
        e.attrs = LogDict(e.attrs)
    LogDict.reads = set()
    FeatureBuilder(FeatureConfig(text_backend="tfidf", text_dim=4)).fit_transform(g, np.flatnonzero(ds.labels.labelled))
    assert {"source_countries", "target_countries", "industries", "text"} <= LogDict.reads       # why the policy is not empty


# ------------------------------------------------------------------------------------------------ injected forbidden fields
def _inject(tikg, labels):
    """Plant label-perfect values in every forbidden field (read and unread) and in Entity.campaign."""
    g = copy.deepcopy(tikg)
    primary = labels.primary_label()
    for i, e in enumerate(g.entities):
        code = f"LABEL_{primary[i]}"
        e.attrs.update({"family": code, "campaign_id": code, "family_id": code, "label_code": code, "class_code": code,
                        "reference_path_member": True, "cwe": code, "cvss_base_score": 9.9, "severity": "CRITICAL",
                        "provenance": code, "synthetic_fields": [code], "source_countries": [code], "target_countries": [code],
                        "industries": [code], "text": f"{code} {code} {code}"})
    return g


@pytest.mark.parametrize("backend", ["none", "tfidf"])
def test_injected_forbidden_fields_cannot_change_the_feature_matrix(ds, policy, backend):
    lab = np.flatnonzero(ds.labels.labelled)
    base, _ = matrix(ds.tikg, cfg_for(policy, text_backend=backend, text_dim=4), lab)
    poisoned = _inject(ds.tikg, ds.labels)
    got, fb = matrix(poisoned, cfg_for(policy, text_backend=backend, text_dim=4), lab)
    assert got.shape == base.shape and np.array_equal(got, base)
    assert not any("LABEL_" in n for n in fb.names)
    assert fb.names and all(n.split(":")[0] in {"active_time", "recency", "update_frequency", "has_time", "type", "subtype"} or n.startswith("text")
                            for n in fb.names)


def test_the_injection_is_effective_without_the_policy(ds):
    """Sensitivity check: with no policy the same injection DOES change the matrix, so the test above is meaningful."""
    lab = np.flatnonzero(ds.labels.labelled)
    base, _ = matrix(ds.tikg, FeatureConfig(), lab)
    got, _ = matrix(_inject(ds.tikg, ds.labels), FeatureConfig(), lab)
    assert got.shape != base.shape or not np.array_equal(got, base)


def test_campaign_membership_on_the_entity_never_enters_the_features(ds, policy):
    lab = np.flatnonzero(ds.labels.labelled)
    base, _ = matrix(ds.tikg, cfg_for(policy), lab)
    g = copy.deepcopy(ds.tikg)
    for i, e in enumerate(g.entities):
        e.campaign = f"shuffled-{(i * 7) % 5}"
        e.name = e.name                                                      # names are used only by the text backend (off)
    got, _ = matrix(g, cfg_for(policy), lab)
    assert np.array_equal(got, base)


def test_node_names_and_text_cannot_leak_when_the_text_backend_is_off(ds, policy):
    assert FeatureConfig().text_backend == "none"
    lab = np.flatnonzero(ds.labels.labelled)
    base, _ = matrix(ds.tikg, cfg_for(policy), lab)
    g = copy.deepcopy(ds.tikg)
    for i, e in enumerate(g.entities):
        e.name = f"LABEL_{ds.labels.primary_label()[i]}"
    got, _ = matrix(g, cfg_for(policy), lab)
    assert np.array_equal(got, base)


# ------------------------------------------------------------------------------------------------ train-only preprocessing
def _mutate_non_train(tikg, non_train):
    g = copy.deepcopy(tikg)
    for i in non_train:
        a = g.entities[int(i)].attrs
        a.update({"first_seen": "1990-01-01T00:00:00", "last_seen": "2099-12-31T00:00:00", "update_count": 10 ** 6,
                  "source_countries": ["ZZ_NOVEL"], "target_countries": ["ZZ_NOVEL"], "industries": ["ZZ_NOVEL_SECTOR"],
                  "text": "zzqxinjected zzqxinjected zzqxinjected"})
    return g


@pytest.mark.parametrize("backend", ["none", "tfidf"])
def test_validation_and_test_statistics_cannot_enter_the_fit(ds, backend):
    lab = np.flatnonzero(ds.labels.labelled)
    train, rest = lab[: len(lab) // 2], np.setdiff1d(np.arange(ds.tikg.n), lab[: len(lab) // 2])
    cfg = FeatureConfig(text_backend=backend, text_dim=4)            # NO exclusion policy: the fit itself must be train-only
    X0, fb0 = matrix(ds.tikg, cfg, train)
    X1, fb1 = matrix(_mutate_non_train(ds.tikg, rest), cfg, train)
    assert fb0._locs == fb1._locs and fb0._orgs == fb1._orgs and "ZZ_NOVEL" not in fb1._locs and "ZZ_NOVEL_SECTOR" not in fb1._orgs
    assert fb0._ref == fb1._ref
    assert np.array_equal(fb0._mu, fb1._mu) and np.array_equal(fb0._sd, fb1._sd)
    assert np.array_equal(X0[train], X1[train])
    if backend == "tfidf":
        assert "zzqxinjected" not in fb1._tfidf.vocabulary_
    # ...and fitting on the mutated rows WOULD have changed the statistics (the test is sensitive)
    _, fb2 = matrix(_mutate_non_train(ds.tikg, rest), cfg, np.arange(ds.tikg.n))
    assert not np.array_equal(fb0._mu, fb2._mu)


def test_prepare_fold_features_are_independent_of_validation_and_test_attributes(ds, policy):
    f = group_stratified_folds(ds.tikg, ds.labels, 5, 0, 0.1)[0]
    cfg = cfg_for(policy)
    ctx0 = prepare_fold(ds.tikg, ds.labels, f, cfg)
    non_train = np.union1d(np.union1d(f.val, f.test), np.flatnonzero(f.test_nodes_mask))
    non_train = np.setdiff1d(non_train, np.flatnonzero(ctx0.masks.train))
    g = _mutate_non_train(ds.tikg, non_train)
    ctx1 = prepare_fold(g, ds.labels, f, cfg)
    tr = np.flatnonzero(ctx0.masks.train)
    assert len(tr) and np.array_equal(ctx0.x[tr], ctx1.x[tr])                  # the scaler / vocabularies saw train nodes only
    # val and test rows are only TRANSFORMED with train statistics (their own values differ, the statistics do not)
    assert ctx0.feature_names == ctx1.feature_names
    cfg_open = FeatureConfig()                                                 # even without the policy the train rows are unaffected
    a, b = prepare_fold(ds.tikg, ds.labels, f, cfg_open), prepare_fold(g, ds.labels, f, cfg_open)
    assert np.array_equal(a.x[tr], b.x[tr])


def test_validation_labels_never_reach_training_labels(ds):
    f = group_stratified_folds(ds.tikg, ds.labels, 5, 0, 0.1)[0]
    ctx = prepare_fold(ds.tikg, ds.labels, f, FeatureConfig())
    assert not (ctx.masks.train & ctx.masks.val).any() and not (ctx.masks.visible & np.asarray(f.test_nodes_mask)).any()
    assert not ctx.Y_visible[np.asarray(f.test_nodes_mask)].any()              # test labels are zeroed in the visible label matrix
