"""Frozen-protocol preflight wired into every route that can launch a final / manuscript-scale experiment.

Covers: a valid launch, dirty-repo rejection, missing feature policy, sensitivity configs rejected as primary, wrong chi
weights, wrong fold / seed count, wrong ranking / path configuration, post-run path-integrity failure, the --dry-run /
--allow-dirty behaviour, and the library-level backstops (evaluate_cv, legacy harness, scalability, path budget, rank).
No final experiment is run: every training launch is on the synthetic ``dev`` profile with a 1-epoch model."""
import importlib.util
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iocevaluator import launch as L
from iocevaluator import protocol as P
from iocevaluator.evaluation import EvalConfig, evaluate_cv
from iocevaluator.path_tracing import PathConfig
from iocevaluator.protocol import (ExperimentIntegrityError, FeaturePolicy, PreflightError, PreflightRequiredError, RUN_DIAGNOSTIC,
                                   RUN_DRY, RUN_FINAL, DevelopmentOnlyError, exhaustive_path_config, final_experiment_preflight,
                                   primary_ranker_config, sensitivity_ranker_configs)
from iocevaluator.ranking import RankerConfig, ThreatPrioritisationRanker
from iocevaluator.supervised_gcn import SupervisedGCNConfig

ROOT = Path(__file__).resolve().parents[1]
PF = ROOT / "PROTOCOL_FREEZE.md"
POLICY = FeaturePolicy((), (), "none required: synthetic development data, labels are generated independently of the features")
FAST = SupervisedGCNConfig(max_epochs=1, lr=1e-2)


# ------------------------------------------------------------------------------------------------ fixtures / helpers
def _git(repo, *a):
    subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, text=True)


@pytest.fixture()
def clean_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("a")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


@pytest.fixture()
def dirty_repo(clean_repo):
    (clean_repo / "uncommitted.txt").write_text("x")
    return clean_repo


@pytest.fixture(scope="module")
def dev_ds(tmp_path_factory):
    from iocevaluator.datasets import load_dataset
    from iocevaluator.synthetic_tikg import generate, write_dataset
    root = tmp_path_factory.mktemp("synthetic")
    write_dataset(generate("dev", 42), root / "dev")
    return load_dataset("synthetic", profile="dev", root=root)


def frozen_cfg(policy=POLICY, **kw):
    return L.eval_config_for(policy, EvalConfig(model=FAST, **kw))


def pre(cfg=None, *, ranker=None, ds=None, policy=POLICY, repo=None, run_class=RUN_DRY, **kw):
    """Low-level preflight, dirty tree tolerated unless the test says otherwise."""
    cfg = cfg if cfg is not None else frozen_cfg(policy)
    return final_experiment_preflight(cfg, ranker, ds, feature_policy=policy, protocol_file=PF, allow_dirty=kw.pop("allow_dirty", True),
                                      repo=repo, run_class=run_class, **kw)


def failed(rep):
    return {c.name for c in rep.failures()}


class Boom(AssertionError):
    pass


@pytest.fixture()
def no_training(monkeypatch):
    """Any attempt to train fails the test: used to prove a preflight failure aborts BEFORE training."""
    def boom(*a, **k):
        raise Boom("training started although the preflight failed / dry run")
    monkeypatch.setattr(L, "_run_task", boom)


# ------------------------------------------------------------------------------------------------ 1. valid launch
def test_valid_final_launch_runs_and_records_the_preflight(dev_ds, clean_repo, tmp_path, capsys):
    out = tmp_path / "out"
    res = L.launch("evaluate", dev_ds, run_class=RUN_FINAL, feature_policy=POLICY, cfg=frozen_cfg(), out=out, repo=clean_repo,
                   protocol_file=PF, use_reference_paths=False)
    summary = capsys.readouterr().out
    # concise summary BEFORE training: protocol hash, commit, dataset fingerprint, policy, folds/seeds, model/config hash, ranking/paths
    for token in ("PREFLIGHT", "protocol hash", "code / git", "dataset", "fingerprint", "feature policy", "folds / seeds",
                  "model / config", "ranking", "paths", "chi weights", "PASS"):
        assert token in summary, token
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=clean_repo, text=True).strip()
    assert commit[:12] in summary
    man = json.loads((out / "manifest.json").read_text())
    pre_ = man["preflight"]
    assert pre_["performed"] and pre_["ok"] and pre_["run_class"] == RUN_FINAL
    assert pre_["git"]["commit"] == commit and pre_["git"]["dirty"] is False
    assert pre_["protocol_sha256"] and pre_["config_sha256"] and pre_["model_sha256"] and pre_["code_sha256"]
    assert pre_["dataset"]["fingerprint_sha256"] == man["dataset"]["fingerprint_sha256"]
    assert pre_["feature_policy"]["source"].startswith("none required")
    assert pre_["folds_seeds"]["n_splits"] == 10 and pre_["folds_seeds"]["seeds"] == [0, 1, 2, 3, 4]
    assert pre_["ranking"]["alpha"] == 0.5 and pre_["ranking"]["tau"] is None
    assert pre_["paths"]["max_edges"] == 6 and pre_["path_integrity_enforced"] is True
    assert pre_["chi_weights"] == "uniform 1/20" and all(pre_["preflight_checks"].values())
    assert pre_["reportable_as_manuscript"] is False                       # synthetic data is never a manuscript result
    assert man["integrity"]["passed"] is True
    assert man["protocol"]["folds_run"] == 10
    # the preflight record is outside content_sha256 (the hash stays a function of data + configuration only)
    import hashlib
    body = {k: v for k, v in man.items() if k not in ("environment", "content_sha256", "preflight", "integrity", "reportability")}
    assert man["content_sha256"] == hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
    assert (out / "preflight.json").exists() and res["integrity"]["passed"]


def test_a_smoke_shaped_config_is_not_a_final_run_and_unguarded_manifests_say_so(dev_ds, clean_repo):
    from iocevaluator.model_comparison import run_model_comparison_dataset
    cfg = frozen_cfg(n_splits=3, seeds=(0,), max_folds=1)
    with pytest.raises(PreflightError, match="folds_and_seeds_fixed"):         # smoke shape is not the frozen protocol
        L.do_preflight(cfg, None, dev_ds, run_class=RUN_FINAL, feature_policy=POLICY, protocol_file=PF, repo=clean_repo, echo=lambda s: None)
    res = run_model_comparison_dataset(dev_ds, ("gcn",), cfg)                    # development smoke run: allowed, and labelled
    assert res.manifest["preflight"]["performed"] is False and res.manifest["preflight"]["run_class"] == "unguarded_development"


# ------------------------------------------------------------------------------------------------ 2. dirty repo
def test_final_run_rejects_a_dirty_repo_before_training(dev_ds, dirty_repo, no_training, tmp_path):
    with pytest.raises(PreflightError, match="clean_working_tree"):
        L.launch("evaluate", dev_ds, run_class=RUN_FINAL, feature_policy=POLICY, cfg=frozen_cfg(), out=tmp_path / "o", repo=dirty_repo,
                 protocol_file=PF, echo=lambda s: None)
    assert not (tmp_path / "o" / "manifest.json").exists()


def test_final_cannot_be_relaxed_through_allow_dirty_on_the_launch_layer(dev_ds, dirty_repo):
    rep = None
    with pytest.raises(PreflightError):
        rep = L.do_preflight(frozen_cfg(), None, dev_ds, run_class=RUN_FINAL, feature_policy=POLICY, protocol_file=PF, repo=dirty_repo,
                             echo=lambda s: None)
    assert rep is None
    assert L.run_class_from_flags(False, False) == RUN_FINAL               # no flags -> the strict class


# ------------------------------------------------------------------------------------------------ 3. feature policy
def test_missing_feature_policy_fails_in_every_mode(dev_ds, clean_repo, no_training):
    for rc in (RUN_FINAL, RUN_DRY, RUN_DIAGNOSTIC):
        with pytest.raises(PreflightError, match="feature_exclusion_policy_supplied"):
            L.launch("evaluate", dev_ds, run_class=rc, feature_policy=None, cfg=L.eval_config_for(None, EvalConfig(model=FAST)),
                     repo=clean_repo, protocol_file=PF, echo=lambda s: None)
    for bad in (FeaturePolicy((), (), ""), FeaturePolicy((), (), "TODO"), FeaturePolicy((), (), "   ")):
        assert "feature_exclusion_policy_supplied" in failed(pre(policy=bad, cfg=EvalConfig(model=FAST)))
    # the policy must actually be applied to the features
    assert "feature_policy_applied" in failed(pre(cfg=EvalConfig(model=FAST), policy=FeaturePolicy(("cwe",), (), "unit test")))


def test_feature_policy_file_round_trip_and_unknown_keys(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"exclude_fields": ["cwe"], "strip_terms": ["x"], "source": "analyst list v1"}))
    p = P.load_feature_policy(f)
    assert p.specified() and list(p.exclude_fields) == ["cwe"]
    f.write_text(json.dumps({"exclude_fields": [], "source": "none required: reason"}))
    assert P.load_feature_policy(f).specified()
    f.write_text(json.dumps({"exclude": []}))
    with pytest.raises(PreflightError):
        P.load_feature_policy(f)


# ------------------------------------------------------------------------------------------------ 4. sensitivity != primary
def test_sensitivity_configs_are_never_accepted_as_primary(dev_ds, clean_repo, no_training):
    for name, c in sensitivity_ranker_configs().items():
        rep = pre(ranker=c)
        assert not rep.ok and "not_a_sensitivity_configuration" in failed(rep) and "ranking_matches_frozen" in failed(rep), name
        with pytest.raises(PreflightError, match="sensitivity"):
            L.launch("evaluate", dev_ds, run_class=RUN_DRY, feature_policy=POLICY, cfg=frozen_cfg(), ranker_cfg=c, repo=clean_repo,
                     protocol_file=PF, echo=lambda s: None)
    # path-budget sensitivity levels are not primary either
    from iocevaluator.path_budget import LEVELS, level_config
    for level, _ in LEVELS:
        cfg = replace(frozen_cfg(), paths=level_config(level, 6))
        assert "not_a_sensitivity_configuration" in failed(pre(cfg)), level
    # an explicit non-primary role is rejected as well
    assert "role_is_primary" in failed(pre(role="sensitivity"))
    # the primary configuration itself is accepted
    assert primary_ranker_config() not in sensitivity_ranker_configs().values()
    assert "not_a_sensitivity_configuration" not in failed(pre())


def test_session_guard_rejects_a_sensitivity_ranker(dev_ds, clean_repo):
    rep = L.do_preflight(frozen_cfg(), None, dev_ds, run_class=RUN_DRY, feature_policy=POLICY, protocol_file=PF, repo=clean_repo, echo=lambda s: None)
    cfg = frozen_cfg()
    with P.preflight_session(rep, cfg):
        for c in sensitivity_ranker_configs().values():
            with pytest.raises(PreflightRequiredError, match="sensitivity"):
                evaluate_cv(dev_ds.tikg, dev_ds.labels, cfg, ranker=ThreatPrioritisationRanker(c))


# ------------------------------------------------------------------------------------------------ 5. chi weights
def test_wrong_chi_weights_fail_the_preflight_in_every_mode(monkeypatch, dev_ds, clean_repo, no_training):
    import iocevaluator.megits as megits
    real = megits.normalise_weights

    def skewed(structures, *a, **k):
        w = dict(real(structures, *a, **k))
        first = next(iter(w))
        w[first] *= 2
        return w
    monkeypatch.setattr(megits, "normalise_weights", skewed)
    assert "chi_weights_fixed" in failed(pre())
    for rc in (RUN_FINAL, RUN_DRY, RUN_DIAGNOSTIC):
        with pytest.raises(PreflightError, match="chi_weights_fixed"):
            L.launch("evaluate", dev_ds, run_class=rc, feature_policy=POLICY, cfg=frozen_cfg(), repo=clean_repo, protocol_file=PF,
                     echo=lambda s: None)
    monkeypatch.setattr(megits, "normalise_weights", lambda structures, *a, **k: {s.id: 1 / 19 for s in list(structures)[:19]})
    assert "chi_weights_fixed" in failed(pre())


# ------------------------------------------------------------------------------------------------ 6. folds / seeds
def test_wrong_fold_and_seed_counts_fail(dev_ds, clean_repo, no_training):
    base = frozen_cfg()
    bad = {"seeds": replace(base, seeds=(0, 1, 2, 3)), "seed values": replace(base, seeds=(1, 2, 3, 4, 5)),
           "n_splits": replace(base, n_splits=5), "fold_seed": replace(base, fold_seed=1), "val_fraction": replace(base, val_fraction=0.2),
           "max_folds": replace(base, max_folds=2), "ungrouped": replace(base, allow_ungrouped=True)}
    for what, cfg in bad.items():
        assert "folds_and_seeds_fixed" in failed(pre(cfg)), what
        with pytest.raises(PreflightError, match="folds_and_seeds_fixed"):
            L.launch("evaluate", dev_ds, run_class=RUN_DRY, feature_policy=POLICY, cfg=cfg, repo=clean_repo, protocol_file=PF,
                     echo=lambda s: None)
    # the fold count actually used is checked too
    assert "n_splits_used_equals_10" in failed(pre(base, n_folds_run=9))
    assert "n_splits_used_equals_10" not in failed(pre(base, n_folds_run=10))


# ------------------------------------------------------------------------------------------------ 7. ranking / path config
def test_wrong_ranking_or_path_configuration_fails(dev_ds, clean_repo, no_training):
    base = frozen_cfg()
    for r in (RankerConfig(alpha=0.7), RankerConfig(tau=0.25), RankerConfig(ec_scope="global"), RankerConfig(ec_scope="component"),
              RankerConfig(ec_norm="minmax"), RankerConfig(missing_severity="zero"), RankerConfig(missing_severity="exclude")):
        assert "ranking_matches_frozen" in failed(pre(ranker=r)), r
    assert {"no_unresolved_configuration", "ranking_matches_frozen"} <= failed(pre(ranker=RankerConfig(alpha=None)))
    for what, cfg in {"legacy PathConfig": replace(base, paths=PathConfig()),
                      "max_edges": replace(base, paths=replace(base.paths, max_edges=5)),
                      "conf_threshold": replace(base, paths=replace(base.paths, conf_threshold=0.6)),
                      "candidate cap": replace(base, paths=replace(base.paths, max_candidates=100)),
                      "expansion ceiling": replace(base, paths=replace(base.paths, max_expansions_per_seed=1000)),
                      "per-seed budget": replace(base, paths=replace(base.paths, max_paths_per_seed=50))}.items():
        assert "paths_match_frozen" in failed(pre(cfg)), what
        with pytest.raises(PreflightError, match="paths_match_frozen"):
            L.launch("evaluate", dev_ds, run_class=RUN_DRY, feature_policy=POLICY, cfg=cfg, repo=clean_repo, protocol_file=PF,
                     echo=lambda s: None)
    off = replace(base, enforce_path_integrity=False)
    assert "path_integrity_enforced" in failed(pre(off))
    with pytest.raises(PreflightError, match="path_integrity_enforced"):
        L.launch("evaluate", dev_ds, run_class=RUN_DIAGNOSTIC, feature_policy=POLICY, cfg=off, repo=clean_repo, protocol_file=PF,
                 echo=lambda s: None)


def test_dataset_provenance_and_protocol_file_are_checked_in_every_mode(dev_ds, clean_repo, tmp_path, no_training):
    assert "dataset_provenance_known" in failed(pre(ds=None))
    assert "dataset_provenance_known" in failed(pre(ds=type("X", (), {"source": "mystery"})()))
    txt = PF.read_text(encoding="utf-8")
    unfrozen = tmp_path / "p.md"
    unfrozen.write_text(txt.replace("Protocol-status: FROZEN", "Protocol-status: PROPOSED"), encoding="utf-8")
    with pytest.raises(PreflightError, match="protocol_marked_frozen"):
        L.launch("evaluate", dev_ds, run_class=RUN_DRY, feature_policy=POLICY, cfg=frozen_cfg(), repo=clean_repo, protocol_file=unfrozen,
                 echo=lambda s: None)


# ------------------------------------------------------------------------------------------------ 8. post-run integrity
class _FakeResult:
    def __init__(self, paths, folds_run=10, enforce=True):
        self.paths = paths
        self.manifest = {"protocol": {"folds_run": folds_run}, "paths": {"enabled": True, "enforce_path_integrity": enforce}}


def _rows(hit):
    return pd.DataFrame({"case_id": ["a", "b"], "fold": [0, 1], "seed": [0, 0], "search_budget_hit": [False, bool(hit)]})


def test_post_run_check_fails_on_any_budget_hit_and_on_unenforced_runs():
    assert P.post_run_integrity(_FakeResult(_rows(False)), expect_folds=10)["passed"]
    with pytest.raises(ExperimentIntegrityError, match="search budget"):
        P.post_run_integrity(_FakeResult(_rows(True)))
    assert not P.post_run_integrity(_FakeResult(_rows(True)), raise_on_fail=False)["passed"]
    assert not P.post_run_integrity(_FakeResult(_rows(False), enforce=False), raise_on_fail=False)["passed"]
    assert not P.post_run_integrity(_FakeResult(_rows(False), folds_run=9), expect_folds=10, raise_on_fail=False)["passed"]
    with pytest.raises(ExperimentIntegrityError, match="search_budget_hit"):
        P.post_run_integrity(_FakeResult(_rows(False).drop(columns="search_budget_hit")))


def test_launch_writes_no_results_when_the_post_run_check_fails(dev_ds, clean_repo, tmp_path, monkeypatch):
    wrote = []
    monkeypatch.setattr(L, "_run_task", lambda *a, **k: (_FakeResult(_rows(True)), lambda out, res: wrote.append(out) or {}))
    out = tmp_path / "bad"
    with pytest.raises(ExperimentIntegrityError):
        L.launch("evaluate", dev_ds, run_class=RUN_FINAL, feature_policy=POLICY, cfg=frozen_cfg(), out=out, repo=clean_repo,
                 protocol_file=PF, echo=lambda s: None)
    assert wrote == []                                                       # result tables were never written
    failure = json.loads((out / "integrity_failure.json").read_text())
    assert failure["integrity"]["passed"] is False and failure["preflight"]["ok"] is True


def test_real_harness_budget_hit_is_caught_by_the_post_run_check(dev_ds):
    """With enforcement switched off the per-case raise cannot fire, but the post-run re-check still fails the experiment."""
    from iocevaluator.evaluation import evaluate_dataset
    tight = replace(exhaustive_path_config(), max_paths_per_seed=1, max_expansions_per_seed=5)
    smoke = EvalConfig(model=FAST, seeds=(0,), max_folds=1, paths=tight, enforce_path_integrity=False)
    res = evaluate_dataset(dev_ds, smoke, use_reference_paths=True)
    assert res.paths["search_budget_hit"].any()
    with pytest.raises(ExperimentIntegrityError):
        P.post_run_integrity(res)
    integrity = P.post_run_integrity(res, raise_on_fail=False)
    P.annotate_integrity(res, integrity)
    assert res.manifest["integrity"]["passed"] is False
    # and the same configuration can never be preflighted (budgets differ from the frozen search; enforcement is off)
    assert {"paths_match_frozen", "path_integrity_enforced"} <= failed(pre(replace(smoke, features=frozen_cfg().features)))


# ------------------------------------------------------------------------------------------------ 9. dry-run / allow-dirty
def test_dry_run_bypasses_only_the_clean_tree_and_never_trains(dev_ds, dirty_repo, no_training, tmp_path, capsys):
    res = L.launch("evaluate", dev_ds, run_class=RUN_DRY, feature_policy=POLICY, cfg=frozen_cfg(), out=tmp_path / "d", repo=dirty_repo,
                   protocol_file=PF)
    assert res["results"] is None and res["report"].ok
    rec = res["report"].record
    assert rec["git"]["dirty"] is True and rec["run_class"] == RUN_DRY and rec["reportable_as_manuscript"] is False
    assert "dry run: preflight passed" in capsys.readouterr().out
    saved = json.loads((tmp_path / "d" / "preflight.json").read_text())
    assert saved["run_class"] == RUN_DRY and saved["performed"] is True
    # ...but every other check still bites in a dry run
    for kw in ({"feature_policy": None}, {"cfg": replace(frozen_cfg(), seeds=(0,))}, {"cfg": replace(frozen_cfg(), paths=PathConfig())}):
        args = {"feature_policy": POLICY, "cfg": frozen_cfg(), **kw}
        with pytest.raises(PreflightError):
            L.launch("evaluate", dev_ds, run_class=RUN_DRY, repo=dirty_repo, protocol_file=PF, echo=lambda s: None, **args)


def test_allow_dirty_trains_but_is_marked_diagnostic_and_not_reportable(dev_ds, dirty_repo, tmp_path):
    out = tmp_path / "diag"
    res = L.launch("evaluate", dev_ds, run_class=RUN_DIAGNOSTIC, feature_policy=POLICY, cfg=frozen_cfg(), out=out, repo=dirty_repo,
                   protocol_file=PF, echo=lambda s: None, use_reference_paths=False)
    assert res["results"] is not None and len(res["results"].runs) == 50          # it really trained the full 10 x 5 shape
    man = json.loads((out / "manifest.json").read_text())
    assert man["preflight"]["run_class"] == RUN_DIAGNOSTIC and man["preflight"]["git"]["dirty"] is True
    assert man["preflight"]["reportable_as_manuscript"] is False and man["preflight"]["allow_dirty"] is True
    assert man["integrity"]["passed"] and man["integrity"]["level"] == "final"
    assert man["reportability"]["reportable"] is False and any("not clean" in r or "'diagnostic'" in r for r in man["reportability"]["reasons"])
    assert res["report"].ok


def test_cli_flags_map_to_run_classes_and_exit_codes(dev_ds, dirty_repo, clean_repo, monkeypatch, tmp_path):
    assert L.run_class_from_flags(False, False) == RUN_FINAL
    assert L.run_class_from_flags(False, True) == RUN_DIAGNOSTIC
    assert L.run_class_from_flags(True, False) == RUN_DRY and L.run_class_from_flags(True, True) == RUN_DRY
    pol = tmp_path / "policy.json"
    pol.write_text(json.dumps({"exclude_fields": [], "strip_terms": [], "source": "none required: synthetic development data"}))
    monkeypatch.setattr("iocevaluator.datasets.load_dataset", lambda *a, **k: dev_ds)
    monkeypatch.setattr(L, "_run_task", lambda *a, **k: (_ for _ in ()).throw(Boom("trained")))
    base = ["--task", "evaluate", "--source", "synthetic", "--feature-policy", str(pol), "--protocol-file", str(PF)]
    msgs = []
    echo = msgs.append
    assert L.main(base + ["--repo-root", str(dirty_repo)], echo=echo) == L.EXIT_PREFLIGHT        # final + dirty -> rejected
    assert any("clean_working_tree" in m for m in msgs)
    assert L.main(base + ["--repo-root", str(dirty_repo), "--dry-run"], echo=echo) == L.EXIT_OK       # dry run passes on a dirty tree
    assert L.main(["--task", "evaluate", "--source", "synthetic", "--dry-run", "--repo-root", str(clean_repo)], echo=echo) == L.EXIT_PREFLIGHT   # no policy


# ------------------------------------------------------------------------------------------------ library backstops
def test_evaluate_cv_refuses_a_final_scale_run_without_a_preflight(dev_ds):
    with pytest.raises(PreflightRequiredError, match="no preflight session"):
        evaluate_cv(dev_ds.tikg, dev_ds.labels, EvalConfig(model=FAST))                    # 10 folds x 5 seeds, no max_folds
    with pytest.raises(PreflightRequiredError, match="real dataset"):
        evaluate_cv(dev_ds.tikg, dev_ds.labels, EvalConfig(model=FAST, seeds=(0,), max_folds=1), dataset_info={"source": "real"})
    from iocevaluator.cti_baselines import select_baselines, evaluate_baseline_cv
    with pytest.raises(PreflightRequiredError):
        evaluate_baseline_cv(dev_ds.tikg, dev_ds.labels, select_baselines(("attackg",))[0], EvalConfig(model=FAST))


def test_smoke_runs_still_work_without_a_preflight_and_say_so(dev_ds):
    res = evaluate_cv(dev_ds.tikg, dev_ds.labels, EvalConfig(model=FAST, seeds=(0,), max_folds=1))
    assert res.manifest["preflight"]["performed"] is False


def test_session_requires_the_preflighted_configuration(dev_ds, clean_repo):
    cfg = frozen_cfg()
    rep = L.do_preflight(cfg, None, dev_ds, run_class=RUN_DRY, feature_policy=POLICY, protocol_file=PF, repo=clean_repo, echo=lambda s: None)
    with P.preflight_session(rep, cfg):
        with pytest.raises(PreflightRequiredError, match="differs from the preflighted"):
            evaluate_cv(dev_ds.tikg, dev_ds.labels, replace(cfg, seeds=(0,), max_folds=1))
    bad = pre(replace(cfg, seeds=(0,)))
    with pytest.raises(PreflightError):
        with P.preflight_session(bad, cfg):
            raise AssertionError("a failed preflight must not open a session")


def test_legacy_harness_and_path_budget_refuse_manuscript_scale(dev_ds, monkeypatch):
    from iocevaluator.experiments import ExperimentConfig, run_cv
    with pytest.raises(DevelopmentOnlyError):
        run_cv(dev_ds.tikg, dev_ds.labels, ["metA4API"], ExperimentConfig())                 # 10 x 5 shape
    monkeypatch.setattr(P, "MANUSCRIPT_SCALE_NODES", 10)
    with pytest.raises(DevelopmentOnlyError):
        run_cv(dev_ds.tikg, dev_ds.labels, ["metA4API"], ExperimentConfig(seeds=(0,), max_folds=1))
    from iocevaluator.path_budget import build_fold_inputs, run_budget_sensitivity
    with pytest.raises(DevelopmentOnlyError):
        build_fold_inputs(dev_ds)
    with pytest.raises(DevelopmentOnlyError):
        run_budget_sensitivity(dev_ds, [])


def test_scalability_and_rank_refuse_manuscript_scale_without_preflight(dev_ds, clean_repo, tmp_path, monkeypatch):
    from iocevaluator.scalability import ScalabilityConfig, run_scalability
    with pytest.raises(PreflightRequiredError, match="manuscript-scale graph"):
        run_scalability(ScalabilityConfig(sizes=(3728,)))
    monkeypatch.setattr(P, "MANUSCRIPT_SCALE_NODES", 100)
    with pytest.raises(PreflightRequiredError):
        run_scalability(ScalabilityConfig(sizes=(250,)))
    # inside a session the truncating legacy path search is refused
    cfg = frozen_cfg()
    rep = L.do_preflight(cfg, None, dev_ds, run_class=RUN_DRY, feature_policy=POLICY, protocol_file=PF, repo=clean_repo, echo=lambda s: None)
    with P.preflight_session(rep, cfg), pytest.raises(PreflightRequiredError, match="frozen search"):
        run_scalability(ScalabilityConfig(sizes=(250,), paths=PathConfig()))
    # rank with reference paths = a path evaluation
    from iocevaluator.workflows import rank
    from .synthetic import make_synthetic
    g, lab = make_synthetic(9)
    g.save(tmp_path / "t.json")
    (tmp_path / "l.json").write_text(json.dumps({"labels": lab}))
    (tmp_path / "r.json").write_text(json.dumps([{"case_id": "c", "campaign": "c0", "nodes": ["ta0_0", "dev0_0", "cve0_0"], "relations": []}]))
    monkeypatch.setattr(P, "MANUSCRIPT_SCALE_NODES", 5)
    with pytest.raises(DevelopmentOnlyError):
        rank(tmp_path / "t.json", tmp_path / "l.json", tmp_path / "rk", reference_paths=tmp_path / "r.json")


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scalability_script_runs_the_preflight_at_manuscript_scale(clean_repo, tmp_path, capsys):
    script = _load_script("run_scalability")
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"exclude_fields": [], "source": "none required: synthetic cost-measurement data"}))
    # manuscript size, no policy -> preflight fails, nothing measured
    assert script.main(["--sizes", "3728", "--repo-root", str(clean_repo), "--protocol-file", str(PF), "--out", str(tmp_path / "o")]) == 2
    # manuscript size with the truncating legacy search -> refused even with a policy
    assert script.main(["--sizes", "3728", "--path-search", "legacy", "--feature-policy", str(pol), "--repo-root", str(clean_repo),
                        "--protocol-file", str(PF), "--out", str(tmp_path / "o")]) == 2
    assert "paths_match_frozen" in capsys.readouterr().out
    # dry run: full preflight, nothing measured
    assert script.main(["--sizes", "3728", "--dry-run", "--feature-policy", str(pol), "--repo-root", str(clean_repo),
                        "--protocol-file", str(PF)]) == 0
    assert not (tmp_path / "o").exists()


def test_experiment_script_and_cli_route_exist(clean_repo, tmp_path):
    script = _load_script("run_experiment")
    assert script.main is L.main
    from iocevaluator.cli import main as cli_main
    with pytest.raises(SystemExit) as e:
        cli_main(["experiment", "--task", "evaluate", "--source", "synthetic", "--dry-run", "--repo-root", str(clean_repo)])   # no policy
    assert e.value.code == L.EXIT_PREFLIGHT


def test_inventory_every_training_entry_point_is_guarded():
    """Static inventory: every function that launches evaluation or measurement calls the guard (or is a wrapper of one)."""
    src = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "iocevaluator").glob("*.py")}
    assert "enforce_launch_guard(what=\"evaluate_cv\"" in src["evaluation.py"]
    assert "enforce_launch_guard(what=\"evaluate_baseline_cv\"" in src["cti_baselines.py"] or "evaluate_baseline_cv" in src["cti_baselines.py"]
    assert "refuse_at_scale(\"experiments.run_cv" in src["experiments.py"]
    assert "enforce_launch_guard" in src["scalability.py"] and "_refuse_non_development" in src["path_budget.py"]
    assert "refuse_at_scale" in src["workflows.py"]
    for wrapper in ("ablation.py", "model_comparison.py"):
        assert "evaluate_cv(" in src[wrapper]                                 # funnels through the guarded harness
