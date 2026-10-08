"""Frozen protocol, path-integrity enforcement, sensitivity configs and the final-experiment preflight."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iocevaluator.evaluation import EvalConfig
from iocevaluator.path_tracing import PathConfig
from iocevaluator.protocol import (FROZEN, FeaturePolicy, PathIntegrityError, PreflightError, check_case_integrity,
                                   exhaustive_path_config, final_experiment_preflight, primary_ranker_config,
                                   sensitivity_ranker_configs, validate_path_rows)
from iocevaluator.ranking import RankerConfig, ThreatPrioritisationRanker

ROOT = Path(__file__).resolve().parents[1]
PF = ROOT / "PROTOCOL_FREEZE.md"
POLICY = FeaturePolicy(("cwe",), (), "unit-test policy")


class DS:
    source = "synthetic"


def pre(cfg=None, **kw):
    kw.setdefault("allow_dirty", True)
    kw.setdefault("protocol_file", PF)
    kw.setdefault("feature_policy", FeaturePolicy((), (), "none required: unit test"))
    cfg = cfg or EvalConfig()
    return final_experiment_preflight(cfg, kw.pop("ranker_cfg", None), kw.pop("dataset", DS()), **kw)


def failed(rep):
    return {c.name for c in rep.failures()}


def test_runtime_defaults_are_the_frozen_protocol():
    cfg = EvalConfig()
    assert cfg.paths == exhaustive_path_config() and cfg.enforce_path_integrity
    assert cfg.paths.max_edges == 6 and cfg.paths.conf_threshold == 0.5 and cfg.paths.max_candidates == 5000
    assert cfg.paths.max_expansions_per_seed == 5_000_000 and cfg.paths.max_paths_per_seed >= 10 ** 9
    assert dict(cfg.paths.weights) == {"priority": 1.0, "confidence": 1.0, "structure_weight": 1.0, "coherence": 0.0}
    r = RankerConfig()
    assert (r.alpha, r.tau, r.ec_scope) == (0.5, None, "type") and r == primary_ranker_config()
    assert ThreatPrioritisationRanker().cfg == primary_ranker_config()
    assert (cfg.n_splits, list(cfg.seeds), cfg.fold_seed, cfg.val_fraction) == (10, [0, 1, 2, 3, 4], 0, 0.10)


def test_case_integrity_raises_only_on_budget_hit():
    check_case_integrity("c", False)
    with pytest.raises(PathIntegrityError, match="c"):
        check_case_integrity("c", True, 1, 0)


def test_validate_rows():
    ok = pd.DataFrame({"case_id": ["a"], "fold": [0], "seed": [0], "search_budget_hit": [False]})
    validate_path_rows(ok)
    validate_path_rows(pd.DataFrame())
    with pytest.raises(PathIntegrityError):
        validate_path_rows(ok.assign(search_budget_hit=True))
    with pytest.raises(PathIntegrityError):
        validate_path_rows(ok.drop(columns="search_budget_hit"))


@pytest.fixture(scope="module")
def dev_ds(tmp_path_factory):
    from iocevaluator.datasets import load_dataset
    from iocevaluator.synthetic_tikg import generate, write_dataset
    root = tmp_path_factory.mktemp("synthetic")
    write_dataset(generate("dev", 42), root / "dev")
    return load_dataset("synthetic", profile="dev", root=root)


def _smoke(paths, **kw):
    from iocevaluator.supervised_gcn import SupervisedGCNConfig
    return EvalConfig(seeds=(0,), max_folds=1, model=SupervisedGCNConfig(max_epochs=3, lr=1e-2), paths=paths, **kw)


def test_harness_fails_integrity_when_a_budget_is_hit(dev_ds):
    from iocevaluator.evaluation import evaluate_dataset
    tight = replace(exhaustive_path_config(), max_paths_per_seed=1, max_expansions_per_seed=5)
    with pytest.raises(PathIntegrityError):
        evaluate_dataset(dev_ds, _smoke(tight), use_reference_paths=True)


def test_exhaustive_default_runs_without_a_hit_and_passes_validation(dev_ds):
    from iocevaluator.evaluation import evaluate_dataset
    res = evaluate_dataset(dev_ds, _smoke(exhaustive_path_config(max_edges=4)), use_reference_paths=True)
    assert len(res.paths) and not res.paths["search_budget_hit"].any()
    validate_path_rows(res.paths)
    assert res.manifest["paths"]["enforce_path_integrity"] is True


def test_opt_out_is_explicit_and_fails_the_preflight(dev_ds):
    from iocevaluator.evaluation import evaluate_dataset
    tight = replace(exhaustive_path_config(), max_paths_per_seed=1, max_expansions_per_seed=5)
    cfg = _smoke(tight, enforce_path_integrity=False)
    res = evaluate_dataset(dev_ds, cfg, use_reference_paths=True)
    assert res.paths["search_budget_hit"].any()
    assert "path_integrity_enforced" in failed(pre(cfg))
    with pytest.raises(PathIntegrityError):
        validate_path_rows(res.paths)


def test_sensitivity_configs_vary_one_setting_and_are_not_primary():
    prim = primary_ranker_config()
    cfgs = sensitivity_ranker_configs()
    assert prim not in cfgs.values() and "alpha_tuned_on_validation" in cfgs
    for name, c in cfgs.items():
        changed = [f for f in ("alpha", "tau", "ec_scope") if getattr(c, f) != getattr(prim, f)]
        assert len(changed) == 1 and changed[0] == ("tau" if name.startswith("tau") else "alpha"), name
    for c in cfgs.values():
        assert final_experiment_preflight(EvalConfig(), c, DS(), feature_policy=POLICY, protocol_file=PF,
                                          allow_dirty=True).ok is False        # a sensitivity ranker is never a primary run


def test_preflight_passes_with_the_frozen_defaults_and_a_policy():
    rep = pre()
    assert rep.ok, [(c.name, c.detail) for c in rep.failures()]
    rec = rep.record
    assert rec["protocol_sha256"] and rec["config_sha256"] and rec["code_sha256"] and rec["git"]["commit"]
    assert rec["protocol_open_items"] == ["real_data_feature_exclusion_list"]


def test_preflight_requires_a_feature_policy():
    assert "feature_exclusion_policy_supplied" in failed(pre(feature_policy=None))
    assert "feature_exclusion_policy_supplied" in failed(pre(feature_policy=FeaturePolicy((), (), "")))
    assert "feature_exclusion_policy_supplied" in failed(pre(feature_policy=FeaturePolicy((), (), "TODO")))
    assert "feature_policy_applied" in failed(pre(feature_policy=POLICY))      # FeatureConfig does not exclude "cwe"


def test_preflight_catches_each_deviation():
    base = EvalConfig()
    assert "folds_and_seeds_fixed" in failed(pre(replace(base, seeds=(0, 1))))
    assert "folds_and_seeds_fixed" in failed(pre(replace(base, max_folds=1)))
    assert "folds_and_seeds_fixed" in failed(pre(replace(base, n_splits=5)))
    assert "paths_match_frozen" in failed(pre(replace(base, paths=PathConfig())))
    assert "paths_match_frozen" in failed(pre(replace(base, paths=replace(base.paths, max_edges=5))))
    assert "paths_match_frozen" in failed(pre(replace(base, paths=replace(base.paths, conf_threshold=0.6))))
    assert "paths_match_frozen" in failed(pre(replace(base, paths=replace(base.paths, weights=(("priority", 1.0), ("confidence", 1.0), ("structure_weight", 1.0), ("coherence", 0.5))))))
    assert "ranking_matches_frozen" in failed(pre(ranker_cfg=RankerConfig(alpha=0.7)))
    assert "ranking_matches_frozen" in failed(pre(ranker_cfg=RankerConfig(ec_scope="global")))
    assert {"no_unresolved_configuration", "ranking_matches_frozen"} <= failed(pre(ranker_cfg=RankerConfig(alpha=None)))
    assert "dataset_provenance_known" in failed(pre(dataset=type("X", (), {"source": "mystery"})()))
    assert "dataset_provenance_known" in failed(pre(dataset=None))


def test_preflight_requires_a_frozen_protocol_file_and_no_other_open_item(tmp_path):
    txt = PF.read_text(encoding="utf-8")
    f = tmp_path / "p.md"
    f.write_text(txt.replace("Protocol-status: FROZEN", "Protocol-status: PROPOSED"), encoding="utf-8")
    assert "protocol_marked_frozen" in failed(pre(protocol_file=f))
    f.write_text(txt.replace("Open-items: real_data_feature_exclusion_list", "Open-items: real_data_feature_exclusion_list, tau"), encoding="utf-8")
    assert "no_unresolved_configuration" in failed(pre(protocol_file=f))
    assert "protocol_marked_frozen" in failed(pre(protocol_file=tmp_path / "missing.md"))


def test_preflight_checks_the_result_and_the_working_tree(dev_ds):
    from iocevaluator.evaluation import evaluate_dataset
    good = evaluate_dataset(dev_ds, _smoke(exhaustive_path_config(max_edges=4)), use_reference_paths=True)
    assert "no_search_budget_hit" not in failed(pre(result=good))
    bad = type("R", (), {"paths": good.paths.assign(search_budget_hit=True)})()
    assert "no_search_budget_hit" in failed(pre(result=bad))
    assert "clean_working_tree" in failed(pre(allow_dirty=False)) or pre(allow_dirty=False).record["git"]["dirty"] is False
    with pytest.raises(PreflightError):
        pre(feature_policy=None).raise_if_failed()


def test_protocol_file_states_the_frozen_settings():
    t = PF.read_text(encoding="utf-8")
    assert "Protocol-status: FROZEN" in t and "Open-items: real_data_feature_exclusion_list" in t
    for s in ("5,000,000", "paired Wilcoxon", "Holm", "exhaustive", "**None**"):
        assert s in t or s.lower() in t.lower(), s
    assert "tuned" not in FROZEN["ranking"] and FROZEN["ranking"]["alpha"] == 0.5
