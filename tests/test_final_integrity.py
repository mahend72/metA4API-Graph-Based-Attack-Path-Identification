"""Final-run integrity hardening: explicit final marking at any size, realised chi weights per fold, the 10x5 completeness /
isolation / hash layer, and non-reportability of legacy and low-level outputs.  No final experiment is run."""
import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iocevaluator import launch as L
from iocevaluator import megits
from iocevaluator import protocol as P
from iocevaluator.evaluation import EvalConfig, EvaluationResult, evaluate_cv
from iocevaluator.protocol import (INTENTS, ExperimentIntegrityError, PreflightRequiredError, RUN_FINAL, post_run_integrity,
                                   reportability, verify_chi_realisation, verify_outputs)
from iocevaluator.supervised_gcn import SupervisedGCNConfig

FAST = SupervisedGCNConfig(max_epochs=1, lr=1e-2)
SMOKE = EvalConfig(model=FAST, seeds=(0,), max_folds=1)


@pytest.fixture(scope="module")
def dev_ds(tmp_path_factory):
    from iocevaluator.datasets import load_dataset
    from iocevaluator.synthetic_tikg import generate, write_dataset
    root = tmp_path_factory.mktemp("synthetic")
    write_dataset(generate("dev", 42), root / "dev")
    return load_dataset("synthetic", profile="dev", root=root)


# ------------------------------------------------------------------------------------------------ a valid 10 x 5 result
def _chi_folds(n=10, **over):
    b = {"structure_ids": list(range(1, 21)), "weights": {f"chi{k}": 0.05 for k in range(1, 21)}, "binary": False, "excluded_nodes": 0}
    return [{"fold": f, "n_adjacency_builds": 2, "builds": [dict(b, **over), dict(b, **over)]} for f in range(n)]


def make_result(real=False, clean=True, run_class=RUN_FINAL, hashes=("fp", "proto", "cfg", "code", "commit")):
    fp, proto, cfg, code, commit = hashes
    camps = [f"c{i}" for i in range(20)]
    finfo = []
    for f in range(10):
        test = camps[2 * f: 2 * f + 2]
        rest = [c for c in camps if c not in test]
        finfo.append({"fold": f, "train_campaigns": rest[:-2], "val_campaigns": rest[-2:], "test_campaigns": test})
    rows = [{"fold": f, "seed": s, "macro_f1": 0.5, "micro_f1": 0.6} for f in range(10) for s in range(5)]
    runs = pd.DataFrame(rows)
    by_type = pd.DataFrame([{"fold": f, "seed": s, "entity_type": "ioc"} for f in range(10) for s in range(5)])
    tim = pd.DataFrame([{"fold": f, "seed": s} for f in range(10) for s in range(5)])
    pre = {"performed": True, "ok": True, "run_class": run_class, "protocol_sha256": proto, "config_sha256": cfg, "code_sha256": code,
           "git": {"commit": commit, "dirty": not clean}, "reportable_as_manuscript": bool(real and clean and run_class == RUN_FINAL)}
    man = {"dataset": {"fingerprint_sha256": fp}, "preflight": pre, "model": {}, "variant": {"name": "megits_full", "adjacency": "megits"},
           "protocol": {"folds_run": 10, "runs": [{"fold": f, "seed": s} for f in range(10) for s in range(5)]}, "folds": finfo,
           "paths": {"enabled": False}, "chi_realisation": P.chi_realisation_block(_chi_folds(), True)}
    return EvaluationResult(runs, by_type, [], man, tim, pd.DataFrame())


def fails(res, **kw):
    return post_run_integrity(res, final=True, raise_on_fail=False, **kw)["failures"]


def test_a_valid_10x5_result_passes_final_integrity():
    res = make_result(real=True)
    out = post_run_integrity(res, expect_folds=10, final=True)
    assert out["passed"] and out["level"] == "final" and not out["failures"]
    P.annotate_integrity(res, out)
    assert res.manifest["reportability"] == {"reportable": True, "reasons": []}


# ------------------------------------------------------------------------------------------------ completeness
def test_missing_fold_missing_seed_duplicate_and_extra_runs_fail():
    r = make_result()
    r.runs = r.runs[r.runs.fold != 7]
    f = fails(r)
    assert any("9 outer folds" in x for x in f) and any("missing (fold, seed)" in x for x in f)

    r = make_result()
    r.runs = r.runs[~((r.runs.fold == 3) & (r.runs.seed == 4))]
    f = fails(r)
    assert any("missing (fold, seed)" in x for x in f) and any("fold 3 completed seeds" in x for x in f) and any("49 run rows" in x for x in f)

    r = make_result()
    r.runs = pd.concat([r.runs, r.runs.iloc[[0]]], ignore_index=True)
    f = fails(r)
    assert any("duplicate (fold, seed)" in x for x in f) and any("51 run rows" in x for x in f)

    r = make_result()
    r.runs = pd.concat([r.runs, pd.DataFrame([{"fold": 10, "seed": 0, "macro_f1": 0.5, "micro_f1": 0.5}])], ignore_index=True)
    assert any("unexpected (fold, seed)" in x for x in fails(r))

    r = make_result()                                                      # a run silently dropped from the table but planned in the manifest
    r.runs = r.runs.iloc[:-1]
    assert any("silently omitted" in x for x in fails(r))

    r = make_result()
    r.runs.loc[5, "macro_f1"] = np.nan
    assert any("undefined macro_f1" in x for x in fails(r))
    r = make_result()
    r.by_type = r.by_type.iloc[:-3]
    assert any("per-type table" in x for x in fails(r))
    r = make_result()
    r.manifest["folds"] = r.manifest["folds"][:9]
    assert any("manifest records 9 folds" in x for x in fails(r))


def test_campaign_leakage_and_split_overlap_fail():
    r = make_result()
    r.manifest["folds"][2]["test_campaigns"] = r.manifest["folds"][2]["test_campaigns"] + [r.manifest["folds"][2]["train_campaigns"][0]]
    f = fails(r)
    assert any("train/test campaign overlap" in x for x in f) and any("more than one test fold" in x for x in f) or any("overlap" in x for x in f)

    r = make_result()
    r.manifest["folds"][4]["val_campaigns"] = [r.manifest["folds"][4]["test_campaigns"][0]]
    assert any("validation/test campaign overlap" in x for x in fails(r))

    r = make_result()                                                      # a campaign tested in two folds
    r.manifest["folds"][1]["test_campaigns"] = r.manifest["folds"][0]["test_campaigns"]
    assert any("more than one test fold" in x for x in fails(r))

    r = make_result()                                                      # a campaign that is never tested
    r.manifest["folds"][1]["test_campaigns"] = r.manifest["folds"][1]["test_campaigns"][:1]
    assert any("in no test fold" in x for x in fails(r))


# ------------------------------------------------------------------------------------------------ hashes / outputs
def test_inconsistent_hashes_fail():
    a, b = make_result(), make_result()
    for key, val in (("dataset", {"fingerprint_sha256": "OTHER"}),):
        b.manifest["dataset"] = val
    f = post_run_integrity({"a": a, "b": b}, final=True, raise_on_fail=False)["failures"]
    assert any("inconsistent dataset hash" in x for x in f)
    for field in ("protocol_sha256", "config_sha256", "code_sha256"):
        a, b = make_result(), make_result()
        b.manifest["preflight"][field] = "OTHER"
        f = post_run_integrity({"a": a, "b": b}, final=True, raise_on_fail=False)["failures"]
        assert any("inconsistent" in x for x in f), field
    a, b = make_result(), make_result()
    b.manifest["preflight"]["git"]["commit"] = "other"
    assert any("inconsistent commit hash" in x for x in post_run_integrity({"a": a, "b": b}, final=True, raise_on_fail=False)["failures"])
    a = make_result()
    a.manifest["preflight"]["code_sha256"] = None
    assert any("code hash missing" in x for x in fails(a))
    a = make_result()
    a.manifest["preflight"] = {"performed": False}
    assert any("no passing preflight" in x for x in fails(a))


def _write_outputs(tmp_path, n_runs=50, fp="fp", proto="proto", skip=None, empty=None):
    out = tmp_path / "o"
    out.mkdir()
    files = {}
    pre = {"protocol_sha256": proto, "config_sha256": "cfg", "code_sha256": "code", "performed": True, "ok": True, "git": {"commit": "commit"}}
    (out / "preflight.json").write_text(json.dumps(pre))
    pd.DataFrame({"fold": range(n_runs), "seed": 0}).to_csv(out / "results_per_run.csv", index=False)
    (out / "summary.json").write_text("{}")
    (out / "manifest.json").write_text(json.dumps({"dataset": {"fingerprint_sha256": fp}, "preflight": pre}))
    for n in ("results_per_run.csv", "summary.json", "manifest.json"):
        files[n] = out / n
    if skip:
        files[skip] = out / skip                                           # reported by the writer, never written
    if empty:
        (out / empty).write_text("")
    return out, files, pre


def test_incomplete_or_inconsistent_outputs_fail(tmp_path):
    out, files, pre = _write_outputs(tmp_path)
    assert verify_outputs(out, files, {"dataset": {"fingerprint_sha256": "fp"}}) == []
    t = tmp_path / "a"; t.mkdir()
    out, files, _ = _write_outputs(t, skip="results_per_fold.csv")
    assert any("results_per_fold.csv is missing" in x for x in verify_outputs(out, files, {}))
    t = tmp_path / "b"; t.mkdir()
    out, files, _ = _write_outputs(t, empty="summary.json")
    assert any("summary.json is missing or empty" in x for x in verify_outputs(out, files, {}))
    t = tmp_path / "c"; t.mkdir()
    out, files, _ = _write_outputs(t, n_runs=49)
    assert any("49 rows, expected 50" in x for x in verify_outputs(out, files, {}))
    t = tmp_path / "d"; t.mkdir()
    out, files, pre = _write_outputs(t)
    man = json.loads((out / "manifest.json").read_text())
    man["preflight"]["code_sha256"] = "tampered"
    (out / "manifest.json").write_text(json.dumps(man))
    assert any("inconsistent code hash" in x for x in verify_outputs(out, files, {}))
    t = tmp_path / "e"; t.mkdir()
    out, files, pre = _write_outputs(t)
    (out / "preflight.json").unlink()
    assert any("preflight.json is missing" in x for x in verify_outputs(out, files, {}))
    assert verify_outputs(tmp_path, {}, {}) == ["no output files were reported by the writer"]


def test_launch_fails_on_incomplete_written_outputs(dev_ds, tmp_path, monkeypatch):
    from .test_preflight_launch import POLICY, PF, frozen_cfg, clean_repo  # noqa: F401  (fixture re-use below)
    import subprocess
    repo = tmp_path / "repo"; repo.mkdir()
    for a in (["init", "-q"], ["config", "user.email", "t@e"], ["config", "user.name", "t"]):
        subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)
    (repo / "a").write_text("a")
    subprocess.run(["git", "add", "a"], cwd=repo, check=True); subprocess.run(["git", "commit", "-qm", "i"], cwd=repo, check=True, capture_output=True)

    def writer(out, res):
        return {"manifest.json": Path(out) / "manifest.json", "results_per_run.csv": Path(out) / "missing.csv"}
    monkeypatch.setattr(L, "_run_task", lambda *a, **k: (make_result(), writer))
    with pytest.raises(ExperimentIntegrityError, match="output verification failed"):
        L.launch("evaluate", dev_ds, run_class=RUN_FINAL, feature_policy=POLICY, cfg=frozen_cfg(), out=tmp_path / "o", repo=repo,
                 protocol_file=PF, echo=lambda s: None)
    assert (tmp_path / "o" / "integrity_failure.json").exists()


# ------------------------------------------------------------------------------------------------ explicit final marking
@pytest.mark.parametrize("intent", [i for i in INTENTS if i != "development"])
def test_a_small_graph_marked_final_still_requires_preflight(dev_ds, intent):
    assert dev_ds.tikg.n < P.MANUSCRIPT_SCALE_NODES
    with pytest.raises(PreflightRequiredError, match=f"explicitly marked '{intent}'"):
        evaluate_cv(dev_ds.tikg, dev_ds.labels, replace(SMOKE, intent=intent))
    from iocevaluator.cti_baselines import evaluate_baseline_cv, select_baselines
    with pytest.raises(PreflightRequiredError, match="explicitly marked"):
        evaluate_baseline_cv(dev_ds.tikg, dev_ds.labels, select_baselines(("attackg",))[0], replace(SMOKE, intent=intent))
    from iocevaluator.ablation import run_ablation_dataset
    from iocevaluator.cti_baselines import run_baseline_comparison_dataset
    from iocevaluator.model_comparison import run_model_comparison_dataset
    for fn in (run_ablation_dataset, run_model_comparison_dataset, run_baseline_comparison_dataset):
        with pytest.raises(PreflightRequiredError, match="no preflight session"):
            fn(dev_ds, cfg=replace(SMOKE, intent=intent))
        with pytest.raises(PreflightRequiredError, match="protocol_frozen"):
            fn(dev_ds, cfg=SMOKE, protocol_frozen=True)


def test_unknown_intent_is_rejected_and_development_stays_unguarded(dev_ds):
    with pytest.raises(PreflightRequiredError, match="unknown run intent"):
        evaluate_cv(dev_ds.tikg, dev_ds.labels, replace(SMOKE, intent="whatever"))
    res = evaluate_cv(dev_ds.tikg, dev_ds.labels, SMOKE)                   # development smoke run: still allowed
    assert res.manifest["preflight"]["performed"] is False


# ------------------------------------------------------------------------------------------------ chi weights inside the folds
def test_every_fold_records_the_realised_chi_structures_and_weights(dev_ds):
    res = evaluate_cv(dev_ds.tikg, dev_ds.labels, replace(SMOKE, max_folds=2))
    cr = res.manifest["chi_realisation"]
    assert cr["ok"] and cr["primary_builder"] and len(cr["folds"]) == 2 and not cr["problems"]
    for fr in cr["folds"]:
        assert fr["n_adjacency_builds"] >= 1
        for b in fr["builds"]:
            assert b["structure_ids"] == list(range(1, 21)) and b["binary"] is False
            assert set(b["weights"]) == {f"chi{k}" for k in range(1, 21)} and all(abs(v - 0.05) < 1e-12 for v in b["weights"].values())
    assert verify_chi_realisation(cr["folds"], expect_folds=2) == []
    json.loads(json.dumps(res.manifest, default=str))                       # survives a JSON round trip (hash-stable keys)


def test_skewed_missing_extra_or_binary_chi_realisation_fails(monkeypatch, dev_ds):
    real = megits.normalise_weights

    def skewed(structures, *a, **k):
        w = dict(real(structures, *a, **k)); w[next(iter(w))] *= 2
        return w
    monkeypatch.setattr(megits, "normalise_weights", skewed)
    res = evaluate_cv(dev_ds.tikg, dev_ds.labels, SMOKE)
    cr = res.manifest["chi_realisation"]
    assert not cr["ok"] and any("weights differ from 1/20" in p for p in cr["problems"])
    monkeypatch.undo()

    monkeypatch.setattr(megits, "default_structures", lambda: list(__import__("iocevaluator.metagraphs", fromlist=["STRUCTURES"]).STRUCTURES)[:19])
    res = evaluate_cv(dev_ds.tikg, dev_ds.labels, SMOKE)
    cr = res.manifest["chi_realisation"]
    assert not cr["ok"] and any("missing" in p for p in cr["problems"]) and any("differ from 1/20" in p for p in cr["problems"])
    monkeypatch.undo()

    ok = _chi_folds(2)
    extra = copy.deepcopy(ok); extra[0]["builds"][0]["structure_ids"].append(21); extra[0]["builds"][0]["weights"]["chi21"] = 0.05
    assert any("extra structure" in p for p in verify_chi_realisation(extra))
    dup = copy.deepcopy(ok); dup[0]["builds"][0]["structure_ids"].append(3)
    assert any("duplicated" in p for p in verify_chi_realisation(dup))
    binary = _chi_folds(2, binary=True)
    assert any("binary adjacency" in p for p in verify_chi_realisation(binary))
    assert any("no realised chi adjacency" in p for p in verify_chi_realisation([{"fold": 0, "builds": []}]))
    assert any("sum to" in p for p in verify_chi_realisation(_chi_folds(1, weights={f"chi{k}": 0.05 for k in range(1, 20)} | {"chi20": 0.06})))


def test_post_run_integrity_fails_on_wrong_realised_weights_and_foreign_builder():
    r = make_result()
    r.manifest["chi_realisation"] = P.chi_realisation_block(_chi_folds(10, weights={f"chi{k}": 1 / 19 for k in range(1, 21)}), True)
    assert any("differ from 1/20" in x for x in fails(r))
    r = make_result()
    r.manifest["chi_realisation"]["folds"] = r.manifest["chi_realisation"]["folds"][:9]
    assert any("expected 10" in x for x in fails(r))
    r = make_result()
    del r.manifest["chi_realisation"]
    assert any("no chi_realisation" in x for x in fails(r))
    r = make_result()                                                      # a recorded 'ok: true' cannot hide bad realised weights
    r.manifest["chi_realisation"]["folds"][3]["builds"][0]["weights"]["chi1"] = 0.5
    r.manifest["chi_realisation"]["ok"] = True
    assert any("fold 3" in x for x in fails(r))


def test_custom_adjacency_builder_is_recorded_but_not_primary(dev_ds):
    from iocevaluator.ablation import build_variant_adjacency, select_variants
    v = select_variants(["binary_semantic"])[0]
    res = evaluate_cv(dev_ds.tikg, dev_ds.labels, SMOKE, adj_builder=lambda g, f: build_variant_adjacency(v, g, f), variant_info=v.info())
    cr = res.manifest["chi_realisation"]
    assert cr["primary_builder"] is False and cr["ok"] and "ablation variant" in cr["builder"]


# ------------------------------------------------------------------------------------------------ path diagnostics
def test_path_metrics_must_expose_budget_hit_and_the_frozen_search():
    r = make_result()
    r.manifest["paths"] = {"enabled": True, "enforce_path_integrity": True, **{k: P.FROZEN["paths"][k] for k in
                           ("max_edges", "conf_threshold", "max_candidates", "max_expansions_per_seed", "max_paths_per_seed")}}
    r.paths = pd.DataFrame({"case_id": ["a"], "fold": [0], "seed": [0], "search_budget_hit": [False]})
    assert post_run_integrity(r, final=True, raise_on_fail=False)["passed"]
    for key, bad in (("max_edges", 5), ("conf_threshold", 0.6), ("max_candidates", 100), ("max_expansions_per_seed", 10), ("max_paths_per_seed", 50)):
        r2 = copy.deepcopy(r); r2.manifest["paths"][key] = bad
        assert any("frozen exhaustive" in x for x in fails(r2)), key
    r3 = copy.deepcopy(r); r3.paths = r3.paths.drop(columns="search_budget_hit")
    assert any("not exposed" in x for x in fails(r3))
    r4 = copy.deepcopy(r); r4.paths["search_budget_hit"] = True
    assert any("search budget" in x for x in fails(r4))


# ------------------------------------------------------------------------------------------------ reportability
def test_legacy_and_low_level_outputs_are_never_reportable(dev_ds, tmp_path):
    # low-level / unguarded harness manifest
    res = evaluate_cv(dev_ds.tikg, dev_ds.labels, SMOKE)
    assert reportability(res.manifest)["reportable"] is False and res.manifest["preflight"]["not_reportable"] is True
    P.annotate_integrity(res, post_run_integrity(res, raise_on_fail=False))        # even annotated with a basic integrity pass
    assert res.manifest["reportability"]["reportable"] is False
    # development-only manifests
    assert P.development_manifest_block("legacy_development_harness")["not_reportable"] is True
    assert reportability({"preflight": P.development_manifest_block("x")})["reportable"] is False
    # legacy path evaluation route (rank --reference-paths)
    from iocevaluator.workflows import rank
    from .synthetic import make_synthetic
    g, lab = make_synthetic(9)
    g.save(tmp_path / "t.json")
    (tmp_path / "l.json").write_text(json.dumps({"labels": lab}))
    summary = rank(tmp_path / "t.json", tmp_path / "l.json", tmp_path / "rk")
    assert summary["reportability"]["not_reportable"] is True and "search_budget_hit" in summary["reportability"]["reason"]
    assert json.loads((tmp_path / "rk" / "rank_summary.json").read_text())["reportability"]["reportable"] is False


def test_reportable_only_for_a_clean_final_real_run_that_passed_the_final_layer():
    res = make_result(real=True); P.annotate_integrity(res, post_run_integrity(res, final=True))
    assert res.manifest["reportability"]["reportable"]
    for kw, why in (({"real": False}, "not real"), ({"real": True, "clean": False}, "not clean"), ({"real": True, "run_class": "diagnostic"}, "class")):
        r = make_result(**kw)
        P.annotate_integrity(r, post_run_integrity(r, raise_on_fail=False, final=True))
        assert r.manifest["reportability"]["reportable"] is False, why
    r = make_result(real=True)
    P.annotate_integrity(r, post_run_integrity(r, raise_on_fail=False))              # basic layer only -> not enough
    assert r.manifest["reportability"]["reportable"] is False
    r = make_result(real=True); r.runs = r.runs.iloc[:-1]
    P.annotate_integrity(r, post_run_integrity(r, raise_on_fail=False, final=True))
    assert r.manifest["reportability"]["reportable"] is False
