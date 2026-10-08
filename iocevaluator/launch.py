"""The guarded launcher: the one route for a final / manuscript-scale experiment.

    python scripts/run_experiment.py --task models --source real --tikg T --labels L --severity S --paths P \\
           --feature-policy policy.json --out results/final_models                    # FINAL: clean tree required
    python scripts/run_experiment.py ... --dry-run                                    # preflight only, no training
    python scripts/run_experiment.py ... --allow-dirty                                # non-final diagnostic run

Order of events (``launch``):
  1. ``final_experiment_preflight`` runs and its summary is printed; any failure aborts BEFORE anything trains;
  2. the experiment runs inside ``preflight_session`` (the harness itself refuses a different configuration);
  3. after the run, ``search_budget_hit`` and the fold count are re-checked on every produced row: any case that hit the
     safety ceiling fails the experiment (no result tables are written, only ``integrity_failure.json``);
  4. outputs are written; every manifest carries the preflight record and the post-run integrity verdict.

Run classes:  ``final`` (default: clean tree required, reportable only on real data), ``dry_run`` (``--dry-run``: the whole
preflight with the clean-tree requirement bypassed, then stop), ``diagnostic`` (``--allow-dirty``: trains on an uncommitted
tree; the manifest says so and the result is never reportable).  Nothing else relaxes a check: protocol settings, dataset
provenance, folds / seeds, chi1..chi20 weights, ranking and path settings, the feature-exclusion policy and path-integrity
enforcement are checked in every mode."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from .evaluation import EvalConfig
from .protocol import (FROZEN, RUN_CLASSES, RUN_DIAGNOSTIC, RUN_DRY, RUN_FINAL, ExperimentIntegrityError, FeaturePolicy,
                       PathIntegrityError, PreflightError, verify_outputs, PreflightReport, PreflightRequiredError, annotate_integrity, final_experiment_preflight, load_feature_policy,
                       post_run_integrity, preflight_manifest_block, preflight_session, preflight_summary,
                       primary_ranker_config)
from .ranking import RankerConfig
from .tikg_features import FeatureConfig

TASKS = ("evaluate", "ablation", "models", "baselines")
EXIT_OK, EXIT_PREFLIGHT, EXIT_INTEGRITY, EXIT_USAGE = 0, 2, 3, 64
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------------ CLI surface
def add_preflight_arguments(ap: argparse.ArgumentParser, *, required_policy: bool = False) -> None:
    g = ap.add_argument_group("frozen-protocol preflight")
    g.add_argument("--dry-run", action="store_true",
                   help="run the whole preflight (the clean-tree requirement is bypassed), print the summary and stop before any training")
    g.add_argument("--allow-dirty", action="store_true",
                   help="NON-FINAL diagnostic run only: train on an uncommitted tree. The manifest is marked diagnostic and the "
                        "result is never reportable. Without this flag (and without --dry-run) a dirty repository is rejected.")
    g.add_argument("--feature-policy", type=Path, required=required_policy,
                   help='JSON {"exclude_fields": [...], "strip_terms": [...], "source": "..."}; an empty list needs '
                        '"source": "none required: <reason>". Required in every mode.')
    g.add_argument("--protocol-file", type=Path, default=PACKAGE_ROOT / "PROTOCOL_FREEZE.md")
    g.add_argument("--repo-root", type=Path, default=None, help="repository whose commit / dirty state is recorded and checked (default: this package's repository)")


def run_class_from_flags(dry_run: bool, allow_dirty: bool) -> str:
    """``--dry-run`` wins (it never trains); ``--allow-dirty`` makes a non-final diagnostic run; otherwise FINAL."""
    return RUN_DRY if dry_run else RUN_DIAGNOSTIC if allow_dirty else RUN_FINAL


def policy_from_args(path: Optional[Path]) -> Optional[FeaturePolicy]:
    return load_feature_policy(path) if path is not None else None


def eval_config_for(policy: Optional[FeaturePolicy], base: Optional[EvalConfig] = None) -> EvalConfig:
    """The frozen EvalConfig with the supplied feature policy applied to the FeatureConfig (the preflight then checks that the
    policy and the features really agree).  Nothing is defaulted when no policy is given: the preflight will fail."""
    base = base or EvalConfig()
    if policy is None:
        return base
    return replace(base, features=replace(base.features, exclude_fields=tuple(policy.exclude_fields),
                                          strip_terms=tuple(policy.strip_terms)))


# ------------------------------------------------------------------------------------------------ preflight + run
def do_preflight(cfg: EvalConfig, ranker_cfg: Optional[RankerConfig], dataset, *, run_class: str,
                 feature_policy: Optional[FeaturePolicy], protocol_file: Path = PACKAGE_ROOT / "PROTOCOL_FREEZE.md",
                 repo: Optional[Path] = None, role: str = "primary", echo: Callable[[str], None] = print) -> PreflightReport:
    """Run the preflight, print the concise summary, and raise ``PreflightError`` on any failure.  Only a non-final run
    (``dry_run`` / ``diagnostic``) bypasses the clean-tree requirement; a final run rejects a dirty repository."""
    if run_class not in RUN_CLASSES:
        raise PreflightError(f"unknown run class {run_class!r}")
    rep = final_experiment_preflight(cfg, ranker_cfg, dataset, feature_policy=feature_policy, protocol_file=protocol_file,
                                     allow_dirty=run_class != RUN_FINAL, repo=repo or PACKAGE_ROOT, role=role, run_class=run_class)
    echo(preflight_summary(rep))
    return rep.raise_if_failed()


def _run_task(task: str, dataset, cfg: EvalConfig, *, variants: Sequence[str], models: Sequence[str], baselines: Sequence[str],
              protocol_ok: bool, verbose: bool, use_reference_paths: bool = True):
    if task == "evaluate":
        from .evaluation import evaluate_dataset, write_evaluation
        return evaluate_dataset(dataset, cfg, verbose=verbose, use_reference_paths=use_reference_paths), write_evaluation
    if task == "ablation":
        from .ablation import run_ablation_dataset, write_ablation
        return run_ablation_dataset(dataset, variants or ("all",), cfg, use_reference_paths=use_reference_paths,
                                    protocol_frozen=protocol_ok, verbose=verbose), write_ablation
    if task == "models":
        from .model_comparison import DEFAULT_MODELS, run_model_comparison_dataset, write_model_comparison
        return run_model_comparison_dataset(dataset, models or DEFAULT_MODELS, cfg, use_reference_paths=use_reference_paths,
                                            protocol_frozen=protocol_ok, verbose=verbose), write_model_comparison
    if task == "baselines":
        from .cti_baselines import DEFAULT_BASELINES, run_baseline_comparison_dataset, write_baseline_comparison
        return run_baseline_comparison_dataset(dataset, baselines or DEFAULT_BASELINES, cfg=cfg, protocol_frozen=protocol_ok,
                                               verbose=verbose), write_baseline_comparison
    raise PreflightError(f"unknown task {task!r}; valid: {TASKS}")


def launch(task: str, dataset, *, run_class: str, feature_policy: Optional[FeaturePolicy],
           cfg: Optional[EvalConfig] = None, ranker_cfg: Optional[RankerConfig] = None, out: Optional[Path] = None,
           protocol_file: Path = PACKAGE_ROOT / "PROTOCOL_FREEZE.md", repo: Optional[Path] = None,
           variants: Sequence[str] = (), models: Sequence[str] = (), baselines: Sequence[str] = (),
           verbose: bool = False, echo: Callable[[str], None] = print, role: str = "primary",
           use_reference_paths: bool = True) -> Dict[str, Any]:
    """Preflight -> (stop if dry run) -> run inside the preflight session -> post-run integrity -> write outputs.

    Raises ``PreflightError`` (before anything trains) or ``ExperimentIntegrityError`` (after the run, nothing written but
    ``integrity_failure.json``).  Returns ``{"report", "results", "files", "integrity"}``."""
    cfg = cfg if cfg is not None else eval_config_for(feature_policy)
    if cfg.intent == "development":
        cfg = replace(cfg, intent="final")                 # a launched experiment is explicitly marked: every layer treats it as final
    ranker_cfg = ranker_cfg or primary_ranker_config()
    rep = do_preflight(cfg, ranker_cfg, dataset, run_class=run_class, feature_policy=feature_policy,
                       protocol_file=protocol_file, repo=repo, role=role, echo=echo)
    out_dir = Path(out) if out is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "preflight.json").write_text(json.dumps(preflight_manifest_block(rep), indent=2, sort_keys=True, default=str), encoding="utf-8")
    if run_class == RUN_DRY:
        echo("dry run: preflight passed; nothing was trained")
        return {"report": rep, "results": None, "files": {}, "integrity": None}
    with preflight_session(rep, cfg, ranker_cfg):
        results, writer = _run_task(task, dataset, cfg, variants=variants, models=models, baselines=baselines,
                                    protocol_ok=rep.ok, verbose=verbose, use_reference_paths=use_reference_paths)
    integrity = post_run_integrity(results, expect_folds=FROZEN["n_splits"], raise_on_fail=False, final=True)
    annotate_integrity(results, integrity)
    if not integrity["passed"]:
        if out_dir is not None:
            (out_dir / "integrity_failure.json").write_text(json.dumps({"preflight": preflight_manifest_block(rep), "integrity": integrity},
                                                                       indent=2, sort_keys=True, default=str), encoding="utf-8")
        raise ExperimentIntegrityError("post-run integrity check failed (no result tables written): " + " | ".join(integrity["failures"]))
    files = writer(out_dir, results) if out_dir is not None else {}
    if out_dir is not None:                                  # the written outputs must be complete and mutually consistent
        n_var = len(getattr(results, "per_variant", None) or {}) or 1
        bad = verify_outputs(out_dir, files, preflight_manifest_block(rep), n_variants=n_var)
        if bad:
            integrity = {**integrity, "passed": False, "failures": integrity["failures"] + bad}
            (out_dir / "integrity_failure.json").write_text(json.dumps({"preflight": preflight_manifest_block(rep), "integrity": integrity},
                                                                       indent=2, sort_keys=True, default=str), encoding="utf-8")
            raise ExperimentIntegrityError("output verification failed (outputs must not be used): " + " | ".join(bad))
    echo(f"integrity: passed ({integrity['path_rows_checked']} path rows re-checked, no search_budget_hit)")
    return {"report": rep, "results": results, "files": files, "integrity": integrity}


# ------------------------------------------------------------------------------------------------ command line
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="run_experiment", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--source", choices=["synthetic", "semi_synthetic", "real"], required=True)
    ap.add_argument("--profile", default="dev", help="synthetic / semi_synthetic profile")
    ap.add_argument("--tikg", type=Path)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--severity", type=Path)
    ap.add_argument("--paths", type=Path, help="reference paths (evaluation only)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--variants", nargs="*", default=[])
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--baselines", nargs="*", default=[])
    ap.add_argument("--quiet", action="store_true")
    add_preflight_arguments(ap)
    return ap


def main(argv: Optional[Sequence[str]] = None, *, echo: Callable[[str], None] = print) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.source == "real" and not (a.tikg and a.labels):
        ap.error("--source real needs --tikg and --labels")
    from .datasets import load_dataset
    ds = load_dataset(a.source, profile=a.profile, tikg_path=a.tikg, labels_path=a.labels, severity_path=a.severity, paths_path=a.paths)
    try:
        launch(a.task, ds, run_class=run_class_from_flags(a.dry_run, a.allow_dirty), feature_policy=policy_from_args(a.feature_policy),
               out=a.out, protocol_file=a.protocol_file, repo=a.repo_root, variants=a.variants, models=a.models, baselines=a.baselines,
               verbose=not a.quiet, echo=echo)
    except (PreflightError, PreflightRequiredError) as e:
        echo(str(e))
        return EXIT_PREFLIGHT
    except PathIntegrityError as e:                  # includes ExperimentIntegrityError (post-run) and the in-run per-case raise
        echo(f"INTEGRITY FAILURE: {e}")
        return EXIT_INTEGRITY
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
