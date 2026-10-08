"""Cost / scalability measurement (synthetic provenance).  Measures the current implementation; no manuscript value is used.

    python scripts/run_scalability.py --sizes 250 500 1000 2000 --out results/scalability
    python scripts/run_scalability.py --sizes 250 500 1000 2000 3728 --out results/scalability_full   # manuscript-scale profile
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iocevaluator.launch import add_preflight_arguments, do_preflight, eval_config_for, policy_from_args, run_class_from_flags  # noqa: E402
from iocevaluator.protocol import (MANUSCRIPT_SCALE_NODES, PreflightError, PreflightRequiredError, RUN_DRY, RUN_FINAL,  # noqa: E402
                                   exhaustive_path_config, preflight_session)
from iocevaluator.path_tracing import PathConfig  # noqa: E402
from iocevaluator.evaluation import EvalConfig  # noqa: E402
from iocevaluator.scalability import DEFAULT_SIZES, ScalabilityConfig, run_scalability, write_scalability  # noqa: E402


def _generator_spec_dataset(cfg):
    """Provenance for the preflight without generating the graph twice: synthetic, identified by profile + generator seed."""
    import hashlib
    import json
    from dataclasses import asdict
    from types import SimpleNamespace
    from iocevaluator.scalability import scaled_profile
    spec = {"sizes": list(cfg.sizes), "seed": cfg.seed, "profiles": {str(n): asdict(scaled_profile(n)) for n in cfg.sizes}}
    return SimpleNamespace(source="synthetic", name=f"scalability_{max(cfg.sizes)}", metadata={"profile": f"scale_{max(cfg.sizes)}", "seed": cfg.seed},
                           fingerprint_override="generator-spec:" + hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats-fast", type=int, default=5)
    ap.add_argument("--repeats-slow", type=int, default=3)
    ap.add_argument("--path-cases", type=int, default=12, help="reference cases traced per size (0 = all)")
    ap.add_argument("--path-search", choices=["exhaustive", "legacy"], default="exhaustive",
                    help="exhaustive: the proposed frozen search (no per-seed path budget, expansion safety ceiling, max_edges 6); "
                         "legacy: the earlier 2000 / 50000 / 5000 budgets (truncates; kept for comparison)")
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", default="results/scalability")
    add_preflight_arguments(ap)
    a = ap.parse_args(argv)
    cfg = ScalabilityConfig(sizes=tuple(a.sizes), seed=a.seed, repeats_fast=a.repeats_fast, repeats_slow=a.repeats_slow,
                            path_cases=a.path_cases or None, measure_memory=not a.no_memory, cache_dir=a.cache_dir,
                            paths=exhaustive_path_config() if a.path_search == "exhaustive" else PathConfig())
    # A manuscript-scale size (>= 3,000 nodes) makes this a manuscript-scale experiment: the preflight is mandatory (and a
    # truncating 'legacy' path search then fails it).  --dry-run runs the preflight at any size and stops before measuring.
    scale = max(cfg.sizes) >= MANUSCRIPT_SCALE_NODES
    if scale or a.dry_run or a.feature_policy is not None:
        policy = policy_from_args(a.feature_policy)
        ds = _generator_spec_dataset(cfg)
        try:
            rep = do_preflight(eval_config_for(policy, EvalConfig(paths=cfg.paths)), None, ds,
                               run_class=run_class_from_flags(a.dry_run, a.allow_dirty), feature_policy=policy,
                               protocol_file=a.protocol_file, repo=a.repo_root)
        except (PreflightError, PreflightRequiredError) as e:
            print(e)
            return 2
        if a.dry_run:
            print("dry run: preflight passed; nothing was measured")
            return 0
        with preflight_session(rep, eval_config_for(policy, EvalConfig(paths=cfg.paths))):
            res = run_scalability(cfg, verbose=True)
    else:
        res = run_scalability(cfg, verbose=True)           # development sizes only: the manifest says 'unguarded_development'
    for name, p in write_scalability(a.out, res).items():
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
