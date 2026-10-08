#!/usr/bin/env python
"""Path-tracing budget-sensitivity experiment on development data (never on manuscript-scale data).

    python scripts/run_path_budget.py --source synthetic --out results/path_budget/synthetic
    python scripts/run_path_budget.py --source semi_synthetic --out results/path_budget/semi_synthetic

Writes path_budget_{cases,summary,convergence,memory}.csv and path_budget_manifest.json."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iocevaluator.datasets import load_dataset                      # noqa: E402
from iocevaluator.path_budget import (EDGE_LIMITS, LEVELS, build_fold_inputs, run_budget_sensitivity,  # noqa: E402
                                      write_budget)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["synthetic", "semi_synthetic"], required=True)
    ap.add_argument("--profile", default="dev")
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", nargs="+", default=[n for n, _ in LEVELS])
    ap.add_argument("--edge-limits", nargs="+", type=int, default=list(EDGE_LIMITS))
    ap.add_argument("--memory-cases", type=int, default=6)
    ap.add_argument("--plant-reference", action="store_true",
                    help="STRESS TEST: give the reference nodes maximal confidence/priority (dev data only; not a performance result)")
    ap.add_argument("--model-seed", type=int, default=0)
    a = ap.parse_args()
    ds = load_dataset(a.source, profile=a.profile)
    if a.profile == "manuscript_scale":
        raise SystemExit("the budget experiment is a development-data experiment; manuscript_scale is refused")
    inputs = build_fold_inputs(ds, seed=a.model_seed)
    res = run_budget_sensitivity(ds, inputs, a.levels, a.edge_limits, a.memory_cases, verbose=True, plant_reference=a.plant_reference)
    paths = write_budget(res, a.out, {"source": ds.source, "profile": a.profile, "model_seed": a.model_seed,
                                      "n_folds": len(inputs), "reference_planted_stress_test": a.plant_reference})
    for p in paths.values():
        print(p)


if __name__ == "__main__":
    main()
