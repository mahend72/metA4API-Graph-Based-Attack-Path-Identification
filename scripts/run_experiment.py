#!/usr/bin/env python
"""Guarded launcher for FINAL / manuscript-scale experiments (evaluate | ablation | models | baselines).

Runs the frozen-protocol preflight first (and aborts before training on any failure), runs inside the preflight session,
re-checks search_budget_hit afterwards, and records the preflight in every manifest.  See iocevaluator/launch.py.

    python scripts/run_experiment.py --task models --source real --tikg T.json --labels L.json --severity S.json --paths P.json \\
        --feature-policy policy.json --out results/final_models
    python scripts/run_experiment.py ... --dry-run        # preflight only (clean-tree requirement bypassed), no training
    python scripts/run_experiment.py ... --allow-dirty    # NON-FINAL diagnostic run; never reportable
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iocevaluator.launch import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
