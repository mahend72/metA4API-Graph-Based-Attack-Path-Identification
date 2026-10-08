"""Print the chi_1..chi_20 table and per-structure matrix shapes / non-zero counts for a dataset.

    python scripts/chi_report.py --source synthetic --profile dev
    python scripts/chi_report.py --source semi_synthetic --profile manuscript_scale
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iocevaluator.datasets import load_dataset  # noqa: E402
from iocevaluator.metagraphs import structure_report  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--source", choices=["synthetic", "semi_synthetic"], default="synthetic")
ap.add_argument("--profile", default="dev")
a = ap.parse_args()
ds = load_dataset(a.source, profile=a.profile)
print(ds.banner())
print(f"{'chi':>4} {'kind':<5} {'type':<14} {'shape':<12} {'nnz':>7} {'offdiag':>8}  formula")
for r in structure_report(ds.tikg):
    sh = f"{r['shape'][0]}x{r['shape'][1]}"
    print(f"{r['id']:>4} {r['kind']:<5} {r['node_type']:<14} {sh:<12} {r['nnz']:>7} {r['offdiag_nnz']:>8}  {r['formula']}")
