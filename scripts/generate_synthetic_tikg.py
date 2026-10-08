"""Generate the SYNTHETIC TIKG datasets (development / pipeline validation only).

    python scripts/generate_synthetic_tikg.py --profile dev --seed 42
    python scripts/generate_synthetic_tikg.py --profile manuscript_scale --seed 42

Semi-synthetic mode (real public NVD CVEs as vulnerability nodes, everything else synthetic):

    python scripts/fetch_cve_pool.py                       # once; fills data/raw/cve/
    python scripts/generate_synthetic_tikg.py --profile manuscript_scale --vulnerability-source real --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iocevaluator.synthetic_tikg import PROFILES, check_reproducible, generate, validate, write_dataset  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=[*PROFILES, "all"], default="dev")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vulnerability-source", choices=["synthetic", "real"], default="synthetic",
                    help="'real' = semi-synthetic dataset with real NVD CVEs (written to data/semi_synthetic/)")
    ap.add_argument("--cve-cache", type=Path, default=Path("data/raw/cve"), help="directory holding cve_pool.json")
    ap.add_argument("--refresh-cve", action="store_true", help="(re)fetch the CVE pool from NVD before generating")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: data/synthetic (synthetic) or data/semi_synthetic (real CVEs)")
    ap.add_argument("--include-x1", action="store_true",
                    help="also generate the NON-manuscript extension triplet <TA, uses, F> (off by default)")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--check-reproducible", action="store_true", help="regenerate in memory and compare")
    ap.add_argument("--report", action="store_true", help="print the full statistics report as JSON")
    a = ap.parse_args(argv)
    rc = 0
    real = a.vulnerability_source == "real"
    out_dir = a.out_dir or Path("data/semi_synthetic" if real else "data/synthetic")
    if real and a.refresh_cve:
        from iocevaluator.synthetic_tikg.cve_pool import build_pool, fetch_all
        fetch_all(a.cve_cache, refresh=False)
        build_pool(a.cve_cache)
    if real and not (a.cve_cache / "cve_pool.json").exists():
        ap.error(f"{a.cve_cache}/cve_pool.json missing - run scripts/fetch_cve_pool.py (or pass --refresh-cve)")
    for name in (list(PROFILES) if a.profile == "all" else [a.profile]):
        ds = generate(name, a.seed, a.include_x1, a.vulnerability_source, a.cve_cache)
        out = write_dataset(ds, out_dir / name)
        print(f"[{name}] {'SEMI-SYNTHETIC' if real else 'SYNTHETIC'} dataset written to {out} "
              f"({len(ds.nodes)} nodes, {len(ds.edges)} edges, {len(ds.campaigns['campaigns'])} campaigns, "
              f"{len(ds.attack_paths)} reference paths)")
        if not a.no_validate:
            rep = validate(ds)
            print(rep.summary())
            rc |= 0 if rep.ok else 1
            if a.report:
                print(json.dumps(rep.stats, indent=1, default=float))
        if a.check_reproducible:
            ok = check_reproducible(name, a.seed, a.include_x1, a.vulnerability_source, a.cve_cache)
            print(f"[{'PASS' if ok else 'FAIL'}] same seed reproduces identical dataset")
            rc |= 0 if ok else 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
