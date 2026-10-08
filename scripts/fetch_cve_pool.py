"""Fetch real public CVE metadata from the NVD API 2.0 into data/raw/cve/ (resumable) and build cve_pool.json.

    python scripts/fetch_cve_pool.py                # fetch missing queries, rebuild the pool
    python scripts/fetch_cve_pool.py --refresh      # re-fetch everything
    python scripts/fetch_cve_pool.py --build-only   # just re-merge cached query files

Set NVD_API_KEY for a faster rate limit (otherwise ~6.5 s between requests).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iocevaluator.synthetic_tikg.cve_pool import DEFAULT_CACHE, build_pool, fetch_all  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
ap.add_argument("--refresh", action="store_true")
ap.add_argument("--build-only", action="store_true")
a = ap.parse_args()
if not a.build_only:
    fetch_all(a.cache, a.refresh)
print("pool:", build_pool(a.cache))
