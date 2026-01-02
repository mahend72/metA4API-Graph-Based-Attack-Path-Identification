"""Export OTX pulses to a normalized JSON file.

Usage:
  python scripts/export_otx_to_json.py --out data/threats.json --limit 200

You must set:
  OTX_API_KEY=...
in your environment (or in a .env file with python-dotenv).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from iocevaluator.logging_utils import setup_logging
from iocevaluator.otx_client import fetch_and_normalize_from_env
from iocevaluator.io_loader import save_threats_json


def main() -> None:
    load_dotenv()
    setup_logging()

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path, help="Where to write the normalized JSON.")
    ap.add_argument("--limit", default=None, type=int, help="Optional cap on number of pulses.")
    args = ap.parse_args()

    threats = fetch_and_normalize_from_env(limit=args.limit)
    save_threats_json(args.out, threats)
    print(f"Wrote {len(threats)} records to {args.out}")


if __name__ == "__main__":
    main()
