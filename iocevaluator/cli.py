from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig
from .logging_utils import setup_logging
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="iocevaluator", description="IOC Evaluation (MPGIIS/MIIS + optional GCN).")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run end-to-end pipeline on a normalized JSON file.")
    run.add_argument("--input", required=True, type=Path, help="Path to normalized IOC JSON (list of threat records).")
    run.add_argument("--out", default=Path("out"), type=Path, help="Output directory.")
    run.add_argument("--feature-dim", default=32, type=int, help="Final attacker feature dimension (after SVD).")
    run.add_argument("--no-text-emb", action="store_true", help="Disable sentence-transformer embeddings.")
    run.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model.")
    run.add_argument("--miis-agg", default="mean", choices=["mean", "sum"], help="Aggregate MIIS across meta-path types.")
    run.add_argument("--gcn", action="store_true", help="Run the optional GCN autoencoder.")
    run.add_argument("--gcn-epochs", default=50, type=int)
    run.add_argument("--gcn-hidden", default=64, type=int)
    run.add_argument("--gcn-lr", default=1e-3, type=float)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging()

    if args.cmd == "run":
        cfg = PipelineConfig(
            input_json=args.input,
            output_dir=args.out,
            feature_dim=args.feature_dim,
            use_text_embeddings=not args.no_text_emb,
            text_embedding_model=args.text_model,
            miis_agg=args.miis_agg,
            run_gcn=args.gcn,
            gcn_epochs=args.gcn_epochs,
            gcn_hidden_dim=args.gcn_hidden,
            gcn_lr=args.gcn_lr,
        )
        run_pipeline(cfg)


if __name__ == "__main__":
    main()
