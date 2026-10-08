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

    b = sub.add_parser("build-tikg", help="Normalised OTX JSON -> TIKG JSON (7 entity types, typed relations).")
    b.add_argument("--input", required=True, type=Path)
    b.add_argument("--out", required=True, type=Path)
    b.add_argument("--extra", type=Path, help="TIKG-format JSON with entities/triplets OTX lacks (devices, platforms, ...).")

    sub.add_parser("experiment", add_help=False,
                   help="GUARDED final / manuscript-scale experiment: frozen-protocol preflight, post-run integrity check "
                        "(run `iocevaluator experiment --help`). The only CLI route for a final run.")

    e = sub.add_parser("evaluate", help="LEGACY development harness (smoke runs only; refuses final / manuscript-scale runs - "
                                        "use `experiment`). Campaign-isolated K-fold x seeds evaluation, ablations and baselines.")
    e.add_argument("--tikg", required=True, type=Path)
    e.add_argument("--labels", required=True, type=Path)
    e.add_argument("--out", required=True, type=Path)
    e.add_argument("--variants", nargs="+", default=["main"], help="groups: main ablation depth baselines per-structure all; or names")
    e.add_argument("--folds", default=10, type=int)
    e.add_argument("--seeds", default=5, type=int)
    e.add_argument("--max-folds", type=int, default=None)
    e.add_argument("--label-fraction", default=1.0, type=float)
    e.add_argument("--exclude-fields", nargs="*", default=[], help="attrs fields that define labels (leakage control)")
    e.add_argument("--text-backend", default="none", choices=["none", "tfidf", "sbert"])
    e.add_argument("--allow-ungrouped", action="store_true")

    r = sub.add_parser("rank", help="Eigenvector-centrality / CVSS-fused ranking and attack-path tracing.")
    r.add_argument("--tikg", required=True, type=Path)
    r.add_argument("--labels", required=True, type=Path)
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--oof-probs", type=Path, help="oof_probs_metA4API.npy from `evaluate` (required for honest path evaluation)")
    r.add_argument("--severity", type=Path, help="JSON {entity_id: CVSS or impact score}; never imputed")
    r.add_argument("--alpha", default=0.5, type=float)
    r.add_argument("--tau", type=float, default=None, help="binarise A_rank at tau")
    r.add_argument("--reference-paths", type=Path)
    return p


def main(argv: list[str] | None = None) -> None:
    import sys
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "experiment":                       # the guarded launcher parses its own (preflight) arguments
        from .launch import main as launch_main
        raise SystemExit(launch_main(raw[1:]))
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
    elif args.cmd == "build-tikg":
        from .workflows import build_tikg
        g = build_tikg(args.input, args.out, args.extra)
        print(g.stats())
    elif args.cmd == "evaluate":
        from .experiments import ExperimentConfig
        from .tikg_features import FeatureConfig
        from .workflows import evaluate
        cfg = ExperimentConfig(n_splits=args.folds, seeds=tuple(range(args.seeds)), max_folds=args.max_folds,
                               label_fraction=args.label_fraction, allow_ungrouped=args.allow_ungrouped,
                               features=FeatureConfig(exclude_fields=tuple(args.exclude_fields), text_backend=args.text_backend))
        evaluate(args.tikg, args.labels, args.out, args.variants, cfg)
    elif args.cmd == "rank":
        from .workflows import rank
        print(rank(args.tikg, args.labels, args.out, oof_probs=args.oof_probs, severity_path=args.severity,
                   alpha=args.alpha, tau=args.tau, reference_paths=args.reference_paths))


if __name__ == "__main__":
    main()
