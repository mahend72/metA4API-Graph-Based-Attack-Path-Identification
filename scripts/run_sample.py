"""Smoke test: run the pipeline on the included sample data."""

from pathlib import Path

from iocevaluator.config import PipelineConfig
from iocevaluator.logging_utils import setup_logging
from iocevaluator.pipeline import run_pipeline


if __name__ == "__main__":
    setup_logging()
    cfg = PipelineConfig(
        input_json=Path("data/sample_threats.json"),
        output_dir=Path("out_sample"),
        feature_dim=16,
        use_text_embeddings=False,  # keep it lightweight
        run_gcn=False,
    )
    run_pipeline(cfg)
    print("Done. See out_sample/")
