from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    """Configuration for the end-to-end pipeline."""

    # Data input
    input_json: Path

    # Outputs
    output_dir: Path = Path("out")

    # Feature building
    feature_dim: int = 32
    use_text_embeddings: bool = True
    text_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # MIIS aggregation
    miis_agg: str = "mean"  # mean | sum

    # Optional GCN
    run_gcn: bool = False
    gcn_hidden_dim: int = 64
    gcn_epochs: int = 50
    gcn_lr: float = 1e-3

    # OTX
    otx_api_key_env: str = "OTX_API_KEY"
    otx_limit: Optional[int] = None  # set to cap pulses fetched

    def ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
