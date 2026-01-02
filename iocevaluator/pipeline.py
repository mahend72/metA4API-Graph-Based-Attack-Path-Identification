from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .event_builder import build_events
from .features import build_attacker_feature_matrix
from .io_loader import load_threats_json, save_threats_json
from .mpgiis import aggregate_miis
from .relations import build_relations, commuting_matrices

log = logging.getLogger(__name__)


def run_pipeline(cfg: PipelineConfig) -> dict:
    """End-to-end pipeline.

    Inputs:
      - cfg.input_json: normalized JSON (or exported from OTX via scripts)

    Outputs (written under cfg.output_dir):
      - threats_normalized.json
      - attackers.csv
      - features.npy, feature_names.json
      - miis.npy
      - optional gcn_embeddings.npy, gcn_recon.npy, gcn_loss.csv
    """
    cfg.ensure_output_dir()

    threats = load_threats_json(cfg.input_json)
    save_threats_json(cfg.output_dir / "threats_normalized.json", threats)

    events = build_events(threats)
    rel = build_relations(events)
    commuting = commuting_matrices(rel)

    miis = aggregate_miis(commuting, agg=cfg.miis_agg)
    np.save(cfg.output_dir / "miis.npy", miis)

    features = build_attacker_feature_matrix(
        threats=threats,
        attackers=rel.attackers,
        feature_dim=cfg.feature_dim,
        use_text_embeddings=cfg.use_text_embeddings,
        text_embedding_model=cfg.text_embedding_model,
    )
    np.save(cfg.output_dir / "features.npy", features.X)
    (cfg.output_dir / "feature_names.json").write_text(json.dumps(features.feature_names, indent=2), encoding="utf-8")
    pd.DataFrame({"attacker": rel.attackers}).to_csv(cfg.output_dir / "attackers.csv", index=False)

    results = {
        "attackers": rel.attackers,
        "miis_path": str(cfg.output_dir / "miis.npy"),
        "features_path": str(cfg.output_dir / "features.npy"),
    }

    if cfg.run_gcn:
        from .gcn import run_gcn_autoencoder

        g = run_gcn_autoencoder(
            adj=miis,
            X=features.X,
            hidden_dim=cfg.gcn_hidden_dim,
            epochs=cfg.gcn_epochs,
            lr=cfg.gcn_lr,
        )
        np.save(cfg.output_dir / "gcn_embeddings.npy", g.Z)
        np.save(cfg.output_dir / "gcn_recon.npy", g.recon)
        pd.DataFrame({"epoch": list(range(1, len(g.loss_history) + 1)), "loss": g.loss_history}).to_csv(
            cfg.output_dir / "gcn_loss.csv", index=False
        )
        results.update(
            {
                "gcn_embeddings_path": str(cfg.output_dir / "gcn_embeddings.npy"),
                "gcn_recon_path": str(cfg.output_dir / "gcn_recon.npy"),
            }
        )

    log.info("Pipeline complete. Outputs in: %s", cfg.output_dir)
    return results
