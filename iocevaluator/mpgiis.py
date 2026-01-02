from __future__ import annotations

import logging
from typing import Dict, Literal

import numpy as np

log = logging.getLogger(__name__)


def dice_similarity_from_commuting(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Dice coefficient between rows using a commuting/count matrix.

    In the notebook, MIIS for a meta-path k uses:
        miis_ij = (2 * num_p_ij) / (num_p_ii + num_p_jj)

    where num_p_ij is the count of meta-paths between i and j.
    Here, M plays the role of the meta-path count matrix.
    """
    M = M.astype(np.float32, copy=False)
    diag = np.diag(M)
    denom = diag[:, None] + diag[None, :] + eps
    return (2.0 * M) / denom


def aggregate_miis(commuting: Dict[str, np.ndarray], agg: Literal["mean", "sum"] = "mean") -> np.ndarray:
    """Aggregate MIIS matrices across meta-path types."""
    miis_list = [dice_similarity_from_commuting(M) for M in commuting.values()]
    stack = np.stack(miis_list, axis=0)  # [K, N, N]
    if agg == "sum":
        out = stack.sum(axis=0)
    else:
        out = stack.mean(axis=0)
    # Ensure self-loop
    np.fill_diagonal(out, 1.0)
    log.info("MIIS aggregated across %d meta-path types using %s", len(miis_list), agg)
    return out
