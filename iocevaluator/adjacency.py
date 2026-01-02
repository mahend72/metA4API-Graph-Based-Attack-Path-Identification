from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .schemas import Event

log = logging.getLogger(__name__)


def build_node_index(events: Iterable[Event]) -> Dict[str, int]:
    """Assign a stable integer index for every unique node across events."""
    nodes = []
    seen = set()
    for e in events:
        for n in e.all_nodes():
            if n not in seen:
                seen.add(n)
                nodes.append(n)
    return {n: i for i, n in enumerate(nodes)}


def build_cooccurrence_adjacency(events: List[Event], add_self_loops: bool = True) -> tuple[np.ndarray, Dict[str, int]]:
    """Build a symmetric adjacency matrix based on co-occurrence in events."""
    idx = build_node_index(events)
    n = len(idx)
    adj = np.zeros((n, n), dtype=np.float32)

    for e in events:
        nodes = [idx[n] for n in e.all_nodes() if n in idx]
        # connect all pairs in this event
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                adj[a, b] = 1.0
                adj[b, a] = 1.0

    if add_self_loops:
        np.fill_diagonal(adj, 1.0)

    log.info("Adjacency built: %d nodes, density %.6f", n, float(adj.sum() / (n * n)))
    return adj, idx
