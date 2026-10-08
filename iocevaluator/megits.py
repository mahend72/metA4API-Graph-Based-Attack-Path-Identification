"""Weighted MeGiTS similarity and leakage-safe per-fold semantic adjacency - manuscript Sec. 3.4.

    MeGiTS(vi, vj) = sum_k w_k * 2 NumP_k(vi,vj) / (NumP_k(vi,vi) + NumP_k(vj,vj))          (Eq. 1)

NumP_k = commuting matrix C_k (Eq. 2-3).  Terms with a zero denominator are set to 0.  Uniform w_k = 1/20 is the
default (Sec. 3.4); weights must be >= 0 and sum to 1.  For ablations on a subset of structures the uniform weights
are renormalised over that subset.  The default structure set is chi_1..chi_20
(all 20 structures of Fig. 3), so the default weight is 1/20 (see metagraphs.py).

Per-fold leakage control (Sec. 4.2).  ``fold_megits_adjacency`` builds two graphs:
  * the *training graph*: all triplets that do not touch an outer-test node (or any node of a test campaign);
  * the *inference graph*: the full graph (test nodes are introduced only at inference time).
Entries between two non-test nodes are taken from the training graph only, so the adjacency among training /
validation nodes never depends on test-node structure; any entry involving a test node comes from the inference
graph.  (No labels are used anywhere in the adjacency.)
"""
from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from .metagraphs import StructureSpec, commuting_matrices, default_structures, global_matrix
from .tikg import TIKG


# Realisation recorder: every adjacency built inside ``record_realisation()`` appends the structure ids and weights it ACTUALLY
# used (not the configured defaults), so the harness can verify per fold that chi_1..chi_20 were used with weight 1/20 each.
_RECORDER: "contextvars.ContextVar[Optional[List[dict]]]" = contextvars.ContextVar("iocevaluator_megits_recorder", default=None)


@contextlib.contextmanager
def record_realisation() -> Iterator[List[dict]]:
    records: List[dict] = []
    token = _RECORDER.set(records)
    try:
        yield records
    finally:
        _RECORDER.reset(token)


def normalise_weights(structures: Sequence[StructureSpec], weights: Optional[Dict[int, float]] = None) -> Dict[int, float]:
    if weights is None:
        return {s.id: 1.0 / len(structures) for s in structures}
    w = {s.id: float(weights.get(s.id, 0.0)) for s in structures}
    if any(v < 0 for v in w.values()):
        raise ValueError("MeGiTS weights must be non-negative")
    tot = sum(w.values())
    if not np.isclose(tot, 1.0, atol=1e-6):
        raise ValueError(f"MeGiTS weights must sum to 1 (got {tot:.6f})")
    return w


def structure_similarity(C: sp.spmatrix) -> sp.csr_matrix:
    """Per-structure term 2*C_ij / (C_ii + C_jj); zero where the denominator is zero."""
    C = sp.csr_matrix(C, dtype=np.float64)
    d = C.diagonal()
    coo = C.tocoo()
    denom = d[coo.row] + d[coo.col]
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.where(denom > 0, 2.0 * coo.data / denom, 0.0)
    out = sp.csr_matrix((val, (coo.row, coo.col)), shape=C.shape)
    out.eliminate_zeros()
    return out


def _embed(tikg: TIKG, spec: StructureSpec, M: sp.spmatrix) -> sp.csr_matrix:
    return global_matrix(tikg, spec, M)


@dataclass
class MeGiTSResult:
    adj: sp.csr_matrix                       # N x N, symmetric, zero diagonal (GCN adds I)
    weights: Dict[int, float]
    components: Dict[int, sp.csr_matrix] = field(default_factory=dict)   # w_k * S_k (global), if requested
    instance_counts: Dict[int, int] = field(default_factory=dict)        # off-diagonal non-zeros of C_k
    self_instances: Dict[int, np.ndarray] = field(default_factory=dict)  # NumP_k(v,v) per node (global)


def megits_adjacency(tikg: TIKG, structures: Optional[Sequence[StructureSpec]] = None,
                     weights: Optional[Dict[int, float]] = None, exclude: Optional[np.ndarray] = None,
                     binary: bool = False, keep_components: bool = False,
                     commuting: Optional[Dict[int, sp.csr_matrix]] = None) -> MeGiTSResult:
    """``commuting``: optional pre-computed ``commuting_matrices(tikg, structures, exclude)`` (same arguments); the output is
    identical, it only lets the cost of the chi construction and of the similarity assembly be separated / reused."""
    structures = list(structures or default_structures())
    w = normalise_weights(structures, weights)
    rec = _RECORDER.get()
    if rec is not None:
        rec.append({"structure_ids": [s.id for s in structures], "weights": {f"chi{int(k)}": float(v) for k, v in w.items()},   # str keys: stable under a JSON round trip
                    "binary": bool(binary), "excluded_nodes": 0 if exclude is None else int(np.asarray(exclude).sum())})
    C = commuting if commuting is not None else commuting_matrices(tikg, structures, exclude=exclude)
    total = sp.csr_matrix((tikg.n, tikg.n), dtype=np.float64)
    comps, counts, selfi = {}, {}, {}
    for s in structures:
        Ck = C[s.id]
        off = Ck - sp.diags(Ck.diagonal())
        off.eliminate_zeros()
        counts[s.id] = int(off.nnz)
        sd = np.zeros(tikg.n)
        sd[tikg.type_nodes[s.node_type]] = Ck.diagonal()
        selfi[s.id] = sd
        if binary:
            term = _embed(tikg, s, (off > 0).astype(np.float64))
        else:
            S = structure_similarity(Ck)
            S = S - sp.diags(S.diagonal())          # drop self-similarity; GCN adds I
            S.eliminate_zeros()
            term = _embed(tikg, s, S)
        if keep_components:
            comps[s.id] = (term * w[s.id]).tocsr()
        total = total + term * w[s.id]
    if binary:
        total = (total > 0).astype(np.float64)
    total = total.tocsr()
    total.setdiag(0)
    total.eliminate_zeros()
    return MeGiTSResult(adj=total.astype(np.float32), weights=w, components=comps, instance_counts=counts,
                        self_instances=selfi)


def fold_megits_adjacency(tikg: TIKG, test_nodes: np.ndarray, structures: Optional[Sequence[StructureSpec]] = None,
                          weights: Optional[Dict[int, float]] = None, binary: bool = False) -> MeGiTSResult:
    """Leakage-safe fold adjacency (see module docstring). ``test_nodes``: boolean mask over all nodes."""
    test_nodes = np.asarray(test_nodes, dtype=bool)
    full = megits_adjacency(tikg, structures, weights, exclude=None, binary=binary)
    if not test_nodes.any():
        return full
    train = megits_adjacency(tikg, structures, weights, exclude=test_nodes, binary=binary)
    coo = full.adj.tocoo()
    keep = test_nodes[coo.row] | test_nodes[coo.col]          # entries involving a test node: inference graph
    mixed = sp.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=full.adj.shape)
    adj = (mixed + train.adj).tocsr()
    adj.eliminate_zeros()
    return MeGiTSResult(adj=adj, weights=full.weights, instance_counts=full.instance_counts,
                        self_instances=full.self_instances)
