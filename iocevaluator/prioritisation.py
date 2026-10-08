"""Threat prioritisation (manuscript Sec. 3.6): eigenvector centrality over the MeGiTS ranking adjacency,
optional binarised variant, CVSS/impact-severity fusion R = alpha*EC + (1-alpha)*Severity, and analyst-facing
explanations (Sec. 4.12 / Table 19).

Notes
 * EC is the principal eigenvector of A_rank (Eq. in Sec. 3.6).  On a disconnected graph the Perron vector
   concentrates on the dominant component (other components get ~0); ``per_component=True`` computes EC per
   connected component instead (extension, off by default = manuscript behaviour).
 * Scores are max-normalised to [0, 1].
 * Severity is never imputed: nodes without a supplied severity use EC only (R = EC) and are flagged.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

from .metagraphs import STRUCTURES, get_structure
from .tikg import TIKG


def rank_adjacency(adj: sp.spmatrix, tau: Optional[float] = None) -> sp.csr_matrix:
    """A_rank = MeGiTS adjacency with self-loops removed; if `tau` is given the binarised variant 1{Adj >= tau}."""
    a = sp.csr_matrix(adj, dtype=np.float64)
    a.setdiag(0)
    a.eliminate_zeros()
    if tau is not None:
        a = (a >= tau).astype(np.float64)
        a.eliminate_zeros()
    return a.tocsr()


def _ec_block(a: sp.csr_matrix, max_iter: int, tol: float) -> np.ndarray:
    n = a.shape[0]
    if a.nnz == 0:
        return np.zeros(n)
    symmetric = abs(a - a.T).sum() < 1e-9
    if symmetric and n > 2:
        try:
            _, v = eigsh(a, k=1, which="LA", tol=1e-10, maxiter=max_iter * 10, v0=np.ones(n) / np.sqrt(n))   # fixed start -> deterministic
            x = np.abs(v[:, 0])
            return x / x.max() if x.max() > 0 else x
        except Exception:
            pass
    x = np.ones(n) / np.sqrt(n)                              # shifted power iteration: (A + I) avoids oscillation
    for _ in range(max_iter):
        y = a @ x + x
        y /= np.linalg.norm(y)
        if np.linalg.norm(y - x) < tol:
            x = y
            break
        x = y
    return x / x.max() if x.max() > 0 else x


def eigenvector_centrality(a: sp.spmatrix, max_iter: int = 1000, tol: float = 1e-10,
                           per_component: bool = False) -> np.ndarray:
    a = sp.csr_matrix(a, dtype=np.float64)
    if not per_component:
        return _ec_block(a, max_iter, tol)
    k, lab = connected_components(a, directed=False)
    out = np.zeros(a.shape[0])
    for c in range(k):
        idx = np.flatnonzero(lab == c)
        if len(idx) > 1:
            out[idx] = _ec_block(a[idx][:, idx].tocsr(), max_iter, tol)
    return out


def timed_centrality(a: sp.spmatrix, **kw) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    ec = eigenvector_centrality(a, **kw)
    return ec, time.perf_counter() - t0


def fused_risk(ec: np.ndarray, severity: Optional[np.ndarray], known: Optional[np.ndarray], alpha: float) -> np.ndarray:
    """R = alpha*EC + (1-alpha)*Severity where severity is known, else R = EC (no imputation)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    if severity is None or known is None:
        return ec.copy()
    return np.where(known, alpha * ec + (1 - alpha) * severity, ec)


def tune_alpha(ec, severity, known, validation_metric: Callable[[np.ndarray], float],
               grid: Sequence[float] = tuple(np.linspace(0, 1, 11))) -> float:
    """alpha in [0,1] tuned on the validation split. `validation_metric(R) -> score` (e.g. top-k hit rate against
    expert-confirmed high-risk IOCs) must be supplied: the manuscript gives no ground truth we could invent."""
    return float(max(grid, key=lambda a: validation_metric(fused_risk(ec, severity, known, a))))


def select_tau(adj: sp.spmatrix, validation_metric: Callable[[np.ndarray], float],
               candidates: Optional[Sequence[float]] = None) -> float:
    """Threshold tau of the binarised A_rank chosen on the validation split."""
    a = rank_adjacency(adj)
    if candidates is None:
        v = a.data
        candidates = np.unique(np.quantile(v, [0.0, 0.25, 0.5, 0.75])) if len(v) else [0.0]
    return float(max(candidates, key=lambda t: validation_metric(eigenvector_centrality(rank_adjacency(adj, t)))))


# --------------------------------------------------------------------------------------- explanations (Table 19)
_ACTIONS = {
    "vulnerability": "Prioritise patching; inspect builds on affected devices; gate new jobs on vulnerable devices.",
    "device": "Isolate or inspect the device; review recent build jobs and firmware integrity.",
    "platform": "Review platform/firmware versions across devices sharing this platform.",
    "file": "Quarantine and inspect the artefact; review supplier-provided files linked to the same path.",
    "threat_actor": "Review access logs of devices linked to this actor; update detections for its infrastructure.",
    "attack_method": "Add or tune detections for this technique across production-network telemetry.",
    "attack_type": "Review incident-response playbook for this outcome category.",
}


def explain_ioc(tikg: TIKG, node: int, components: Dict[int, sp.csr_matrix], adj: sp.csr_matrix, ec: np.ndarray,
                severity: Optional[np.ndarray] = None, known: Optional[np.ndarray] = None, risk: Optional[np.ndarray] = None,
                top_neighbours: int = 5) -> Dict:
    """Four-element explanation: contributing structures, strongest neighbours, EC/severity components, action.
    `components` must come from ``megits_adjacency(..., keep_components=True)``. The action text is a static
    per-entity-type template (not learned)."""
    contrib = {k: float(c[node].sum()) for k, c in components.items() if c[node].nnz}
    contrib = dict(sorted(contrib.items(), key=lambda kv: -kv[1]))
    row = adj[node].tocoo()
    order = np.argsort(-row.data)[:top_neighbours]
    nbrs = [(tikg.entities[row.col[i]].id, float(row.data[i])) for i in order]
    e = tikg.entities[node]
    return {
        "ioc": e.id, "type": e.type.value,
        "contributing_structures": [{"structure": get_structure(k).name, "schema": get_structure(k).schema,
                                     "weighted_similarity_mass": v} for k, v in contrib.items()],
        "top_neighbours": nbrs,
        "eigenvector_centrality": float(ec[node]),
        "severity": None if severity is None or known is None or not known[node] else float(severity[node]),
        "fused_risk": None if risk is None else float(risk[node]),
        "recommended_action": _ACTIONS.get(e.type.value, ""),
    }
