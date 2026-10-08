"""Evaluation metrics (manuscript Sec. 4.5): Macro/Micro-F1 (+P/R), ROC-AUC, PR-AUC, top-k IOC hit rate,
Exact-Path Hit@k, Edge-F1, mean +- std aggregation and paired Wilcoxon tests.

Conventions the manuscript leaves open (documented, not silently chosen):
  * Predictions are thresholded at 0.5 per label (independent sigmoids); a node may receive no label.
  * Only (node, label) pairs that are *valid* for the node's entity type are scored (Table 7: per-type #Class).
  * Macro averages run over labels that have >=1 positive among the scored nodes ("supported" labels); set
    ``macro_over="all"`` to include unsupported labels as F1 = 0.
  * ROC-AUC / PR-AUC are macro-averaged over labels containing both classes.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


def multilabel_prf(Y: np.ndarray, pred: np.ndarray, valid: np.ndarray, idx: np.ndarray,
                   columns: Optional[np.ndarray] = None, macro_over: str = "supported") -> Dict[str, float]:
    idx = np.asarray(idx)
    cols = np.arange(Y.shape[1]) if columns is None else np.asarray(columns)
    y = Y[np.ix_(idx, cols)] > 0
    p = pred[np.ix_(idx, cols)] > 0
    v = valid[np.ix_(idx, cols)]
    y, p = y & v, p & v
    tp = (y & p).sum(0).astype(float)
    fp = (~y & p & v).sum(0).astype(float)
    fn = (y & ~p).sum(0).astype(float)
    per = np.array([_prf(a, b, c) for a, b, c in zip(tp, fp, fn)]).reshape(-1, 3)
    support = y.sum(0) > 0
    sel = support if macro_over == "supported" else np.ones_like(support)
    if sel.any():
        mp, mr = per[sel, 0].mean(), per[sel, 1].mean()
        mf = per[sel, 2].mean()
    else:
        mp = mr = mf = 0.0
    up, ur, uf = _prf(tp.sum(), fp.sum(), fn.sum())
    return {"macro_precision": float(mp), "macro_recall": float(mr), "macro_f1": float(mf),
            "micro_precision": float(up), "micro_recall": float(ur), "micro_f1": float(uf),
            "n_labels_scored": int(sel.sum())}


def auc_scores(Y: np.ndarray, prob: np.ndarray, valid: np.ndarray, idx: np.ndarray) -> Dict[str, float]:
    roc, pr = [], []
    for k in range(Y.shape[1]):
        m = valid[idx, k]
        y = Y[idx, k][m] > 0
        if m.sum() == 0 or y.all() or not y.any():
            continue
        s = prob[idx, k][m]
        roc.append(roc_auc_score(y, s))
        pr.append(average_precision_score(y, s))
    return {"roc_auc": float(np.mean(roc)) if roc else float("nan"),
            "pr_auc": float(np.mean(pr)) if pr else float("nan")}


def per_type_prf(Y, pred, valid, idx, type_columns: Dict[str, np.ndarray], node_types: np.ndarray) -> Dict[str, Dict]:
    """Table-14 style breakdown: metrics restricted to nodes of one entity type and that type's labels."""
    out = {}
    for t, cols in type_columns.items():
        sub = np.asarray(idx)[node_types[np.asarray(idx)] == t]
        if len(cols) and len(sub):
            out[t] = multilabel_prf(Y, pred, valid, sub, columns=cols)
    return out


def topk_hit_rate(cases: Sequence[Tuple[Sequence[str], Iterable[str]]], k: int) -> float:
    """Top-k hit rate = (1/Q) sum_q 1{R_q^+ intersect R_{q,k} != empty}; cases = [(ranked_ids, confirmed_set)]."""
    if not cases:
        return float("nan")
    hits = [1.0 if set(pos) & set(ranked[:k]) else 0.0 for ranked, pos in cases]
    return float(np.mean(hits))


def _edges(nodes: Sequence[str], rels: Optional[Sequence[str]]) -> set:
    e = set()
    for i in range(len(nodes) - 1):
        key = frozenset((nodes[i], nodes[i + 1]))
        e.add((key, rels[i]) if rels else (key, None))
    return e


def exact_path_hit_at_k(ranked: Sequence[Sequence[str]], reference: Sequence[str], k: int = 5) -> bool:
    ref = list(reference)
    return any(list(p) == ref or list(p) == ref[::-1] for p in ranked[:k])


def edge_f1(pred_nodes: Optional[Sequence[str]], pred_rels: Optional[Sequence[str]],
            ref_nodes: Sequence[str], ref_rels: Optional[Sequence[str]] = None) -> float:
    """F1 between the edge sets of the top-ranked predicted path and the reference path.
    Relation names are compared only if the reference supplies them."""
    if not pred_nodes or len(pred_nodes) < 2:
        return 0.0
    ref = _edges(ref_nodes, ref_rels or None)
    pr = _edges(pred_nodes, pred_rels) if ref_rels else {(k, None) for k, _ in _edges(pred_nodes, None)}
    tp = len(ref & pr)
    p = tp / len(pr) if pr else 0.0
    r = tp / len(ref) if ref else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray([v for v in values if v == v], dtype=float)
    return (float(a.mean()), float(a.std(ddof=0))) if len(a) else (float("nan"), float("nan"))


def paired_wilcoxon(a: Sequence[float], b: Sequence[float], alpha: float = 0.05) -> Dict[str, float]:
    """Paired Wilcoxon signed-rank test over per-fold scores."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b) or len(a) < 2 or np.allclose(a, b):
        return {"statistic": float("nan"), "p_value": float("nan"), "significant": False}
    st, p = wilcoxon(a, b)
    return {"statistic": float(st), "p_value": float(p), "significant": bool(p < alpha)}


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error control).  NaN p-values are left as NaN and are not
    counted in the family size m.  adj_(i) = max_{j <= i} min(1, (m - j + 1) * p_(j)) over the ascending order."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    m = len(ok)
    if m == 0:
        return out
    order = ok[np.argsort(p[ok], kind="stable")]
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        out[idx] = running
    return out
