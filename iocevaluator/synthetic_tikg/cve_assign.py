"""Deterministic assignment of real CVEs to synthetic campaigns.

CVEs are embedded from their own public metadata (description + vendor/product + CWE + severity) with TF-IDF -> SVD,
clustered hierarchically (families, then the campaigns of a family) and assigned under the exact per-campaign
capacities the generator needs.  Family clusters are matched to the latent behaviour families by similarity between
the cluster centroid and the family's theme vocabulary, so related campaigns draw from the same region of CVE space
while unrelated families draw from different regions.  No actor information is involved.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from . import vocab as V


def _doc(r: dict) -> str:
    toks = [r["description"]] + [f"vendor_{v}" for v in r["vendors"][:3]] + [f"prod_{p.split(':')[-1]}" for p in r["products"][:3]]
    toks += [c.replace("-", "_").lower() for c in r["cwe"]] + [f"sev_{(r['severity'] or 'none').lower()}"]
    return " ".join(toks)


def _family_text(fd: dict) -> str:
    return " ".join(fd["weakness"] + fd["tradecraft"] + [V.DEVICE_KIND_LABEL[k] for k in fd["devices"]]
                    + [V.PLATFORM_KIND_LABEL[k] for k in fd["platforms"]])


def _greedy(dist: np.ndarray, capacity: Sequence[int]) -> np.ndarray:
    """dist: (n_items, n_bins).  Assign every item to a bin without exceeding capacity (closest pairs first)."""
    n, k = dist.shape
    order = np.argsort(dist, axis=None, kind="stable")
    cap, out = list(capacity), np.full(n, -1)
    for flat in order:
        i, j = divmod(int(flat), k)
        if out[i] < 0 and cap[j] > 0:
            out[i] = j
            cap[j] -= 1
    assert (out >= 0).all(), "capacities must sum to the number of items"
    return out


def assign_cves(records: List[dict], alloc: Sequence[int], camp_family: Sequence[int], fam_defs: Sequence[dict],
                seed: int) -> List[List[dict]]:
    """Return, for every campaign index, its list of CVE records (len == alloc[c])."""
    n, F = len(records), len(fam_defs)
    if sum(alloc) != n:
        raise ValueError("allocation must equal number of CVEs")
    tf = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2 if n > 50 else 1, max_features=4000,
                         token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z_0-9]+\b")
    M = tf.fit_transform([_doc(r) for r in records])
    dim = max(2, min(32, M.shape[1] - 1, n - 1))
    svd = TruncatedSVD(dim, random_state=seed, n_iter=7).fit(M)
    X = normalize(svd.transform(M))
    P = normalize(svd.transform(tf.transform([_family_text(f) for f in fam_defs])))

    km = KMeans(F, n_init=3, random_state=seed, algorithm="lloyd").fit(X)
    C = normalize(km.cluster_centers_)
    r_idx, c_idx = linear_sum_assignment(-(P @ C.T))              # family f <- cluster c_idx[f]
    fam_cluster = dict(zip(r_idx.tolist(), c_idx.tolist()))
    fam_cap = [sum(a for a, f in zip(alloc, camp_family) if f == fi) for fi in range(F)]
    dist = np.stack([1.0 - X @ C[fam_cluster[f]] for f in range(F)], axis=1)
    fam_of = _greedy(dist, fam_cap)

    out: List[List[dict]] = [[] for _ in alloc]
    for f in range(F):
        members = np.flatnonzero(fam_of == f)
        camps = [c for c, cf in enumerate(camp_family) if cf == f]
        if len(camps) == 1:
            groups = {camps[0]: members}
        else:
            sub = KMeans(len(camps), n_init=3, random_state=seed + 1 + f, algorithm="lloyd").fit(X[members])
            sizes = np.bincount(sub.labels_, minlength=len(camps))
            cl_order = np.argsort(-sizes, kind="stable")                       # big clusters <-> big campaigns
            camp_order = sorted(camps, key=lambda c: (-alloc[c], c))
            cen = normalize(sub.cluster_centers_)
            d = np.stack([1.0 - X[members] @ cen[cl_order[j]] for j in range(len(camps))], axis=1)
            a = _greedy(d, [alloc[c] for c in camp_order])
            groups = {camp_order[j]: members[a == j] for j in range(len(camps))}
        for c, idx in groups.items():
            out[c] = sorted((records[int(i)] for i in idx), key=lambda r: r["cve_id"])
    return out
