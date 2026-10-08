"""Threat prioritisation / ranking stage (manuscript Sec. 3.6, 4.11).  No attack-path tracing here.

    A_rank  = Adj (the MeGiTS-weighted adjacency over chi_1..chi_20) with self-loops removed          (Eq. arank-def)
    A_bar   = 1{ Adj_ij >= tau }  (binarised variant, self-loops removed)                              (Eq. arank-bin)
    x       = (1/lambda) A_rank x      principal eigenvector                                           (Eq. ec-arank)
    R(v)    = alpha * EC(v) + (1 - alpha) * Severity(v)                                                (Eq. risk-fusion)

What the manuscript fixes and what it leaves open (each open point is a named, documented option):

  * alpha in [0, 1] "tuned on the validation split" and tau "chosen on the validation split": no value is stated.
    ``RankerConfig.alpha`` defaults to 0.5 (NOT a manuscript value); ``alpha=None`` tunes it on validation campaigns
    inside the evaluation harness.  ``tau=None`` = weighted A_rank; a number selects the binarised variant.
  * Severity: "CVSS for vulnerabilities and incident-impact proxies for files, devices, and platforms".  Only the CVSS
    part is defined, so Severity exists for vulnerabilities (CVSS base score / 10, in [0, 1]) and is UNAVAILABLE for
    every other entity type; no proxy is invented and no CVSS is propagated or aggregated to other nodes (the
    manuscript gives no propagation/aggregation rule).  ``missing_severity`` selects how nodes without severity are scored:
        "ec_only"  (default)  R = EC            - the fusion is applied only where Severity exists
        "exclude"             R = NaN           - such nodes are not ranked
        "zero"                Severity := 0     - an explicit imputation, NOT the default
  * EC scope.  Because every chi_k connects nodes of ONE entity type, A_rank is block-diagonal by entity type, so the
    principal eigenvector of the whole matrix is supported on a single block (here: the threat actors) and is
    exactly 0 for all other types.  ``ec_scope``: "global" (literal manuscript), "type" (default: the same eigenvector
    equation on each entity-type block - consistent with Table 18, which ranks entities per type) or "component"
    (per connected component).
  * Normalisation: EC is scaled to [0, 1] within its scope group (``ec_norm``: "max" = x / max (default) or "minmax");
    Severity = CVSS / 10 is already in [0, 1], so R lies in [0, 1].
Rankings are deterministic: descending R (rounded to 10 decimals), ties broken by entity id.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .prioritisation import eigenvector_centrality, rank_adjacency
from .tikg import TIKG, EntityType

EC_SCOPES = ("global", "type", "component")
EC_NORMS = ("max", "minmax")
MISSING_POLICIES = ("ec_only", "exclude", "zero")
ROUND_DECIMALS = 10


# ------------------------------------------------------------------------------------------------ configuration
@dataclass(frozen=True)
class RankerConfig:
    alpha: Optional[float] = 0.5             # None -> tuned on the validation split (evaluation harness only)
    tau: Optional[float] = None              # None -> weighted A_rank; number -> binarised A_bar (Adj >= tau)
    ec_scope: str = "type"
    ec_norm: str = "max"
    missing_severity: str = "ec_only"
    severity_types: Tuple[str, ...] = ("vulnerability",)     # entity types that have a severity (CVSS)
    alpha_grid: Tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(11))
    tune_k: int = 5                          # top-k used when alpha is tuned on validation cases

    def __post_init__(self):
        if self.alpha is not None and not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.ec_scope not in EC_SCOPES or self.ec_norm not in EC_NORMS or self.missing_severity not in MISSING_POLICIES:
            raise ValueError("unknown ec_scope / ec_norm / missing_severity")
        if self.tau is not None and self.tau < 0:
            raise ValueError("tau must be >= 0")


# ------------------------------------------------------------------------------------------------ A_rank and EC
def build_rank_adjacency(adj: sp.spmatrix, tau: Optional[float] = None) -> sp.csr_matrix:
    """A_rank = Adj without self-loops; with ``tau`` the binarised A_bar_ij = 1{Adj_ij >= tau}."""
    return rank_adjacency(adj, tau)


def _groups(tikg: TIKG, a: sp.csr_matrix, scope: str) -> np.ndarray:
    if scope == "global":
        return np.zeros(tikg.n, dtype=int)
    if scope == "type":
        order = {t.value: i for i, t in enumerate(EntityType)}
        return np.array([order[t] for t in tikg.types])
    return connected_components(a, directed=False)[1]


def centrality(tikg: TIKG, a_rank: sp.spmatrix, scope: str = "type", norm: str = "max") -> np.ndarray:
    """Eigenvector centrality of A_rank in [0, 1] (principal eigenvector, deterministic start vector).

    The eigenvector equation is solved on each scope group (whole matrix / entity-type block / connected component)
    and scaled within the group: "max" -> x / max x ; "minmax" -> (x - min) / (max - min).  Groups without any edge get 0."""
    a = sp.csr_matrix(a_rank, dtype=np.float64)
    g = _groups(tikg, a, scope)
    out = np.zeros(tikg.n)
    for gid in np.unique(g):
        idx = np.flatnonzero(g == gid)
        if len(idx) < 2:
            continue
        x = eigenvector_centrality(a[idx][:, idx].tocsr())          # max-normalised
        x = np.where(x < 1e-12, 0.0, x)                             # numerical zeros (eigensolver noise) -> exactly 0
        if norm == "minmax" and x.max() > x.min():
            x = (x - x.min()) / (x.max() - x.min())
        elif norm == "minmax":
            x = np.zeros_like(x)
        out[idx] = x
    return out


# ------------------------------------------------------------------------------------------------ severity / fusion
def severity_vector(tikg: TIKG, severity: Optional[np.ndarray], known: Optional[np.ndarray],
                    types: Sequence[str] = ("vulnerability",)) -> Tuple[np.ndarray, np.ndarray]:
    """Per-node Severity in [0, 1] (NaN where unavailable) and the availability mask.

    Only entity types in ``types`` can have a severity, and only where a real value was supplied (CVSS base score / 10).
    Nothing is imputed, propagated or aggregated across nodes."""
    n = tikg.n
    sev = np.full(n, np.nan)
    if severity is None or known is None:
        return sev, np.zeros(n, dtype=bool)
    severity, known = np.asarray(severity, dtype=float), np.asarray(known, dtype=bool)
    ok = known & np.isin(tikg.types, list(types))
    if ((severity[ok] < 0) | (severity[ok] > 1)).any():
        raise ValueError("severity must be CVSS / 10 in [0, 1]")
    sev[ok] = severity[ok]
    return sev, ok


def fuse(ec: np.ndarray, sev: np.ndarray, available: np.ndarray, alpha: float, missing: str = "ec_only") -> np.ndarray:
    """R = alpha * EC + (1 - alpha) * Severity where Severity is available; ``missing`` governs the other nodes."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    r = alpha * ec + (1 - alpha) * np.where(available, sev, 0.0)
    if missing == "ec_only":
        r = np.where(available, r, ec)
    elif missing == "exclude":
        r = np.where(available, r, np.nan)
    elif missing != "zero":
        raise ValueError(missing)
    return r


@dataclass
class RankScores:
    a_rank: sp.csr_matrix
    ec: np.ndarray
    severity: np.ndarray                     # NaN where unavailable
    severity_available: np.ndarray
    risk: np.ndarray                         # R (NaN for excluded nodes)
    alpha: float
    config: RankerConfig = field(default_factory=RankerConfig)


def prioritise(tikg: TIKG, adj: sp.spmatrix, severity: Optional[np.ndarray] = None, known: Optional[np.ndarray] = None,
               cfg: RankerConfig = RankerConfig(), alpha: Optional[float] = None, ec: Optional[np.ndarray] = None) -> RankScores:
    """Full prioritisation score from the MeGiTS adjacency.  ``alpha`` overrides ``cfg.alpha`` (tuned value);
    ``ec`` may be passed to reuse a centrality computed for the same adjacency."""
    a = alpha if alpha is not None else cfg.alpha
    if a is None:
        raise ValueError("alpha is None: pass a tuned alpha (the evaluation harness tunes it on validation campaigns)")
    a_rank = build_rank_adjacency(adj, cfg.tau)
    if ec is None:
        ec = centrality(tikg, a_rank, cfg.ec_scope, cfg.ec_norm)
    sev, avail = severity_vector(tikg, severity, known, cfg.severity_types)
    return RankScores(a_rank, ec, sev, avail, fuse(ec, sev, avail, a, cfg.missing_severity), float(a), cfg)


# ------------------------------------------------------------------------------------------------ rankings
def rank_nodes(tikg: TIKG, score: np.ndarray, nodes: Optional[Sequence[int]] = None, top: Optional[int] = None) -> List[int]:
    """Deterministic ranking: descending score (rounded to 10 decimals), ties by entity id; NaN scores are not ranked."""
    idx = np.arange(tikg.n) if nodes is None else np.asarray(nodes, dtype=int)
    idx = idx[~np.isnan(score[idx])]
    ids = np.array([tikg.entities[i].id for i in idx], dtype=object)
    order = sorted(range(len(idx)), key=lambda j: (-round(float(score[idx[j]]), ROUND_DECIMALS), ids[j]))
    out = [int(idx[j]) for j in order]
    return out[:top] if top else out


def campaign_rankings(tikg: TIKG, score: np.ndarray, campaigns: Optional[Sequence[str]] = None,
                      entity_types: Optional[Sequence[str]] = None, top: Optional[int] = None
                      ) -> Dict[str, List[Tuple[str, float]]]:
    """Per-campaign ranked [(entity id, score)] lists (optionally restricted to some entity types)."""
    camp = tikg.campaigns()
    names = sorted({c for c in camp if c}) if campaigns is None else list(campaigns)
    out = {}
    for c in names:
        sel = np.flatnonzero(camp == c)
        if entity_types is not None:
            sel = sel[np.isin(tikg.types[sel], list(entity_types))]
        out[c] = [(tikg.entities[i].id, float(score[i])) for i in rank_nodes(tikg, score, sel, top)]
    return out


def ranked_by_type(tikg: TIKG, score: np.ndarray, top: Optional[int] = None) -> Dict[str, List[Tuple[str, float]]]:
    """Table-18 style listing: highest-ranked entities per entity type."""
    return {t.value: [(tikg.entities[i].id, float(score[i])) for i in rank_nodes(tikg, score, tikg.type_nodes[t], top)]
            for t in EntityType}


# ------------------------------------------------------------------------------------------------ rankers for the harness
@dataclass
class RankingInputs:
    """Everything a ranker may use for one (fold, seed) run - fold-safe by construction."""
    tikg: TIKG
    fold: object                             # splits.Fold
    prob: np.ndarray                         # N x K GCN probabilities
    valid: np.ndarray
    infer_adj: sp.csr_matrix                 # MeGiTS adjacency of the inference graph (no labels)
    train_adj: sp.csr_matrix                 # MeGiTS adjacency with test nodes isolated
    severity: Optional[np.ndarray] = None
    severity_known: Optional[np.ndarray] = None
    cases: Sequence = ()                     # TopKCase list (expert-confirmed high-risk IOCs)


class ProbabilityBaselineRanker:
    """BASELINE / PLACEHOLDER (not the manuscript ranker): score = highest GCN probability among a node's valid labels."""
    name = "baseline_max_gcn_probability"

    def info(self) -> Dict:
        return {"name": self.name, "manuscript_ranker": False}

    def __call__(self, inp: RankingInputs) -> np.ndarray:
        return np.where(inp.valid, inp.prob, -np.inf).max(1)


class ThreatPrioritisationRanker:
    """The manuscript's ranker: R = alpha * EC(A_rank) + (1 - alpha) * Severity.  EC does not depend on the GCN.

    Test-fold scores use the inference adjacency (full topology, no labels) and the nodes' own CVSS values.  With
    ``cfg.alpha is None`` alpha is chosen per fold on the VALIDATION campaigns' cases only, from the training-graph
    adjacency (test nodes isolated), maximising the top-``tune_k`` hit rate (ties: alpha closest to 0.5, then smaller)."""
    name = "threat_prioritisation"

    def __init__(self, cfg: RankerConfig = RankerConfig()):
        self.cfg = cfg
        self._cache: Dict[Tuple, RankScores] = {}
        self.alpha_by_fold: Dict[int, Optional[float]] = {}

    def info(self) -> Dict:
        return {"name": self.name, "manuscript_ranker": True, **{k: (list(v) if isinstance(v, tuple) else v)
                                                                 for k, v in asdict(self.cfg).items()}}

    def _alpha(self, inp: RankingInputs) -> float:
        fi = inp.fold.index
        if fi in self.alpha_by_fold:
            return self.alpha_by_fold[fi] if self.alpha_by_fold[fi] is not None else 0.5
        if self.cfg.alpha is not None:
            self.alpha_by_fold[fi] = self.cfg.alpha
            return self.cfg.alpha
        from .metrics import topk_hit_rate
        camp = inp.tikg.campaigns()
        val_c = {camp[i] for i in inp.fold.val}
        cases = [c for c in inp.cases if c.campaign in val_c]
        best = None
        if cases:
            base = prioritise(inp.tikg, inp.train_adj, inp.severity, inp.severity_known, self.cfg, alpha=0.5)
            for a in self.cfg.alpha_grid:
                r = fuse(base.ec, base.severity, base.severity_available, a, self.cfg.missing_severity)
                hits = []
                for c in cases:
                    ranked = [inp.tikg.entities[i].id for i in rank_nodes(inp.tikg, r, np.flatnonzero(camp == c.campaign))]
                    hits.append((ranked, c.confirmed))
                s = topk_hit_rate(hits, self.cfg.tune_k)
                key = (-s, abs(a - 0.5), a)
                best = key if best is None or key < best else best
        alpha = best[2] if best else None
        self.alpha_by_fold[fi] = alpha
        return alpha if alpha is not None else 0.5

    def scores(self, inp: RankingInputs) -> RankScores:
        alpha = self._alpha(inp)
        key = (inp.fold.index, alpha)
        if key not in self._cache:
            self._cache[key] = prioritise(inp.tikg, inp.infer_adj, inp.severity, inp.severity_known, self.cfg, alpha=alpha)
        return self._cache[key]

    def __call__(self, inp: RankingInputs) -> np.ndarray:
        return self.scores(inp).risk
