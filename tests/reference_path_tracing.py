"""FROZEN COPY of iocevaluator/path_tracing.py before the scalability optimisation (equivalence oracle; do not edit).

Attack-path tracing and path-level evaluation (manuscript Algorithm 2, Sec. 3.5 "GCN Model", Sec. 4.6 path-level
validation).

Algorithm 2 (the steps this module implements):
    select high-confidence predicted nodes  ->  trace candidate paths conforming to the chi_1..chi_20 set
    ->  rank candidate paths using MeGiTS weights and eigenvector-centrality (prioritisation) scores.

1. SEEDS (start nodes).  Within a case scope (the entities of the case's campaign) a node is a seed iff its GCN
   confidence (highest predicted probability among the labels valid for its entity type) is >= ``conf_threshold``.
   Optionally only the ``priority_top_n`` seeds with the highest prioritisation score R (``ranking.py``) are kept.
   Candidate paths connect two DIFFERENT seeds ("paths that connect predicted high-risk nodes").
2. TRACING.  Depth-first search over the OBSERVED TIKG triplets.  Edge direction and relation are preserved: every hop is
   a typed step (from_type, relation, to_type, 'fwd' | 'rev'), where 'fwd' follows the stored triplet <s, r, t> and 'rev'
   traverses it backwards.  A path is schema-valid when every hop is such a step of a chi-structure template
   (``support="step"``, default; because chi_1..chi_11 contain all of T1..T11 this equals Table-2 triplet validity),
   or, stricter, when the whole hop sequence is a contiguous segment of ONE chi template or its reverse
   (``support="template"``).  Paths are simple (no node repeated => no cycles), have ``min_edges..max_edges`` edges, and
   the search is bounded per seed (``max_paths_per_seed``, ``max_expansions_per_seed``) and per case (``max_candidates``).
   Neighbours are visited in a fixed order (entity id, relation), so the output is deterministic.
3. ORIENTATION.  Each undirected path is emitted once, read from its higher-priority endpoint (larger R; ties: smaller
   entity id) to the other one.  Exact matching is order-sensitive, so this fixed reading rule matters; a looser,
   orientation-agnostic match is reported as a separate diagnostic.
4. SCORE (the manuscript names the ingredients but gives no formula, so each ingredient is a configurable component):
        score(P) = sum_c lambda_c * component_c(P) / sum_c lambda_c            (every component is in [0, 1])
        priority          mean R over the path's nodes             (eigenvector centrality + severity, Sec. 3.6)
        confidence        mean GCN confidence over the path's nodes (predicted high-risk nodes)
        structure_weight  mean over hops of sum_{k : hop is a step of chi_k} w_k   (MeGiTS structure weights, Eq. 1)
        coherence         mean MeGiTS adjacency over same-type node pairs of the path (default weight 0)
   Defaults: lambda = 1 for priority, confidence and structure_weight; 0 for coherence.
5. Ranking is by descending score (rounded to 10 decimals); ties by the node-id sequence, then the relation sequence.

Evaluation (reference paths are used ONLY here, never for training, seeds, scope or tracing):
  Exact-Path Hit@5   1 if one of the 5 highest-ranked candidates equals the reference in ordered nodes, relations and
                     edge directions (``match="strict"``).  ``exact_hit_either`` also accepts the reversed reading.
  Edge-F1            F1 between the typed-edge sets {(source id, relation, target id) in STORED direction} of the
                     top-ranked candidate and of the reference (0 if there is no candidate).  ``relation_f1`` is the F1 of the
                     relation-name multisets (the looser "overlap between the relations" reading).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import scipy.sparse as sp

from iocevaluator.labels import ReferencePath
from iocevaluator.megits import normalise_weights
from iocevaluator.metagraphs import STRUCTURES, StructureSpec, default_structures, structure_templates
from iocevaluator.tikg import TIKG

Step = Tuple[object, str, object, str]
SUPPORTS = ("step", "template")
COMPONENTS = ("priority", "confidence", "structure_weight", "coherence")
ROUND_DECIMALS = 10


# ------------------------------------------------------------------------------------------------ configuration
@dataclass(frozen=True)
class PathConfig:
    min_edges: int = 2                      # the manuscript groups reference paths as 2, 3 and >= 4 edges
    max_edges: int = 6                      # NOT stated in the manuscript: a computational bound (documented)
    conf_threshold: float = 0.5             # "high-confidence predicted nodes" (same 0.5 as the label threshold)
    priority_top_n: Optional[int] = None    # keep only the n best-R seeds (None = all high-confidence seeds)
    support: str = "step"                   # "step" | "template" (see module docstring)
    max_paths_per_seed: int = 2000
    max_expansions_per_seed: int = 50000
    max_candidates: int = 5000              # per case, deterministic truncation after ranking
    weights: Tuple[Tuple[str, float], ...] = (("priority", 1.0), ("confidence", 1.0),
                                              ("structure_weight", 1.0), ("coherence", 0.0))
    top_k: int = 5                          # Exact-Path Hit@k
    match: str = "strict"                   # "strict" (ordered) | "either" (also the reversed reading)

    def __post_init__(self):
        if self.support not in SUPPORTS or self.match not in ("strict", "either"):
            raise ValueError("unknown support / match")
        if not 1 <= self.min_edges <= self.max_edges:
            raise ValueError("need 1 <= min_edges <= max_edges")
        w = dict(self.weights)
        if set(w) != set(COMPONENTS) or min(w.values()) < 0 or sum(w.values()) <= 0:
            raise ValueError(f"weights must give a non-negative value for each of {COMPONENTS}, not all zero")

    def lambdas(self) -> Dict[str, float]:
        return dict(self.weights)


# ------------------------------------------------------------------------------------------------ schema support
def _flip(step: Step) -> Step:
    f, r, t, d = step
    return (t, r, f, "rev" if d == "fwd" else "fwd")


class StructureSupport:
    """chi-structure templates (and their reversals) as typed steps; per-step and per-template lookups."""

    def __init__(self, structures: Optional[Sequence[StructureSpec]] = None):
        structures = list(structures) if structures is not None else default_structures()
        self.ids = [s.id for s in structures]
        self.templates: List[Tuple[int, Tuple[Step, ...]]] = []
        self.step_structures: Dict[Step, Set[int]] = {}
        for s in structures:
            for tpl in structure_templates(s.id):
                for seq in (tpl, tuple(_flip(x) for x in reversed(tpl))):
                    self.templates.append((s.id, tuple(seq)))
                    for st in seq:
                        self.step_structures.setdefault(st, set()).add(s.id)

    def supports_step(self, step: Step) -> bool:
        return step in self.step_structures

    def template_matches(self, steps: Sequence[Step]) -> Set[int]:
        """chi ids having a template that contains ``steps`` as a contiguous segment."""
        L, out = len(steps), set()
        for k, tpl in self.templates:
            for off in range(len(tpl) - L + 1):
                if tuple(tpl[off:off + L]) == tuple(steps):
                    out.add(k)
        return out


# ------------------------------------------------------------------------------------------------ candidate paths
@dataclass
class CandidatePath:
    nodes: Tuple[int, ...]
    relations: Tuple[str, ...]
    directions: Tuple[str, ...]              # per hop: "fwd" (stored s->t) or "rev" (traversed t->s)
    structures: Tuple[int, ...] = ()         # chi ids supporting EVERY hop (may be empty under support="step")
    score: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def n_edges(self) -> int:
        return len(self.relations)

    def key(self, tikg: TIKG) -> Tuple:
        return (tuple(tikg.entities[i].id for i in self.nodes), self.relations, self.directions)

    def typed_edges(self, tikg: TIKG) -> List[Tuple[str, str, str]]:
        """Edges as stored triplets (source id, relation, target id) - independent of reading order."""
        out = []
        for i, (r, d) in enumerate(zip(self.relations, self.directions)):
            a, b = tikg.entities[self.nodes[i]].id, tikg.entities[self.nodes[i + 1]].id
            out.append((a, r, b) if d == "fwd" else (b, r, a))
        return out

    def audit(self, tikg: TIKG) -> List[Dict]:
        """Human-auditable hops: node ids, relation, traversal direction, Table-2 triplet signature."""
        rows = []
        for i, (r, d) in enumerate(zip(self.relations, self.directions)):
            u, v = self.nodes[i], self.nodes[i + 1]
            s, t = (u, v) if d == "fwd" else (v, u)
            sig = next((tr.sig for tr in tikg.triplets if tr.s == s and tr.t == t and tr.r == r), "")
            rows.append({"from": tikg.entities[u].id, "relation": r, "to": tikg.entities[v].id, "direction": d,
                         "triplet": sig, "from_type": tikg.entities[u].type.value, "to_type": tikg.entities[v].type.value})
        return rows

    def describe(self, tikg: TIKG) -> str:
        parts = [tikg.entities[self.nodes[0]].id]
        for i, (r, d) in enumerate(zip(self.relations, self.directions)):
            arrow = f"-{r}->" if d == "fwd" else f"<-{r}-"
            parts += [arrow, tikg.entities[self.nodes[i + 1]].id]
        return " ".join(parts)


def _neighbours(tikg: TIKG) -> Dict[int, List[Tuple[int, str, str, Step]]]:
    """Typed neighbours of every node in a fixed, deterministic order (entity id, relation, direction)."""
    nb: Dict[int, List] = {i: [] for i in range(tikg.n)}
    for t in tikg.triplets:
        if t.s == t.t:
            continue
        ts, tt = tikg.entities[t.s].type, tikg.entities[t.t].type
        nb[t.s].append((t.t, t.r, "fwd", (ts, t.r, tt, "fwd")))
        nb[t.t].append((t.s, t.r, "rev", (tt, t.r, ts, "rev")))
    for i in nb:
        nb[i].sort(key=lambda x: (tikg.entities[x[0]].id, x[1], x[2]))
    return nb


def priority_key(tikg: TIKG, risk: np.ndarray, i: int) -> Tuple[float, str]:
    r = risk[i]
    return (-round(float(r), ROUND_DECIMALS) if r == r else float("inf"), tikg.entities[i].id)


@dataclass
class TraceResult:
    candidates: List[CandidatePath]
    seeds: np.ndarray
    scope: np.ndarray
    truncated: bool = False


def select_seeds(tikg: TIKG, scope: np.ndarray, conf: np.ndarray, risk: np.ndarray, cfg: PathConfig) -> np.ndarray:
    """High-confidence nodes of the scope (confidence >= threshold), optionally the ``priority_top_n`` best by R."""
    idx = [int(i) for i in scope if conf[i] >= cfg.conf_threshold]
    idx.sort(key=lambda i: priority_key(tikg, risk, i))
    if cfg.priority_top_n is not None:
        idx = idx[: cfg.priority_top_n]
    return np.array(idx, dtype=int)


def trace_paths(tikg: TIKG, scope: np.ndarray, conf: np.ndarray, risk: np.ndarray, mega_adj: Optional[sp.spmatrix] = None,
                weights: Optional[Dict[int, float]] = None, cfg: PathConfig = PathConfig(),
                support: Optional[StructureSupport] = None, nb: Optional[Dict] = None) -> TraceResult:
    """Trace, score and rank candidate attack paths inside ``scope`` (node indices).  Takes NO reference paths.

    conf / risk: per-node GCN confidence and prioritisation score R (``ranking.ThreatPrioritisationRanker``)."""
    support = support or StructureSupport()
    nb = nb or _neighbours(tikg)
    scope = np.unique(np.asarray(scope, dtype=int))
    in_scope = np.zeros(tikg.n, dtype=bool)
    in_scope[scope] = True
    seeds = select_seeds(tikg, scope, conf, risk, cfg)
    seed_set = set(int(s) for s in seeds)
    prio = {i: priority_key(tikg, risk, i) for i in seed_set}
    found: Dict[Tuple, CandidatePath] = {}
    truncated = False

    for s in seeds:
        s = int(s)
        budget = [cfg.max_paths_per_seed, cfg.max_expansions_per_seed]
        stack_nodes, stack_rels, stack_dirs, stack_steps = [s], [], [], []

        def dfs(alive):
            nonlocal truncated
            if budget[0] <= 0 or budget[1] <= 0:
                truncated = True
                return
            L = len(stack_rels)
            end = stack_nodes[-1]
            if L >= cfg.min_edges and end in seed_set and end != s and prio[s] < prio[end]:
                cand = CandidatePath(tuple(stack_nodes), tuple(stack_rels), tuple(stack_dirs))
                found.setdefault(cand.key(tikg), cand)
                budget[0] -= 1
            if L >= cfg.max_edges:
                return
            for nxt, rel, d, step in nb[end]:
                budget[1] -= 1
                if nxt in stack_nodes or not in_scope[nxt] or not support.supports_step(step):
                    continue
                new_alive = None
                if cfg.support == "template":
                    new_alive = {(ti, off) for ti, off in alive
                                 if off + L < len(support.templates[ti][1]) and support.templates[ti][1][off + L] == step}
                    if not new_alive:
                        continue
                stack_nodes.append(nxt); stack_rels.append(rel); stack_dirs.append(d); stack_steps.append(step)
                dfs(new_alive)
                stack_nodes.pop(); stack_rels.pop(); stack_dirs.pop(); stack_steps.pop()
                if budget[0] <= 0 or budget[1] <= 0:
                    truncated = True
                    return

        alive0 = None
        if cfg.support == "template":
            alive0 = {(ti, off) for ti, (_, tpl) in enumerate(support.templates) for off in range(len(tpl))}
        dfs(alive0)

    cands = list(found.values())
    w = weights if weights is not None else normalise_weights(default_structures())
    lam = cfg.lambdas()
    total = sum(lam.values())
    A = sp.csr_matrix(mega_adj) if mega_adj is not None else None
    for p in cands:
        steps = [(tikg.entities[p.nodes[i]].type, p.relations[i], tikg.entities[p.nodes[i + 1]].type, p.directions[i])
                 for i in range(p.n_edges)]
        per_step = [support.step_structures.get(st, set()) for st in steps]
        p.structures = tuple(sorted(set.intersection(*per_step))) if per_step else ()
        idx = list(p.nodes)
        comp = {"priority": float(np.nan_to_num(risk[idx], nan=0.0, posinf=0.0, neginf=0.0).mean()),
                "confidence": float(np.mean(conf[idx])),
                "structure_weight": float(np.mean([sum(w.get(k, 0.0) for k in ks) for ks in per_step])),
                "coherence": 0.0}
        if A is not None and lam["coherence"] > 0:
            vals = [A[a, b] for ii, a in enumerate(idx) for b in idx[ii + 1:] if tikg.types[a] == tikg.types[b]]
            comp["coherence"] = float(np.mean(vals)) if vals else 0.0
        p.components = comp
        p.score = float(sum(lam[c] * comp[c] for c in COMPONENTS) / total)
    cands.sort(key=lambda p: (-round(p.score, ROUND_DECIMALS), p.key(tikg)))
    if len(cands) > cfg.max_candidates:
        cands, truncated = cands[: cfg.max_candidates], True
    return TraceResult(cands, seeds, scope, truncated)


# ------------------------------------------------------------------------------------------------ reference paths
@dataclass
class ResolvedReference:
    case_id: str
    nodes: Tuple[str, ...]
    relations: Tuple[str, ...]
    directions: Tuple[str, ...]
    edges: Tuple[Tuple[str, str, str], ...]     # stored-direction typed edges
    valid: bool = True                          # every edge exists in the TIKG with that relation


def resolve_reference(tikg: TIKG, ref: ReferencePath) -> ResolvedReference:
    """Express a reference path in the TIKG's terms: edge directions are derived from the stored triplets
    (the optional ``directions`` of the file are only cross-checked by the tests).  Used for scoring only."""
    stored = {(t.s, t.r, t.t) for t in tikg.triplets}
    ok = all(n in tikg.index for n in ref.nodes)
    rels, dirs, edges = [], [], []
    if ok:
        idx = [tikg.index[n] for n in ref.nodes]
        for i in range(len(idx) - 1):
            a, b = idx[i], idx[i + 1]
            want = ref.relations[i] if ref.relations else None
            cand = [(r, "fwd") for (s, r, t) in stored if (s, t) == (a, b) and want in (None, r)] + \
                   [(r, "rev") for (s, r, t) in stored if (s, t) == (b, a) and want in (None, r)]
            if len(cand) != 1:
                ok = False
                break
            r, d = sorted(cand)[0]
            rels.append(r); dirs.append(d)
            edges.append((ref.nodes[i], r, ref.nodes[i + 1]) if d == "fwd" else (ref.nodes[i + 1], r, ref.nodes[i]))
    return ResolvedReference(ref.case_id, tuple(ref.nodes), tuple(rels), tuple(dirs), tuple(edges), ok)


def _f1(pred: Counter, ref: Counter) -> float:
    tp = sum((pred & ref).values())
    if not pred or not ref or tp == 0:
        return 0.0
    p, r = tp / sum(pred.values()), tp / sum(ref.values())
    return 2 * p * r / (p + r)


def edge_f1_typed(pred_edges: Sequence[Tuple[str, str, str]], ref_edges: Sequence[Tuple[str, str, str]]) -> float:
    """F1 of the typed-edge sets {(source, relation, target)} (stored direction); 0 when nothing is predicted."""
    return _f1(Counter(set(pred_edges)), Counter(set(ref_edges)))


def relation_f1(pred_relations: Sequence[str], ref_relations: Sequence[str]) -> float:
    """F1 of the relation-name multisets (looser reading of "overlap between the relations")."""
    return _f1(Counter(pred_relations), Counter(ref_relations))


def exact_match(tikg: TIKG, cand: CandidatePath, ref: ResolvedReference, either: bool = False) -> bool:
    """Same ordered nodes, relations and edge directions (``either``: also the reversed reading)."""
    if not ref.valid:
        return False
    ids, rels, dirs = cand.key(tikg)
    if (ids, rels, dirs) == (ref.nodes, ref.relations, ref.directions):
        return True
    if either:
        flip = {"fwd": "rev", "rev": "fwd"}
        return (ids[::-1], rels[::-1], tuple(flip[d] for d in dirs[::-1])) == (ref.nodes, ref.relations, ref.directions)
    return False


def length_bucket(n_edges: int) -> str:
    return "2" if n_edges == 2 else "3" if n_edges == 3 else ">=4"


def evaluate_case(tikg: TIKG, ref: ReferencePath, trace: TraceResult, cfg: PathConfig = PathConfig()) -> Dict:
    """Per-case result: Exact-Path Hit@k, Edge-F1 of the top-ranked path, diagnostics.  The only place a reference
    path is touched."""
    rr = resolve_reference(tikg, ref)
    cands = trace.candidates
    top = cands[: cfg.top_k]
    strict = [exact_match(tikg, c, rr, False) for c in cands]
    either = [exact_match(tikg, c, rr, True) for c in cands]
    hit_strict, hit_either = any(strict[: cfg.top_k]), any(either[: cfg.top_k])
    first = cands[0] if cands else None
    seeded = rr.valid and {tikg.index[rr.nodes[0]], tikg.index[rr.nodes[-1]]} <= set(int(s) for s in trace.seeds)
    return {"case_id": ref.case_id, "campaign": ref.campaign, "n_edges": len(ref.nodes) - 1,
            "bucket": length_bucket(len(ref.nodes) - 1), "ref_valid": rr.valid,
            "ref_within_max_edges": (len(ref.nodes) - 1) <= cfg.max_edges,
            "ref_endpoints_seeded": bool(seeded), "n_scope_nodes": int(len(trace.scope)), "n_seeds": int(len(trace.seeds)),
            "n_candidates": len(cands), "truncated": bool(trace.truncated),
            "exact_hit": float(hit_strict if cfg.match == "strict" else hit_either),
            "exact_hit_strict": float(hit_strict), "exact_hit_either": float(hit_either),
            "ref_rank": (strict.index(True) + 1) if any(strict) else float("nan"),
            "edge_f1": edge_f1_typed(first.typed_edges(tikg), rr.edges) if first and rr.valid else 0.0,
            "relation_f1": relation_f1(first.relations, rr.relations) if first and rr.valid else 0.0,
            "top1_path": first.describe(tikg) if first else "",
            "top1_score": float(first.score) if first else float("nan"),
            "top_paths": " | ".join(c.describe(tikg) for c in top),
            "top_scores": " | ".join(f"{c.score:.6f}" for c in top)}


def trace_case(tikg: TIKG, campaign: str, conf: np.ndarray, risk: np.ndarray, mega_adj=None, weights=None,
               cfg: PathConfig = PathConfig(), support: Optional[StructureSupport] = None, nb=None) -> TraceResult:
    """Candidate paths for one case.  Scope = all entities of the case's campaign (from case metadata only)."""
    scope = np.flatnonzero(tikg.campaigns() == campaign)
    return trace_paths(tikg, scope, conf, risk, mega_adj, weights, cfg, support, nb)


# ------------------------------------------------------------------------------------------------ aggregation
def summarise_cases(cases, n_boot: int = 2000, level: float = 0.95) -> Dict:
    """Path-level summary by length bucket (2, 3, >=4, overall) like Table ``path_length``.

    Per bucket: n_cases; Exact-Path Hit@k and Edge-F1 as case-pooled means (seed-averaged per case); the std across
    seeds of the pooled mean; and a bootstrap-over-cases percentile interval (deterministic)."""
    import pandas as pd
    df = cases if isinstance(cases, pd.DataFrame) else pd.DataFrame(cases)
    out = {}
    if df.empty:
        return out
    rng = np.random.RandomState(0)
    for name, sub in [(b, df[df.bucket == b]) for b in ("2", "3", ">=4")] + [("overall", df)]:
        if sub.empty:
            continue
        entry = {"n_cases": int(sub.case_id.nunique()), "n_seeds": int(sub.seed.nunique())}
        for m in ("exact_hit", "exact_hit_either", "edge_f1", "relation_f1"):
            per_case = sub.groupby("case_id")[m].mean().to_numpy()
            per_seed = sub.groupby("seed")[m].mean().to_numpy()
            boots = per_case[rng.randint(0, len(per_case), (n_boot, len(per_case)))].mean(1)
            entry[m] = {"mean": float(per_case.mean()),
                        "std_over_seeds": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else float("nan"),
                        "ci95_cases_boot": [float(np.quantile(boots, (1 - level) / 2)),
                                            float(np.quantile(boots, 0.5 + level / 2))]}
        out[name] = entry
    return out


def path_config_info(cfg: PathConfig) -> Dict:
    d = asdict(cfg)
    d["weights"] = dict(cfg.weights)
    return d
