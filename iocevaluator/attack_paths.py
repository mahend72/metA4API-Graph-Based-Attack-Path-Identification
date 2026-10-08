"""Attack-path tracing and ranking (manuscript Algorithm 2, steps "select high-confidence nodes / trace
candidate paths conforming to chi / rank using MeGiTS weights and eigenvector-centrality").

The manuscript gives no closed-form path score or seed rule, so these are explicit, configurable choices:
  * seeds      : nodes whose max predicted label probability >= ``conf_threshold`` (within the case scope);
  * candidates : simple paths of ``min_edges..max_edges`` edges over the *observed* TIKG, with both endpoints
                 seeds, whose sequence of typed traversal steps is a contiguous segment of a schema template
                 of some chi_k (templates in ``metagraphs.STRUCTURES``);
  * score      : mean(node risk) * mean(node confidence) * (1 + coherence) * max_k w_k  where node risk is EC (or the
                 fused risk R), coherence is the mean MeGiTS adjacency over all node pairs of the path and w_k are
                 the MeGiTS weights of the matched structures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import scipy.sparse as sp

from .labels import ReferencePath
from .metagraphs import Step, default_structures, structure_templates
from .metrics import edge_f1, exact_path_hit_at_k
from .tikg import TIKG


@dataclass
class CandidatePath:
    nodes: Tuple[int, ...]
    relations: Tuple[str, ...]
    structures: Tuple[int, ...]
    score: float = 0.0

    @property
    def n_edges(self) -> int:
        return len(self.nodes) - 1


@dataclass
class PathConfig:
    min_edges: int = 2
    max_edges: int = 5
    conf_threshold: float = 0.5
    max_paths_per_seed: int = 5000


def _flip(step: Step) -> Step:
    f, r, t, d = step
    return (t, r, f, "rev" if d == "fwd" else "fwd")


def _template_table(structure_ids: Sequence[int]):
    """List of (chi_id, steps) including reversed templates."""
    out = []
    for k in structure_ids:
        for tpl in structure_templates(k):
            out.append((k, tpl))
            out.append((k, tuple(_flip(s) for s in reversed(tpl))))
    return out


def _adjacency_lists(tikg: TIKG):
    nb: Dict[int, List[Tuple[int, str, Step]]] = {i: [] for i in range(tikg.n)}
    for t in tikg.triplets:
        ts, tt = tikg.entities[t.s].type, tikg.entities[t.t].type
        nb[t.s].append((t.t, t.r, (ts, t.r, tt, "fwd")))
        nb[t.t].append((t.s, t.r, (tt, t.r, ts, "rev")))
    return nb


def trace_candidate_paths(tikg: TIKG, seeds: np.ndarray, node_risk: np.ndarray, node_conf: np.ndarray,
                          megits_adj: sp.csr_matrix, weights: Dict[int, float], cfg: PathConfig = PathConfig(),
                          scope: Optional[np.ndarray] = None, structure_ids: Optional[Sequence[int]] = None
                          ) -> List[CandidatePath]:
    ids = list(structure_ids or [s.id for s in default_structures()])
    table = _template_table(ids)
    nb = _adjacency_lists(tikg)
    seed_set = set(int(s) for s in seeds)
    in_scope = (lambda i: True) if scope is None else (lambda i: bool(scope[i]))
    found: Dict[Tuple[int, ...], CandidatePath] = {}
    A = sp.csr_matrix(megits_adj)

    def dfs(path: List[int], rels: List[str], alive: Set[Tuple[int, int]], budget: List[int]):
        if budget[0] <= 0:
            return
        L = len(rels)
        if L >= cfg.min_edges and path[-1] in seed_set and path[-1] != path[0]:
            key = tuple(path) if tuple(path) <= tuple(reversed(path)) else tuple(reversed(path))
            if key not in found:
                chis = sorted({table[ti][0] for ti, _ in alive})
                rr = tuple(rels) if key == tuple(path) else tuple(reversed(rels))
                found[key] = CandidatePath(key, rr, tuple(chis))
            budget[0] -= 1
        if L >= cfg.max_edges:
            return
        for nxt, rel, step in nb[path[-1]]:
            if nxt in path or not in_scope(nxt):
                continue
            new_alive = {(ti, off) for ti, off in alive
                         if off + L < len(table[ti][1]) and table[ti][1][off + L] == step}
            if not new_alive:
                continue
            dfs(path + [nxt], rels + [rel], new_alive, budget)

    for s in seed_set:
        if not in_scope(s):
            continue
        alive0 = {(ti, off) for ti, (_, tpl) in enumerate(table) for off in range(len(tpl))}
        dfs([s], [], alive0, [cfg.max_paths_per_seed])

    out = list(found.values())
    for p in out:
        idx = list(p.nodes)
        sub = A[idx][:, idx]
        pairs = len(idx) * (len(idx) - 1)
        coherence = float(sub.sum() / pairs) if pairs else 0.0
        wmax = max((weights.get(k, 0.0) for k in p.structures), default=0.0)
        p.score = float(np.mean(node_risk[idx]) * np.mean(node_conf[idx]) * (1.0 + coherence) * wmax)
    out.sort(key=lambda p: -p.score)
    return out


def evaluate_reference_paths(tikg: TIKG, refs: Sequence[ReferencePath], probs: np.ndarray, node_risk: np.ndarray,
                             megits_adj: sp.csr_matrix, weights: Dict[int, float], cfg: PathConfig = PathConfig(),
                             k: int = 5) -> Dict:
    """Exact-Path Hit@k and Edge-F1 per path-length bucket (Table `path_length`).

    `probs` should be out-of-fold predictions. Each case is scoped to the entities of its `campaign` (or the
    explicit `case_nodes`); the reference path itself is never shown to the tracer."""
    conf = probs.max(1) if probs.ndim == 2 else probs
    rows = []
    camp = tikg.campaigns()
    for r in refs:
        if r.case_nodes:
            scope = np.zeros(tikg.n, dtype=bool)
            scope[[tikg.index[i] for i in r.case_nodes if i in tikg.index]] = True
        else:
            scope = camp == r.campaign
        seeds = np.flatnonzero(scope & (conf >= cfg.conf_threshold))
        cands = trace_candidate_paths(tikg, seeds, node_risk, conf, megits_adj, weights, cfg, scope=scope)
        ranked = [[tikg.entities[i].id for i in c.nodes] for c in cands]
        top = cands[0] if cands else None
        hit = exact_path_hit_at_k(ranked, r.nodes, k)
        ef = edge_f1([tikg.entities[i].id for i in top.nodes] if top else None, list(top.relations) if top else None,
                     r.nodes, r.relations or None)
        n_edges = len(r.nodes) - 1
        rows.append({"case_id": r.case_id, "n_edges": n_edges, "hit": float(hit), "edge_f1": ef,
                     "n_candidates": len(cands)})
    def agg(sel):
        return {"n": len(sel), "exact_path_hit": float(np.mean([x["hit"] for x in sel])) if sel else float("nan"),
                "edge_f1": float(np.mean([x["edge_f1"] for x in sel])) if sel else float("nan")}
    return {"2_edges": agg([x for x in rows if x["n_edges"] == 2]), "3_edges": agg([x for x in rows if x["n_edges"] == 3]),
            ">=4_edges": agg([x for x in rows if x["n_edges"] >= 4]), "overall": agg(rows), "cases": rows}
