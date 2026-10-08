"""External CTI comparison baselines: AttacKG (Li et al., ESORICS 2022) and LADDER (Alam et al., RAID 2023).

RESULT INTEGRITY.  No manuscript score is used, tuned against or compared here.  Every hyper-parameter lives in
``AttacKGConfig`` / ``LADDERConfig``; the values the source papers leave unspecified were fixed from the descriptions alone
before any result existed.  The only fitted quantity is each baseline's decision threshold, selected on the inner
validation partition (the analogue of the "pre-defined" / "experimentally identified" thresholds of the papers).

Both source systems consume free-text CTI REPORTS; our data is a typed knowledge graph with short per-entity text.  What is
reproduced exactly, what is adapted and what cannot be reproduced is listed in ``COMPONENTS`` (and recorded in every manifest).

AttacKG (arXiv 2111.07093).  Reproduced: technique templates as graphs whose nodes aggregate IoC-term and NLP-description
sets with occurrence counts, the node alignment score (Eq. 1-2: 0 for different types, gamma + (1-gamma) Sim otherwise,
Sim = max(sim_IoC, sim_NLP)), the node-level (Eq. 3), dependency-level (Eq. 4, with 1/Cmin hop penalty) and combined (Eq. 5)
graph alignment scores, a candidate threshold on node alignment, and a decision threshold on the graph alignment score.
Adapted: "technique" = label class; "technique example" = training node with that label; the "attack graph" = the h-hop
neighbourhood of the node being classified in the TIKG; template nodes = entity type (+ file subtype).  Not reproducible:
the five-stage NLP report parser, ATT&CK procedure-example crawling, template updating from new reports.

LADDER (arXiv 2211.01753).  Reproduced: step 3 of TTPClassifier - the weighted title / description cosine distance
d_i = w_t cos(v_phrase, v_title_i) + (1 - w_t) cos(v_phrase, v_desc_i), the arg-min over the technique list of the
"platform" (here: the classes valid for the node's entity type), accepted only if d < tau, ONE technique per phrase.
Adapted: phrase = node name + text; title = class name; descriptions = texts of the TRAINING nodes of the class; embedder =
TF-IDF (``sbert`` available when sentence-transformers is installed).  Not reproducible: the fine-tuned transformer sentence
classifier (step 1), the sequence-tagging phrase extractor (step 2), the relation classifier and the NER / knowledge-graph
construction (they need annotated report text).  LADDER ignores the graph (text only).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

from .ablation import AblationResult, build_ablation_manifest, provenance_info, write_ablation
from .evaluation import EvalConfig, EvaluationResult, evaluate_cv, verify_fold
from .labels import LabelSpace
from .metrics import auc_scores, multilabel_prf, per_type_prf
from .model_comparison import MODELS, parameter_table
from .splits import Fold, group_stratified_folds
from .supervised_gcn import LabelLeakageError, isolate_nodes, make_masks, type_heads, visible_labels
from .tikg import TIKG

REFERENCE_MODEL = "gcn"

COMPONENTS: Dict[str, Dict[str, List[str]]] = {
    "attackg": {
        "reproduced_exactly": [
            "node alignment score, Eq.(1)-(2): 0 if types differ, gamma + (1-gamma)*Sim otherwise; Sim = max(sim_IoC, sim_NLP)",
            "node-level alignment, Eq.(3): occurrence-weighted mean of node alignment scores",
            "dependency-level alignment, Eq.(4): Gamma(i:k)*Gamma(j:l)/Cmin(k->l) weighted by dependency occurrences",
            "combined graph alignment, Eq.(5): (Gamma_N + Gamma_E)/2, all scores in [0,1]",
            "template nodes aggregate IoC-term and NLP-description sets with occurrence counts; dependencies carry counts",
            "a node-alignment candidate threshold and a graph-alignment decision threshold",
        ],
        "adapted_to_tikg": [
            "technique = label class; technique examples = TRAINING nodes holding the class (no MITRE crawling)",
            "attack graph = h-hop neighbourhood (inference graph) of the node being classified; the anchor template node can only align to that node",
            "template node = entity type (+ file subtype) instead of free-form report entities; dependency = typed edge between template nodes",
            "IoC term = entity name; NLP description = entity text; character-level similarity = Sorensen-Dice over character bigrams (the paper cites an unspecified character-level measure)",
            "alignment is greedy and injective per template node (the paper enumerates candidate permutations)",
            "Eq.(4) is normalised by the total template dependency occurrences so that it lies in [0,1]",
            "decision threshold on the graph alignment score selected on the inner validation partition",
        ],
        "not_reproducible": [
            "five-stage NLP CTI report parser (needs report text and the EXTRACTOR-style pipeline)",
            "technique templates initialised from crawled MITRE ATT&CK procedure examples",
            "template updating / TKG construction from new reports",
            "the paper's gamma and threshold values (unspecified)",
        ]},
    "ladder": {
        "reproduced_exactly": [
            "step 3 of TTPClassifier: d_i = w_t*cos_dist(phrase,title_i) + (1-w_t)*cos_dist(phrase,desc_i), cos_dist = 1 - cosine",
            "arg-min over the candidate technique list, accepted only if d < tau, one technique per phrase",
        ],
        "adapted_to_tikg": [
            "technique list = classes valid for the node's entity type (analogue of the per-platform ATT&CK list)",
            "phrase = node name + text (no sentence / phrase extraction); title = class name; description = training-node text of the class (nearest example or centroid)",
            "embedder = TF-IDF fitted on training nodes (sentence-transformer when installed; offline default)",
            "tau selected on the inner validation partition; w_t fixed in configuration (unspecified in the paper)",
            "continuous score 1 - d_i for ROC/PR-AUC; the label decision is the arg-min / tau rule",
        ],
        "not_reproducible": [
            "fine-tuned transformer relevant-sentence classifier (step 1) and sequence-tagging attack-phrase extractor (step 2)",
            "NER and relation-classification models and the report-level knowledge graph",
            "ATT&CK technique descriptions as the description corpus (no external knowledge is used)",
            "the paper's pre-trained sentence-transformer, w_t and tau (unspecified / unavailable offline)",
        ]},
}

UNSUPPORTED_REASONS = {
    "top5_hit_rate": "no per-entity threat-priority ranking compatible with R = alpha*EC + (1-alpha)*Severity",
    "top10_hit_rate": "no per-entity threat-priority ranking compatible with R = alpha*EC + (1-alpha)*Severity",
    "exact_path_hit5": "the baseline does not output attack paths in the TIKG (template alignment / technique mapping only)",
    "path_edge_f1": "the baseline does not output attack paths in the TIKG (template alignment / technique mapping only)",
}
METRICS = ["macro_f1", "micro_f1", "roc_auc", "pr_auc", "per_entity_type", "top5_hit_rate", "top10_hit_rate",
           "exact_path_hit5", "path_edge_f1"]


# ------------------------------------------------------------------------------------------------ configuration
@dataclass(frozen=True)
class AttacKGConfig:
    gamma: float = 0.5                  # type-matched base score of Eq.(1); unspecified in the paper
    node_threshold: float = 0.5         # minimal Gamma(i:k) of an alignment candidate (0.5 = every type-matched node)
    hops: int = 2                       # radius of the attack graph around the classified node
    max_graph_nodes: int = 40           # nearest-first cap on the attack graph size
    max_template_instances: int = 200   # positives used per template (deterministic: lowest node index first)
    threshold_grid: Tuple[float, ...] = tuple(round(x, 2) for x in np.arange(0.05, 1.0, 0.05))
    bigram_features: int = 2 ** 18      # hashing space of the character-bigram vectors (stateless, nothing is fitted)


@dataclass(frozen=True)
class LADDERConfig:
    w_t: float = 0.5                    # title weight of the weighted distance; unspecified in the paper
    embedder: str = "tfidf"             # tfidf | sbert
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ngram_range: Tuple[int, int] = (1, 2)
    description_pooling: str = "nearest"    # nearest | centroid over the class's training-node descriptions
    tau_grid: Tuple[float, ...] = tuple(round(x, 2) for x in np.arange(0.05, 1.01, 0.05))


@dataclass
class BaselineOutput:
    score: np.ndarray                   # N x K continuous scores (0 outside the node's type)
    pred: np.ndarray                    # N x K bool decisions
    threshold: float
    val_macro_f1: float
    fit_time_s: float
    inference_time_s: float
    n_parameters: int
    diagnostics: Dict[str, float] = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------ shared helpers
def node_text(tikg: TIKG) -> Tuple[List[str], List[str]]:
    """(IoC term = name, NLP description = text-or-name) per node; no label field is read."""
    names = [e.name or e.id for e in tikg.entities]
    texts = [str(e.attrs.get("text") or e.name or e.id) for e in tikg.entities]
    return names, texts


def label_term_in_text_rate(tikg: TIKG, labels: LabelSpace) -> float:
    """Diagnostic: share of labelled nodes whose own name/text literally contains one of their positive class names.
    Text baselines read node text, so a high value signals label-revealing text (never used to alter a result)."""
    names, texts = node_text(tikg)
    hit = tot = 0
    for i in np.flatnonzero(labels.labelled):
        pos = [labels.names[k].split(":", 1)[1].lower() for k in np.flatnonzero((labels.Y[i] > 0) & labels.valid[i])]
        if not pos:
            continue
        tot += 1
        blob = f"{names[i]} {texts[i]}".lower()
        hit += any(p in blob for p in pos)
    return hit / tot if tot else float("nan")


def undirected_adjacency(tikg: TIKG) -> sp.csr_matrix:
    a = sum(tikg.relation_adjacencies().values()) if tikg.triplets else sp.csr_matrix((tikg.n, tikg.n))
    a = sp.csr_matrix(a)
    a = ((a + a.T) > 0).astype(np.float32)
    a.setdiag(0)
    a.eliminate_zeros()
    return a.tocsr()


def _guard(labels: LabelSpace, Yv: np.ndarray, masks) -> None:
    if (Yv[~masks.visible] != 0).any():
        raise LabelLeakageError("baseline received labels outside train U validation")


def _select_threshold(score: np.ndarray, pred_fn, Yv, valid, val_idx, grid) -> Tuple[float, float]:
    """Grid threshold maximising validation Macro-F1 (validation nodes only); ties -> the smallest threshold."""
    best, best_t = -1.0, float(grid[0])
    for t in grid:
        f = multilabel_prf(Yv, pred_fn(t), valid, val_idx)["macro_f1"]
        if f > best + 1e-12:
            best, best_t = f, float(t)
    return best_t, float(best)


def _ego(adj: sp.csr_matrix, v: int, hops: int, cap: int) -> List[int]:
    """Nearest-first BFS ball around v (ties by node index), at most ``cap`` nodes, v first."""
    seen = {v: 0}
    frontier = [v]
    for h in range(1, hops + 1):
        nxt = []
        for u in frontier:
            for w in adj.indices[adj.indptr[u]:adj.indptr[u + 1]]:
                if int(w) not in seen:
                    seen[int(w)] = h
                    nxt.append(int(w))
        frontier = sorted(nxt)
        if len(seen) >= cap:
            break
    return sorted(seen, key=lambda n: (seen[n], n))[:cap]


# ------------------------------------------------------------------------------------------------ AttacKG
def _bigram_dice(strings: Sequence[str], n_features: int) -> np.ndarray:
    """N x N character-bigram Sorensen-Dice similarity (stateless hashing: nothing is fitted on any node)."""
    hv = HashingVectorizer(analyzer="char", ngram_range=(2, 2), n_features=n_features, binary=True, norm=None,
                           alternate_sign=False, lowercase=True)
    a = hv.transform(list(strings)).astype(np.float32)
    inter = (a @ a.T).toarray()
    size = np.asarray(a.sum(1)).ravel()
    den = size[:, None] + size[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(den > 0, 2.0 * inter / den, 0.0)
    return d.astype(np.float32)


@dataclass
class _Template:
    keys: List[str]                      # node categories; keys[0] is the anchor (the class's own type)
    occ: np.ndarray                      # node occurrence counts
    sim: np.ndarray                      # (n_template_nodes x N) Sim(i, k) of Eq.(2) against every graph node
    deps: List[Tuple[int, int, float]]   # (i, j, occurrence) of template dependencies


def _category(tikg: TIKG, i: int) -> str:
    e = tikg.entities[i]
    return f"{e.type.value}:{e.subtype}" if e.subtype else e.type.value


def build_templates(tikg: TIKG, labels: LabelSpace, Yv: np.ndarray, masks, train_adj: sp.csr_matrix, cfg: AttacKGConfig,
                    dice_ioc: np.ndarray, dice_nlp: np.ndarray) -> Dict[int, _Template]:
    """One technique template per label class, built from TRAINING positives and the training graph only."""
    cats = [_category(tikg, i) for i in range(tikg.n)]
    train = np.flatnonzero(masks.train)
    templates: Dict[int, _Template] = {}
    for c in range(labels.K):
        pos = [int(i) for i in train if Yv[i, c] > 0 and labels.valid[i, c]][: cfg.max_template_instances]
        if not pos:
            continue
        anchor = cats[pos[0]]
        node_members: Dict[str, Dict[int, int]] = {anchor: {p: 1 for p in pos}}
        occ: Dict[str, int] = {anchor: len(pos)}
        dep: Dict[Tuple[str, str], int] = {}
        for p in pos:
            ball = _ego(train_adj, p, cfg.hops, cfg.max_graph_nodes)
            inst = set()
            for u in ball:
                k = cats[u]
                if u != p:
                    node_members.setdefault(k, {})
                    node_members[k][u] = node_members[k].get(u, 0) + 1
                    inst.add(k)
            for k in inst:
                occ[k] = occ.get(k, 0) + 1
            inb = set(ball)
            pairs = set()
            for u in ball:
                for w in train_adj.indices[train_adj.indptr[u]:train_adj.indptr[u + 1]]:
                    w = int(w)
                    if w in inb and cats[u] != cats[w]:
                        pairs.add((cats[u], cats[w]))
            for pr in pairs:
                dep[pr] = dep.get(pr, 0) + 1
        keys = [anchor] + sorted(k for k in node_members if k != anchor)
        sim = np.zeros((len(keys), tikg.n), dtype=np.float32)
        for a, k in enumerate(keys):
            cols = np.fromiter(node_members[k].keys(), dtype=int)
            sim[a] = np.maximum(dice_ioc[:, cols].max(1), dice_nlp[:, cols].max(1))     # Eq.(2)
        pos_in = {k: a for a, k in enumerate(keys)}
        templates[c] = _Template(keys, np.array([occ[k] for k in keys], dtype=float), sim,
                                 [(pos_in[a], pos_in[b], float(n)) for (a, b), n in sorted(dep.items())
                                  if a in pos_in and b in pos_in])
    return templates


def alignment_score(tpl: _Template, v: int, ball: List[int], cats: List[str], hop: np.ndarray, cfg: AttacKGConfig) -> float:
    """Gamma(Gt :: Ga) of Eq.(3)-(5) for the attack graph ``ball`` (v first) and ``hop`` = pairwise hop distances."""
    pos = {u: a for a, u in enumerate(ball)}
    taken: Dict[int, int] = {}                       # template node -> aligned graph node
    used: set = set()
    gam: Dict[int, float] = {}
    order = np.argsort(-tpl.occ, kind="stable")
    for i in [0] + [int(x) for x in order if x != 0]:
        cand = [v] if i == 0 else [u for u in ball if u != v]
        best, bu = -1.0, None
        for u in cand:
            if u in used or cats[u] != tpl.keys[i]:
                continue
            g = cfg.gamma + (1.0 - cfg.gamma) * float(tpl.sim[i, u])                   # Eq.(1)
            if g >= cfg.node_threshold and g > best + 1e-12:
                best, bu = g, u
        if bu is not None:
            taken[i], gam[i] = bu, best
            used.add(bu)
    gn = sum(gam[i] * tpl.occ[i] for i in gam) / tpl.occ.sum()                          # Eq.(3)
    tot = sum(o for _, _, o in tpl.deps)
    if tot <= 0:
        return float(gn)                                                                # no dependency in the template
    ge = 0.0
    for i, j, o in tpl.deps:
        if i in taken and j in taken:
            cmin = hop[pos[taken[i]], pos[taken[j]]]
            if np.isfinite(cmin) and cmin > 0:
                ge += gam[i] * gam[j] / cmin * o                                        # Eq.(4)
    return float(0.5 * (gn + ge / tot))                                                 # Eq.(5)


def run_attackg(tikg: TIKG, labels: LabelSpace, Yv: np.ndarray, masks, test_nodes: np.ndarray, adj: sp.csr_matrix,
                cfg: AttacKGConfig = AttacKGConfig()) -> BaselineOutput:
    """Fit on the fold's visible labels (templates from training nodes) and score every node; test labels never enter."""
    _guard(labels, Yv, masks)
    t0 = time.perf_counter()
    train_adj = isolate_nodes(adj, test_nodes)
    names, texts = node_text(tikg)
    di, dn = _bigram_dice(names, cfg.bigram_features), _bigram_dice(texts, cfg.bigram_features)
    templates = build_templates(tikg, labels, Yv, masks, train_adj, cfg, di, dn)
    t_build = time.perf_counter() - t0
    t1 = time.perf_counter()
    cats = [_category(tikg, i) for i in range(tikg.n)]
    score = np.zeros((tikg.n, labels.K), dtype=np.float64)
    for v in range(tikg.n):
        cols = [c for c in np.flatnonzero(labels.valid[v]) if c in templates]
        if not cols:
            continue
        ball = _ego(adj, v, cfg.hops, cfg.max_graph_nodes)
        sub = adj[ball][:, ball]
        hop = shortest_path(sub, method="D", unweighted=True, directed=False)
        for c in cols:
            score[v, c] = alignment_score(templates[c], v, ball, cats, hop, cfg)
    t_inf = time.perf_counter() - t1
    t2 = time.perf_counter()
    theta, vf = _select_threshold(score, lambda t: score > t, Yv, labels.valid, np.flatnonzero(masks.val), cfg.threshold_grid)
    return BaselineOutput(score, score > theta, theta, vf, t_build + time.perf_counter() - t2, t_inf, 0,
                          {"n_templates": float(len(templates)),
                           "n_template_nodes": float(sum(len(t.keys) for t in templates.values())),
                           "n_template_dependencies": float(sum(len(t.deps) for t in templates.values()))})


# ------------------------------------------------------------------------------------------------ LADDER
def _embedder(cfg: LADDERConfig, fit_texts: Sequence[str]):
    if cfg.embedder == "tfidf":
        vec = TfidfVectorizer(ngram_range=cfg.ngram_range, sublinear_tf=True, lowercase=True).fit(list(fit_texts))
        return lambda t: vec.transform(list(t))
    if cfg.embedder == "sbert":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:                                     # pragma: no cover - environment dependent
            raise RuntimeError("embedder='sbert' needs sentence-transformers (not installed)") from e
        m = SentenceTransformer(cfg.sbert_model)
        return lambda t: sp.csr_matrix(m.encode(list(t), normalize_embeddings=True))
    raise ValueError(f"unknown embedder {cfg.embedder!r}")


def _cos_dist(a, b) -> np.ndarray:
    """1 - cosine similarity between the rows of two (sparse) matrices."""
    def norm(m):
        m = sp.csr_matrix(m, dtype=np.float64)
        n = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel())
        return sp.diags(np.where(n > 0, 1.0 / np.where(n > 0, n, 1.0), 0.0)) @ m
    return 1.0 - (norm(a) @ norm(b).T).toarray()


def run_ladder(tikg: TIKG, labels: LabelSpace, Yv: np.ndarray, masks, cfg: LADDERConfig = LADDERConfig()) -> BaselineOutput:
    """TTPClassifier step 3 over node text: weighted title/description cosine distance, arg-min, threshold tau."""
    _guard(labels, Yv, masks)
    t0 = time.perf_counter()
    names, texts = node_text(tikg)
    phrases = [f"{n}. {t}" if t != n else n for n, t in zip(names, texts)]
    titles = [labels.names[k].split(":", 1)[1].replace("_", " ") for k in range(labels.K)]
    train = np.flatnonzero(masks.train)
    emb = _embedder(cfg, [phrases[i] for i in train] + titles)       # vocabulary / idf: TRAINING nodes + class titles only
    ph, ti = emb(phrases), emb(titles)
    desc: Dict[int, Any] = {}
    for c in range(labels.K):
        pos = [int(i) for i in train if Yv[i, c] > 0 and labels.valid[i, c]]
        if pos:
            desc[c] = ph[pos]
    t_fit = time.perf_counter() - t0
    t1 = time.perf_counter()
    d_title = _cos_dist(ph, ti)                                       # N x K
    d_desc = np.full_like(d_title, 1.0)                               # class without training description -> max distance
    for c, m in desc.items():
        if cfg.description_pooling == "centroid":
            m = sp.csr_matrix(m.mean(0))
        d_desc[:, c] = _cos_dist(ph, m).min(1)                        # nearest description / centroid
    dist = np.where(labels.valid, cfg.w_t * d_title + (1.0 - cfg.w_t) * d_desc, np.inf)
    score = np.where(labels.valid, 1.0 - dist, 0.0)
    best = np.argmin(dist, axis=1)
    t_inf = time.perf_counter() - t1

    def pred_fn(tau: float) -> np.ndarray:
        p = np.zeros(dist.shape, dtype=bool)
        ok = dist[np.arange(len(best)), best] < tau
        p[np.flatnonzero(ok), best[ok]] = True
        return p

    t2 = time.perf_counter()
    tau, vf = _select_threshold(score, pred_fn, Yv, labels.valid, np.flatnonzero(masks.val), cfg.tau_grid)
    return BaselineOutput(score, pred_fn(tau), tau, vf, t_fit + time.perf_counter() - t2, t_inf, 0,
                          {"n_class_descriptions": float(len(desc)), "n_vocabulary": float(ph.shape[1])})


# ------------------------------------------------------------------------------------------------ variants
@dataclass(frozen=True)
class BaselineVariant:
    name: str                        # attackg | ladder
    config: Any
    description: str = ""

    def info(self) -> Dict[str, Any]:
        return {"name": self.name, "group": "external_baseline", "model": self.name,
                "config": json.loads(json.dumps(asdict(self.config), default=str)),
                "components": COMPONENTS[self.name], "unsupported_metrics": UNSUPPORTED_REASONS,
                "n_trainable_parameters": 0, "seed_invariant": True, "description": self.description}


BASELINES: Dict[str, BaselineVariant] = {b.name: b for b in (
    BaselineVariant("attackg", AttacKGConfig(), "AttacKG technique-template graph alignment adapted to the TIKG"),
    BaselineVariant("ladder", LADDERConfig(), "LADDER TTPClassifier weighted title/description cosine mapping over node text"),
)}
DEFAULT_BASELINES = ("attackg", "ladder")


def select_baselines(names: Sequence[str] = DEFAULT_BASELINES) -> List[BaselineVariant]:
    unknown = [n for n in names if n not in BASELINES]
    if unknown:
        raise KeyError(f"unknown baseline(s): {unknown}")
    return [BASELINES[n] for n in dict.fromkeys(names)]


def metric_support(name: str, topk: bool = False, paths: bool = False) -> Dict[str, Dict[str, Any]]:
    """Which metrics a method supports; unsupported ones are recorded as N/A with the reason."""
    out = {m: {"supported": True, "reason": ""} for m in METRICS}
    if name == REFERENCE_MODEL or name in MODELS:
        out["top5_hit_rate"] = out["top10_hit_rate"] = {"supported": bool(topk), "reason": "" if topk else "not requested (no top-k cases supplied)"}
        out["exact_path_hit5"] = out["path_edge_f1"] = {"supported": bool(paths), "reason": "" if paths else "not requested (no reference paths supplied)"}
        return out
    for m, why in UNSUPPORTED_REASONS.items():
        out[m] = {"supported": False, "reason": why}
    return out


# ------------------------------------------------------------------------------------------------ evaluation
def _metrics(labels: LabelSpace, tikg: TIKG, out: BaselineOutput, fold: Fold, heads, ks) -> Tuple[Dict, List[Dict]]:
    m = multilabel_prf(labels.Y, out.pred, labels.valid, fold.test)
    m["macro_f1_all_labels"] = multilabel_prf(labels.Y, out.pred, labels.valid, fold.test, macro_over="all")["macro_f1"]
    m.update(auc_scores(labels.Y, out.score, labels.valid, fold.test))
    m.update({f"top{k}_hit_rate": float("nan") for k in ks}, topk_n_cases=0)          # N/A: see UNSUPPORTED_REASONS
    cols = {t.value: h.columns for t, h in heads.items()}
    typed = per_type_prf(labels.Y, out.pred, labels.valid, fold.test, cols, tikg.types)
    rows = []
    for t, tm in typed.items():
        sub = fold.test[tikg.types[fold.test] == t]
        rows.append({"entity_type": t, "n_test_nodes": int(len(sub)), **tm,
                     **auc_scores(labels.Y, out.score, labels.valid, sub)})
    return m, rows


def evaluate_baseline_cv(tikg: TIKG, labels: LabelSpace, variant: BaselineVariant, cfg: EvalConfig = EvalConfig(),
                         folds: Optional[List[Fold]] = None, dataset_info: Optional[Dict[str, Any]] = None,
                         verbose: bool = False) -> EvaluationResult:
    """Same outer folds, inner validation partitions, seeds and test campaigns as ``evaluate_cv``.  Both baselines are
    deterministic given the fold, so each fold is computed once and the (fold, seed) rows are paired with the GCN's."""
    from .evaluation import build_manifest
    from .protocol import enforce_launch_guard
    enforce_launch_guard(what="evaluate_baseline_cv", n_nodes=tikg.n, cfg=cfg, source=(dataset_info or {}).get("source"))
    folds = folds or group_stratified_folds(tikg, labels, cfg.n_splits, cfg.fold_seed, cfg.val_fraction, cfg.allow_ungrouped)
    use = folds[: cfg.max_folds] if cfg.max_folds else folds
    fold_info = [verify_fold(tikg, labels, f) for f in use]
    adj = undirected_adjacency(tikg)
    heads = type_heads(tikg, labels)
    runs, tr_rows, tim = [], [], []
    for f in use:
        masks = make_masks(labels, f.train, f.val, f.test)
        test_nodes = np.asarray(f.test_nodes_mask, dtype=bool)
        if (test_nodes & masks.visible).any():
            raise LabelLeakageError("a train/validation node lies in a test campaign")
        Yv = visible_labels(labels, masks)
        if variant.name == "attackg":
            out = run_attackg(tikg, labels, Yv, masks, test_nodes, adj, variant.config)
        elif variant.name == "ladder":
            out = run_ladder(tikg, labels, Yv, masks, variant.config)
        else:
            raise KeyError(variant.name)
        m, typed = _metrics(labels, tikg, out, f, heads, cfg.ks)
        for seed in cfg.seeds:
            runs.append({"fold": f.index, "seed": seed, "model": variant.name, "n_parameters": out.n_parameters, **m,
                         "decision_threshold": out.threshold, "val_macro_f1": out.val_macro_f1, **out.diagnostics,
                         "n_train": int(f.train.size), "n_train_used": int(masks.train.sum()),
                         "n_val": int(f.val.size), "n_test": int(f.test.size)})
            tr_rows += [{"fold": f.index, "seed": seed, **r} for r in typed]
            tim.append({"fold": f.index, "seed": seed, "prepare_fold_s": 0.0, "fit_predict_s": out.fit_time_s + out.inference_time_s,
                        "train_time_per_epoch_s": float("nan"), "train_time_total_s": out.fit_time_s,
                        "inference_time_s": out.inference_time_s})
        if verbose:
            print(f"[{variant.name} fold {f.index}] macro-F1={m['macro_f1']:.4f} micro-F1={m['micro_f1']:.4f}", flush=True)
    manifest = build_manifest(tikg, labels, cfg, fold_info, dataset_info, len(use), {}, None)
    for k in ("model", "chi_weights", "chi_structures", "graph_representation", "ranking", "paths"):
        manifest.pop(k, None)
    manifest["variant"] = variant.info()
    manifest["baseline"] = {"config": manifest["variant"]["config"], "components": COMPONENTS[variant.name],
                            "metric_support": metric_support(variant.name),
                            "label_term_in_text_rate": label_term_in_text_rate(tikg, labels),
                            "graph": ("native TIKG, undirected, held-out-campaign nodes isolated for template building"
                                      if variant.name == "attackg" else "none (text only)"),
                            "fit_inputs": "visible labels (train U validation), training nodes' text; test labels never passed",
                            "seed_invariant": True}
    manifest.pop("content_sha256", None)
    env, pre = manifest.pop("environment"), manifest.pop("preflight", None)
    manifest["content_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, default=str).encode()).hexdigest()
    manifest["environment"], manifest["preflight"] = env, pre
    return EvaluationResult(pd.DataFrame(runs), pd.DataFrame(tr_rows), use, manifest, pd.DataFrame(tim), pd.DataFrame())


def run_baseline_comparison(tikg: TIKG, labels: LabelSpace, baselines: Sequence[BaselineVariant],
                            cfg: EvalConfig = EvalConfig(), neural: Sequence[str] = (REFERENCE_MODEL,),
                            dataset_info: Optional[Dict[str, Any]] = None, provenance: Optional[Dict[str, Any]] = None,
                            folds: Optional[List[Fold]] = None, verbose: bool = False) -> AblationResult:
    """Reference GCN (and optional GAT / HGT) and the external baselines on IDENTICAL folds, validation partitions, seeds."""
    folds = folds or group_stratified_folds(tikg, labels, cfg.n_splits, cfg.fold_seed, cfg.val_fraction, cfg.allow_ungrouped)
    use = folds[: cfg.max_folds] if cfg.max_folds else folds
    info = {**(dataset_info or {}), "provenance": (provenance or {}).get("status")}
    neural = list(dict.fromkeys([REFERENCE_MODEL, *neural]))
    variants: List[Any] = [MODELS[n] for n in neural] + list(baselines)
    results: Dict[str, EvaluationResult] = {}
    for n in neural:
        mv = MODELS[n]
        results[n] = evaluate_cv(tikg, labels, replace(cfg, model=mv.config(cfg.model)), None, None, None, None, None,
                                 folds=folds, dataset_info=info, variant_info=mv.info())
    for b in baselines:
        results[b.name] = evaluate_baseline_cv(tikg, labels, b, cfg, folds, info, verbose)
    tag = lambda df, n: df.assign(variant=n) if len(df) else df
    cat = lambda attr: pd.concat([tag(getattr(r, attr), n) for n, r in results.items() if len(getattr(r, attr))],
                                 ignore_index=True)
    status = (provenance or {}).get("status", "unspecified")
    runs, by_type, timings = cat("runs"), cat("by_type"), cat("timings")
    for df in (runs, by_type, timings):
        df["data_provenance"] = status
    ptab = parameter_table(runs)
    manifest = build_ablation_manifest(results[REFERENCE_MODEL], variants, results, provenance, use, "external_baseline_comparison",
                                       REFERENCE_MODEL,
                                       {"parameter_counts": ptab.round(6).to_dict(orient="records"),
                                        "metric_support": {v.name: metric_support(v.name) for v in variants},
                                        "shared_across_methods": ["outer folds", "inner validation partitions", "seeds",
                                                                  "test campaigns", "labels", "metrics"]})
    return AblationResult(runs, by_type, pd.DataFrame(), timings, use, manifest, variants, results, REFERENCE_MODEL)


def run_baseline_comparison_dataset(ds, baselines: Sequence[str] = DEFAULT_BASELINES, neural: Sequence[str] = (REFERENCE_MODEL,),
                                    cfg: EvalConfig = EvalConfig(), protocol_frozen: bool = False,
                                    verbose: bool = False) -> AblationResult:
    from .protocol import require_preflight_if_marked
    require_preflight_if_marked("run_baseline_comparison_dataset", cfg, protocol_frozen=protocol_frozen)
    meta = ds.metadata or {}
    info = {"source": ds.source, "name": ds.name, "profile": meta.get("profile"), "dataset_type": meta.get("dataset_type"),
            "generator_seed": meta.get("seed"), "is_synthetic": bool(ds.is_synthetic)}
    return run_baseline_comparison(ds.tikg, ds.labels, select_baselines(baselines), cfg, neural, info,
                                   provenance_info(ds, protocol_frozen), verbose=verbose)


def paired_with_reasons(res: AblationResult) -> pd.DataFrame:
    """Paired table vs the reference GCN (raw Wilcoxon and Holm-adjusted p) with N/A reasons for unsupported metrics."""
    from .ablation import PAIRED_METRICS
    df = res.paired(PAIRED_METRICS)
    sup = res.manifest.get("metric_support", {})
    df["na_reason"] = [
        "" if r.n_pairs else (sup.get(r.variant, {}).get(r.metric, {}).get("reason")
                              or sup.get(res.reference, {}).get(r.metric, {}).get("reason") or "metric undefined for this run set")
        for r in df.itertuples()]
    return df


def write_baseline_comparison(out_dir, res: AblationResult):
    """``write_ablation`` files + model_parameters.csv + paired table with N/A reasons + metric_support.csv."""
    from pathlib import Path
    paths = write_ablation(out_dir, res)
    out = Path(out_dir)
    parameter_table(res.runs, res.reference).to_csv(out / "model_parameters.csv", index=False)
    paired_with_reasons(res).to_csv(out / "ablation_paired.csv", index=False, float_format="%.10g")
    rows = [{"method": v, "metric": m, **s} for v, d in res.manifest.get("metric_support", {}).items() for m, s in d.items()]
    pd.DataFrame(rows).to_csv(out / "metric_support.csv", index=False)
    paths["model_parameters.csv"], paths["metric_support.csv"] = out / "model_parameters.csv", out / "metric_support.csv"
    return paths
