"""Supervised multi-label GCN of the manuscript (Sec. 3.5 Eq. 4-5, Sec. 4.2, Sec. 4.3).

    Z = sigmoid( A^ ReLU( A^ F W0 ) W1 ),   A^ = D~^-1/2 (Adj + I) D~^-1/2,   Adj = MeGiTS adjacency (chi_1..chi_20)
    H = - sum_{i in y_L} sum_k [ l_k ln Z_k + (1 - l_k) ln (1 - Z_k) ]           (binary cross-entropy, Eq. 5)

This module is separate from the optional unsupervised GCN autoencoder in ``gcn.py`` (which is untouched).  It reuses
the Eq.-4 layer stack (``models.GCN``), the adjacency normalisation, the masked BCE and the metrics, and adds what the
manuscript's protocol needs around them:

  * default hyper-parameters exactly as in Sec. 4.2 (2 layers, hidden 64, dropout 0.5, Adam lr 1e-3, L2 5e-4,
    early stopping on validation Macro-F1 with patience 20); depth 1-4 selectable through ``layers``;
  * heterogeneous label spaces: one global sigmoid output layer with K = sum_t K_t columns, but every node only
    owns the columns of its entity type (``LabelSpace.valid``).  The loss, the early-stopping metric and the
    predictions are restricted to those columns, so no type ever trains or predicts a class of another type.
    Per-type views (``predict_by_type``) expose the K_t-dimensional output of each entity type;
  * label-leakage guards: the trainer only accepts a label matrix in which every row outside train U validation is
    zero (``visible_labels``); disjoint train / validation / test masks are enforced; test labels are touched only
    by the scoring function, never by training or model selection;
  * seed determinism (Python / NumPy / PyTorch RNGs, deterministic algorithms).

Inductive vs transductive (w.r.t. held-out campaigns).  TRAINING is inductive: the training graph, the MeGiTS
statistics, the feature scaler and every loss / early-stopping computation contain no held-out-campaign node, edge,
feature or label.  PREDICTION is transductive in the topology of the held-out campaigns: at inference the full graph
(including the test campaigns' nodes and relations, but no labels) is used for the MeGiTS adjacency and for message
passing, i.e. the manuscript's "semi-inductive" protocol (Sec. 4.2: "test nodes are introduced only during
inference").  ``FoldContext.train_graph`` / ``infer_graph`` are the two graphs.
"""
from __future__ import annotations

import copy
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as Fn
from sklearn.model_selection import GroupShuffleSplit

from .labels import LabelSpace
from .megits import fold_megits_adjacency
from .metrics import multilabel_prf, per_type_prf
from .models import GAT, GCN, HGT, GraphData, make_graph_data, masked_bce
from .splits import Fold
from .tikg import TIKG, EntityType
from .tikg_features import FeatureBuilder, FeatureConfig


class LabelLeakageError(ValueError):
    """Raised when train / validation / test masks overlap or labels outside train U validation reach the trainer."""


# ------------------------------------------------------------------------------------------------ configuration
@dataclass(frozen=True)
class SupervisedGCNConfig:
    """Defaults = manuscript Sec. 4.2.  ``layers`` in {1, 2, 3, 4}; the default is the manuscript's 2-layer model."""
    layers: int = 2
    hidden: int = 64
    dropout: float = 0.5
    lr: float = 1e-3
    weight_decay: float = 5e-4          # L2 = 5e-4 (Adam weight_decay, all weights)
    patience: int = 20                  # early stopping on validation Macro-F1
    max_epochs: int = 500               # not stated in the manuscript
    threshold: float = 0.5              # per-label decision threshold on the sigmoid output
    model: str = "gcn"                  # "gcn" (the manuscript model) | "gat" | "hgt" (comparison baselines)
    heads: int = 4                      # attention heads (gat / hgt only; ``layers`` is ignored by gat / hgt)

    def __post_init__(self):
        if self.layers not in (1, 2, 3, 4):
            raise ValueError("layers must be 1, 2, 3 or 4 (Sec. 4.3)")
        if self.model not in ("gcn", "gat", "hgt"):
            raise ValueError("model must be 'gcn', 'gat' or 'hgt'")
        if self.model != "gcn" and self.hidden % self.heads:
            raise ValueError("hidden must be divisible by heads for gat / hgt")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


# ------------------------------------------------------------------------------------------------ determinism
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@contextmanager
def deterministic(seed: int):
    """Seed all RNGs and force deterministic kernels for the duration of a fit (previous setting is restored)."""
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    seed_everything(seed)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev)


# ------------------------------------------------------------------------------------------------ masks / leakage
@dataclass(frozen=True)
class SplitMasks:
    train: np.ndarray       # boolean masks over all N nodes
    val: np.ndarray
    test: np.ndarray

    @property
    def visible(self) -> np.ndarray:
        """Nodes whose labels the trainer may see (train for the loss, validation for early stopping)."""
        return self.train | self.val


def make_masks(labels: LabelSpace, train: Sequence[int], val: Sequence[int], test: Sequence[int]) -> SplitMasks:
    """Boolean train / validation / test masks; raises ``LabelLeakageError`` on overlap or unlabelled nodes."""
    n = labels.Y.shape[0]
    out = []
    for name, idx in (("train", train), ("val", val), ("test", test)):
        m = np.zeros(n, dtype=bool)
        m[np.asarray(idx, dtype=int)] = True
        if (m & ~labels.labelled).any():
            raise LabelLeakageError(f"{name} contains unlabelled nodes")
        out.append(m)
    tr, va, te = out
    if (tr & va).any() or (tr & te).any() or (va & te).any():
        raise LabelLeakageError("train / validation / test masks must be disjoint")
    if not tr.any():
        raise LabelLeakageError("empty training mask")
    if not va.any():
        raise LabelLeakageError("an inner validation split is required for early stopping (patience 20)")
    return SplitMasks(tr, va, te)


def visible_labels(labels: LabelSpace, masks: SplitMasks) -> np.ndarray:
    """Copy of Y in which every row outside train U validation is zero: test (and unlabelled) labels never reach
    the trainer."""
    y = np.zeros_like(labels.Y)
    y[masks.visible] = labels.Y[masks.visible]
    return y


def inner_validation_split(tikg: TIKG, train_idx: np.ndarray, val_fraction: float = 0.10, seed: int = 0):
    """Carve a validation set out of the OUTER-TRAIN nodes only (Sec. 4.2: "10% validation carve-out"),
    campaign-isolated when campaigns are known.  Returns (train, val) index arrays."""
    train_idx = np.asarray(train_idx)
    groups = tikg.campaigns()[train_idx]
    if (groups == "").any() or len(set(groups)) < 2:
        groups = np.arange(len(train_idx))
    a, b = next(GroupShuffleSplit(1, test_size=val_fraction, random_state=seed).split(train_idx, groups=groups))
    return np.sort(train_idx[a]), np.sort(train_idx[b])


# ------------------------------------------------------------------------------------------------ typed outputs
@dataclass(frozen=True)
class TypeHead:
    """Output block of one entity type: its nodes and its K_t label columns in the global output layer."""
    entity_type: EntityType
    nodes: np.ndarray           # global node indices of this type
    columns: np.ndarray         # global output columns owned by this type
    classes: List[str]

    @property
    def out_dim(self) -> int:
        return len(self.columns)


def type_heads(tikg: TIKG, labels: LabelSpace) -> Dict[EntityType, TypeHead]:
    heads = {}
    for t in EntityType:
        cols = labels.type_columns(t)
        heads[t] = TypeHead(t, tikg.type_nodes[t], cols, list(labels.classes[t]))
        assert len(cols) == len(labels.classes[t])
    return heads


@dataclass
class TypedPrediction:
    nodes: np.ndarray           # node indices of the type
    classes: List[str]
    prob: np.ndarray            # [n_nodes_t, K_t], in [0, 1]


class MultiLabelGCN(GCN):
    """Eq. 4: sigmoid(A^ ReLU(A^ F W0) W1) for 2 layers (no biases); ``layers`` 1..4 stacks A^ . W per layer.
    The output layer has K = sum_t K_t columns; ``valid`` (N x K) marks the columns each node's type owns."""

    def __init__(self, in_dim: int, out_dim: int, valid: np.ndarray, cfg: SupervisedGCNConfig = SupervisedGCNConfig()):
        super().__init__(in_dim, cfg.hidden, out_dim, layers=cfg.layers, dropout=cfg.dropout)
        self.register_buffer("valid", torch.as_tensor(valid, dtype=torch.bool))

    def proba(self, x: torch.Tensor, g: GraphData) -> torch.Tensor:
        """Sigmoid probabilities; columns that do not belong to a node's type are exactly 0."""
        return torch.sigmoid(self.forward(x, g)) * self.valid.float()


class MultiLabelGAT(GAT):
    """Graph-attention baseline (2 layers, ELU, multi-head attention over the SAME graph the GCN receives)."""

    def __init__(self, in_dim: int, out_dim: int, valid: np.ndarray, cfg: SupervisedGCNConfig = SupervisedGCNConfig(model="gat")):
        super().__init__(in_dim, cfg.hidden, out_dim, heads=cfg.heads, dropout=cfg.dropout)
        self.register_buffer("valid", torch.as_tensor(valid, dtype=torch.bool))

    def proba(self, x: torch.Tensor, g: GraphData) -> torch.Tensor:
        return torch.sigmoid(self.forward(x, g)) * self.valid.float()


class MultiLabelHGT(HGT):
    """HGT-style baseline: node-type-specific Q/K/V/output maps and relation-specific attention / message matrices over
    the native TIKG entity and relation types."""

    def __init__(self, in_dim: int, out_dim: int, valid: np.ndarray, cfg: SupervisedGCNConfig, n_type: int, n_rel: int):
        super().__init__(in_dim, cfg.hidden, out_dim, n_type, n_rel, heads=cfg.heads, dropout=cfg.dropout)
        self.register_buffer("valid", torch.as_tensor(valid, dtype=torch.bool))

    def proba(self, x: torch.Tensor, g: GraphData) -> torch.Tensor:
        return torch.sigmoid(self.forward(x, g)) * self.valid.float()


def build_supervised_model(cfg: SupervisedGCNConfig, in_dim: int, out_dim: int, valid: np.ndarray, graph: GraphData):
    if cfg.model == "gcn":
        return MultiLabelGCN(in_dim, out_dim, valid, cfg)
    if cfg.model == "gat":
        return MultiLabelGAT(in_dim, out_dim, valid, cfg)
    return MultiLabelHGT(in_dim, out_dim, valid, cfg, graph.n_type, graph.n_rel)


def count_parameters(model: torch.nn.Module) -> int:
    """Number of trainable parameters (buffers such as the label-validity mask are not counted)."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def parameter_breakdown(model: torch.nn.Module) -> Dict[str, int]:
    return {n: int(p.numel()) for n, p in model.named_parameters() if p.requires_grad}


# ------------------------------------------------------------------------------------------------ trainer
@dataclass
class SupervisedFit:
    model: MultiLabelGCN
    cfg: SupervisedGCNConfig
    best_epoch: int
    epochs_run: int
    stopped_early: bool
    loss_history: List[float]               # training BCE per epoch
    val_loss_history: List[float]
    val_macro_f1_history: List[float]
    epoch_times: List[float] = field(default_factory=list)
    n_parameters: int = 0

    def predict(self, x: np.ndarray, graph: GraphData) -> np.ndarray:
        """N x K probabilities in [0, 1] (exactly 0 for labels outside the node's entity type)."""
        self.model.eval()
        with torch.no_grad():
            return self.model.proba(torch.as_tensor(x, dtype=torch.float32), graph).numpy()

    def predict_by_type(self, x: np.ndarray, graph: GraphData, heads: Dict[EntityType, TypeHead]) -> Dict[EntityType, TypedPrediction]:
        p = self.predict(x, graph)
        return {t: TypedPrediction(h.nodes, h.classes, p[np.ix_(h.nodes, h.columns)]) for t, h in heads.items()}


def fit_supervised(x: np.ndarray, graph: GraphData, Y_visible: np.ndarray, valid: np.ndarray, masks: SplitMasks,
                   cfg: SupervisedGCNConfig = SupervisedGCNConfig(), seed: int = 0) -> SupervisedFit:
    """Train on ``masks.train`` (BCE, Eq. 5, type-valid labels only); early-stop on ``masks.val`` Macro-F1.

    ``Y_visible`` must come from :func:`visible_labels`: any label outside train U validation is rejected."""
    hidden_rows = ~masks.visible
    if np.asarray(Y_visible)[hidden_rows].any():
        raise LabelLeakageError("Y_visible contains labels of nodes outside train U validation (use visible_labels())")
    if x.shape[0] != Y_visible.shape[0] or graph.A_norm.shape[0] != x.shape[0]:
        raise ValueError("feature matrix, labels and adjacency must have the same number of nodes")
    tr = torch.as_tensor(np.flatnonzero(masks.train), dtype=torch.long)
    va_np = np.flatnonzero(masks.val)
    va = torch.as_tensor(va_np, dtype=torch.long)
    xt = torch.as_tensor(x, dtype=torch.float32)
    Yt = torch.as_tensor(Y_visible, dtype=torch.float32)
    vt = torch.as_tensor(valid)
    with deterministic(seed):
        model = build_supervised_model(cfg, x.shape[1], Y_visible.shape[1], valid, graph)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        best, best_vloss, best_ep, bad = -1.0, float("inf"), 0, 0
        best_state = copy.deepcopy(model.state_dict())
        lh, vlh, vfh, times = [], [], [], []
        stopped = False
        for ep in range(1, cfg.max_epochs + 1):
            t0 = time.perf_counter()
            model.train()
            opt.zero_grad()
            loss = masked_bce(model(xt, graph), Yt, vt, tr)
            loss.backward()
            opt.step()
            times.append(time.perf_counter() - t0)
            lh.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                logits = model(xt, graph)
                vloss = float(masked_bce(logits, Yt, vt, va))
                pred = (model.proba(xt, graph) > cfg.threshold).numpy()
            vf1 = multilabel_prf(Y_visible, pred, valid, va_np)["macro_f1"]
            vlh.append(vloss)
            vfh.append(vf1)
            # improvement = higher validation Macro-F1; ties (e.g. F1 == 0 early on) go to the lower validation loss
            if vf1 > best + 1e-12 or (abs(vf1 - best) <= 1e-12 and vloss < best_vloss - 1e-9):
                best, best_vloss, best_ep, bad = vf1, vloss, ep, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= cfg.patience:
                    stopped = True
                    break
        model.load_state_dict(best_state)
        model.eval()
    return SupervisedFit(model, cfg, best_ep, ep, stopped, lh, vlh, vfh, times, count_parameters(model))


# ------------------------------------------------------------------------------------------------ fold pipeline
def isolate_nodes(adj: sp.spmatrix, mask: np.ndarray) -> sp.csr_matrix:
    """Adjacency with every edge touching a masked node removed (the masked nodes become isolated)."""
    keep = sp.diags((~np.asarray(mask, dtype=bool)).astype(np.float32))
    out = (keep @ sp.csr_matrix(adj, dtype=np.float32) @ keep).tocsr()
    out.eliminate_zeros()
    return out


@dataclass
class FoldContext:
    """Everything an outer fold derives from its OUTER-TRAIN partition (reused by all seeds of the fold)."""
    masks: SplitMasks
    x: np.ndarray                    # features, scaler / vocabularies fitted on train U val only
    train_graph: GraphData           # inductive training: test nodes are isolated while training / early stopping
    infer_graph: GraphData           # full graph, used only to predict (held-out topology enters at inference)
    train_adj: sp.csr_matrix         # raw MeGiTS adjacency of the training graph (test nodes isolated)
    infer_adj: sp.csr_matrix         # raw MeGiTS adjacency of the inference graph (no labels involved)
    Y_visible: np.ndarray            # labels of train U val only
    heads: Dict[EntityType, TypeHead]
    feature_names: List[str]


def prepare_fold(tikg: TIKG, labels: LabelSpace, fold: Fold, feature_cfg: Optional[FeatureConfig] = None,
                 adj: Optional[sp.spmatrix] = None, model: str = "gcn") -> FoldContext:
    """Recompute every train-dependent artefact from the fold's outer-train data (Sec. 4.2 "Leakage control"):
    feature scaling / vocabularies (training nodes only, not validation), chi_1..chi_20 commuting matrices and MeGiTS similarities (training graph =
    triplets that touch no test-campaign node), normalised adjacency.  Test nodes are added back only for inference.

    ``adj``: optional pre-built N x N inference adjacency (default: the fold's MeGiTS adjacency over chi_1..chi_20);
    the training graph is always that adjacency with the test nodes isolated.

    ``model``: "gcn" and "gat" receive the same graph (the MeGiTS adjacency; GAT uses its support as the edge set);
    "hgt" receives the native TIKG instead: node-type ids and one forward + one reverse relation per triplet signature,
    with the held-out nodes' relations removed from the training graph exactly like the MeGiTS edges."""
    masks = make_masks(labels, fold.train, fold.val, fold.test)
    test_nodes = np.asarray(fold.test_nodes_mask, dtype=bool)
    if (test_nodes & masks.visible).any():
        raise LabelLeakageError("a train/validation node lies in a test campaign")
    fb = FeatureBuilder(feature_cfg or FeatureConfig())
    x = fb.fit_transform(tikg, np.flatnonzero(masks.train))                    # scaler / vocabularies: TRAIN nodes only
    if adj is None:
        adj = fold_megits_adjacency(tikg, test_nodes).adj
    order = {t.value: i for i, t in enumerate(EntityType)}
    tids = np.array([order[t] for t in tikg.types])
    train_adj = isolate_nodes(adj, test_nodes)
    if model == "hgt":
        rel_full = tikg.relation_adjacencies()
        rel_train = {k: isolate_nodes(m, test_nodes) for k, m in rel_full.items()}
        g_train, g_infer = make_graph_data(train_adj, tids, rel_train), make_graph_data(adj, tids, rel_full)
    else:
        g_train, g_infer = make_graph_data(train_adj, tids), make_graph_data(adj, tids)
    return FoldContext(masks, x, g_train, g_infer,
                       sp.csr_matrix(train_adj), sp.csr_matrix(adj), visible_labels(labels, masks),
                       type_heads(tikg, labels), list(fb.names))


def label_coverage(labels: LabelSpace, idx: np.ndarray) -> Dict[str, int]:
    """Labels (valid columns) with at least one positive among ``idx``."""
    idx = np.asarray(idx, dtype=int)
    pos = (labels.Y[idx] > 0) & labels.valid[idx]
    return {"n_labels_with_positive": int((pos.sum(0) > 0).sum()), "n_positive_pairs": int(pos.sum())}


def label_distribution_tv(labels: LabelSpace, idx: np.ndarray, ref_idx: np.ndarray) -> float:
    """Total-variation distance between the normalised positive-label frequencies of ``idx`` and ``ref_idx``."""
    def freq(i):
        pos = ((labels.Y[np.asarray(i, dtype=int)] > 0) & labels.valid[np.asarray(i, dtype=int)]).sum(0).astype(float)
        return pos / pos.sum() if pos.sum() else pos
    return float(0.5 * np.abs(freq(idx) - freq(ref_idx)).sum())


def label_preserving_order(labels: LabelSpace, train_idx: np.ndarray, seed_seq: Sequence[int]) -> np.ndarray:
    """Deterministic ordering of the training nodes whose every PREFIX preserves the label coverage and distribution.

    1. coverage phase: greedy set cover over the labels that occur in ``train_idx`` - rarest label first, pick the node
       covering most still-uncovered labels (ties: seeded random rank) - so a prefix covers as many labels as its size allows;
    2. stratified phase: the remaining nodes are interleaved by stratum (each node's rarest label): node r of a
       stratum of n gets the key (r + u_s) / n with a seeded offset u_s, so every prefix of fraction f holds ~f of each
       stratum.
    Prefixes of one ordering are nested by construction (25% subset of 50% subset of 75% subset ...)."""
    train = np.asarray(train_idx, dtype=int)
    rng = np.random.RandomState(int(np.random.SeedSequence([int(s) for s in seed_seq]).generate_state(1)[0]))
    perm = rng.permutation(len(train))
    pos = np.empty(len(train), dtype=int)
    pos[perm] = np.arange(len(train))                                   # seeded random rank of every node
    Y = (labels.Y[train] > 0) & labels.valid[train]
    support = Y.sum(0)
    covered = np.zeros(Y.shape[1], dtype=bool)
    cover: List[int] = []
    for k in np.argsort(support, kind="stable"):                        # rarest label first
        if support[k] == 0 or covered[k]:
            continue
        cand = np.flatnonzero(Y[:, k])
        gain = (Y[cand] & ~covered).sum(1)
        best = int(cand[np.lexsort((pos[cand], -gain))[0]])
        cover.append(best)
        covered |= Y[best]
    in_cover = np.zeros(len(train), dtype=bool)
    in_cover[cover] = True
    prim = labels.primary_label()[train]
    keys = np.zeros(len(train))
    for s in np.unique(prim[~in_cover]):
        members = np.flatnonzero((prim == s) & ~in_cover)
        members = members[np.argsort(pos[members], kind="stable")]
        u = rng.rand()
        keys[members] = (np.arange(len(members)) + u) / len(members)
    rest = np.flatnonzero(~in_cover)
    rest = rest[np.lexsort((pos[rest], keys[rest]))]
    return train[np.concatenate([np.array(cover, dtype=int), rest])]


def reduce_training_labels(ctx: FoldContext, labels: LabelSpace, fold: Fold, seed: int, fraction: float,
                           base_seed: int = 0) -> FoldContext:
    """Reduced-label training (ablation): keep the labels of a ``fraction`` of the labelled TRAINING nodes.

    * the kept nodes are the first ``max(1, round(fraction * n))`` of ``label_preserving_order`` - a deterministic,
      seeded ordering (per base_seed, fold, seed) that preserves label coverage and distribution where feasible, so
      the 25% / 50% / 75% subsets are nested and reproducible;
    * only the training partition is subsampled - validation and test masks and labels are untouched, so campaign
      isolation is preserved (a subset of the training campaigns' nodes);
    * the dropped nodes stay in the graph with their features (only their labels are withheld), and the feature scaler
      and MeGiTS statistics stay those of the full outer-train partition, so variants are paired;
    * coverage is guaranteed only as far as the subset size allows (a 25% subset can hold fewer nodes than there are labels)."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if fraction >= 1.0:
        return ctx
    train = np.flatnonzero(ctx.masks.train)
    order = label_preserving_order(labels, train, [base_seed, fold.index, seed, 7919])
    keep = np.sort(order[: max(1, int(round(fraction * len(train))))])
    m = np.zeros_like(ctx.masks.train)
    m[keep] = True
    masks = SplitMasks(m, ctx.masks.val, ctx.masks.test)
    y = visible_labels(labels, masks)
    if not np.array_equal(y[masks.val], ctx.Y_visible[masks.val]) or y[masks.test].any():
        raise LabelLeakageError("reduced-label sampling changed validation / test labels")
    return replace(ctx, masks=masks, Y_visible=y)


def fit_predict(ctx: FoldContext, labels: LabelSpace, cfg: SupervisedGCNConfig, seed: int):
    """Train on the training graph, predict on the inference graph.  Returns (SupervisedFit, N x K probabilities)."""
    res = fit_supervised(ctx.x, ctx.train_graph, ctx.Y_visible, labels.valid, ctx.masks, cfg, seed)
    return res, res.predict(ctx.x, ctx.infer_graph)


@dataclass
class FoldRun:
    fit: SupervisedFit
    prob: np.ndarray                         # N x K
    heads: Dict[EntityType, TypeHead]
    test_metrics: Dict[str, float]           # Macro/Micro P/R/F1 over the test nodes
    test_metrics_by_type: Dict[str, Dict[str, float]]
    in_dim: int


def run_supervised_fold(tikg: TIKG, labels: LabelSpace, fold: Fold, cfg: SupervisedGCNConfig = SupervisedGCNConfig(),
                        feature_cfg: Optional[FeatureConfig] = None, adj: Optional[sp.spmatrix] = None,
                        seed: int = 0) -> FoldRun:
    """One outer fold, leakage-safe: see :func:`prepare_fold`.  Test labels are used solely to score the predictions."""
    ctx = prepare_fold(tikg, labels, fold, feature_cfg, adj)
    res, prob = fit_predict(ctx, labels, cfg, seed)
    pred = prob > cfg.threshold
    type_cols = {t.value: h.columns for t, h in ctx.heads.items()}
    return FoldRun(res, prob, ctx.heads, multilabel_prf(labels.Y, pred, labels.valid, fold.test),
                   per_type_prf(labels.Y, pred, labels.valid, fold.test, type_cols, tikg.types), ctx.x.shape[1])


def dimension_report(tikg: TIKG, labels: LabelSpace, in_dim: int, cfg: SupervisedGCNConfig = SupervisedGCNConfig()
                     ) -> List[dict]:
    """Input / hidden / output dimensions per entity type (the weights are shared; the output columns are type-owned)."""
    rows = []
    for t, h in type_heads(tikg, labels).items():
        rows.append({"entity_type": t.value, "n_nodes": len(h.nodes), "input_dim": in_dim,
                     "hidden_dims": [cfg.hidden] * (cfg.layers - 1), "output_dim": h.out_dim,
                     "output_columns": (int(h.columns[0]), int(h.columns[-1])) if h.out_dim else None})
    return rows
