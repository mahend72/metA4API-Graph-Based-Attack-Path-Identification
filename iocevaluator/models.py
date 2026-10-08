"""Graph models and the training loop (manuscript Sec. 3.5, 4.2, 4.3).

* ``GCN``  : Z = sigmoid(A^ ReLU(A^ F W0) W1)  (Eq. 4), A^ = D~^-1/2 (Adj + I) D~^-1/2; depth 1-4, no bias (Eq. 4),
             dropout 0.5.  Sigmoid + BCE (Eq. 5) is implemented as BCE-with-logits (numerically identical).
* ``GAT``  : edge-wise attention baseline over the observed graph.
* ``HGT``  : HGT-style heterogeneous attention (node-type Q/K/V, relation-specific attention/message matrices).
Both baselines are compact re-implementations (no torch_geometric dependency); they follow the original papers'
structure but are not guaranteed bit-identical to reference implementations.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .metrics import multilabel_prf


# ------------------------------------------------------------------------------------------------- graph inputs
def normalise_adjacency(adj: sp.spmatrix) -> sp.csr_matrix:
    """A^ = D~^-1/2 (A + I) D~^-1/2 with D~_ii = sum_j (A + I)_ij."""
    a = sp.csr_matrix(adj, dtype=np.float32)
    a = a - sp.diags(a.diagonal())
    a = a + sp.eye(a.shape[0], dtype=np.float32)
    d = np.asarray(a.sum(1)).ravel()
    dinv = sp.diags(1.0 / np.sqrt(d))
    return (dinv @ a @ dinv).tocsr()


def _to_torch_sparse(m: sp.spmatrix) -> torch.Tensor:
    c = m.tocoo()
    return torch.sparse_coo_tensor(np.vstack([c.row, c.col]), c.data.astype(np.float32), c.shape).coalesce()


@dataclass
class GraphData:
    A_norm: torch.Tensor                       # sparse normalised adjacency (GCN)
    src: torch.Tensor                          # edge list incl. self loops (GAT / HGT)
    dst: torch.Tensor
    rel: torch.Tensor                          # relation id per edge (HGT)
    n_rel: int
    node_type: torch.Tensor                    # type id per node (HGT)
    n_type: int


def make_graph_data(adj: sp.spmatrix, node_type_ids: np.ndarray,
                    rel_adjs: Optional[Dict[str, sp.spmatrix]] = None) -> GraphData:
    a = sp.csr_matrix(adj)
    coo = a.tocoo()
    n = a.shape[0]
    if rel_adjs:
        rs, rd, rr = [], [], []
        for r, (name, m) in enumerate(sorted(rel_adjs.items())):
            c = sp.csr_matrix(m).tocoo()
            rs += [c.row, c.col]; rd += [c.col, c.row]            # forward then reverse relation
            rr += [np.full(len(c.row), 2 * r), np.full(len(c.row), 2 * r + 1)]
        loop = np.arange(n)
        rs.append(loop); rd.append(loop); rr.append(np.full(n, 2 * len(rel_adjs)))
        src, dst, rel = np.concatenate(rs), np.concatenate(rd), np.concatenate(rr)
        n_rel = 2 * len(rel_adjs) + 1
    else:
        src = np.concatenate([coo.row, np.arange(n)])
        dst = np.concatenate([coo.col, np.arange(n)])
        rel = np.zeros(len(src), dtype=np.int64)
        n_rel = 1
    return GraphData(_to_torch_sparse(normalise_adjacency(a)), torch.as_tensor(src, dtype=torch.long),
                     torch.as_tensor(dst, dtype=torch.long), torch.as_tensor(rel, dtype=torch.long), n_rel,
                     torch.as_tensor(node_type_ids, dtype=torch.long), int(node_type_ids.max()) + 1)


def _scatter_softmax(score: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    shape = (n,) + score.shape[1:]
    mx = torch.full(shape, -1e30, dtype=score.dtype).scatter_reduce(
        0, index.view(-1, *[1] * (score.dim() - 1)).expand_as(score), score, "amax", include_self=True)
    ex = torch.exp(score - mx[index])
    den = torch.zeros(shape, dtype=score.dtype).index_add_(0, index, ex)
    return ex / (den[index] + 1e-16)


# ------------------------------------------------------------------------------------------------------ models
class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int = 2, dropout: float = 0.5):
        super().__init__()
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        self.W = nn.ParameterList([nn.Parameter(nn.init.xavier_uniform_(torch.empty(a, b)))
                                   for a, b in zip(dims[:-1], dims[1:])])
        self.dropout = dropout

    def forward(self, x: torch.Tensor, g: GraphData) -> torch.Tensor:
        for i, W in enumerate(self.W):
            x = Fn.dropout(x, self.dropout, self.training)
            x = torch.sparse.mm(g.A_norm, x @ W)
            if i < len(self.W) - 1:
                x = Fn.relu(x)
        return x  # logits; sigmoid(logits) = Z of Eq. 4


class _GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads, concat=True):
        super().__init__()
        assert out_dim % heads == 0 or not concat
        self.h, self.c, self.concat = heads, (out_dim // heads if concat else out_dim), concat
        self.W = nn.Linear(in_dim, heads * self.c, bias=False)
        self.a_src = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, heads, self.c)))
        self.a_dst = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, heads, self.c)))

    def forward(self, x, g: GraphData):
        n = x.shape[0]
        h = self.W(x).view(n, self.h, self.c)
        e = Fn.leaky_relu((h * self.a_src).sum(-1)[g.src] + (h * self.a_dst).sum(-1)[g.dst], 0.2)   # [E,H]
        a = _scatter_softmax(e, g.dst, n)
        out = torch.zeros(n, self.h, self.c).index_add_(0, g.dst, a.unsqueeze(-1) * h[g.src])
        return out.reshape(n, -1) if self.concat else out.mean(1)


class GAT(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, heads=4, dropout=0.5, **_):
        super().__init__()
        self.l1, self.l2, self.dropout = _GATLayer(in_dim, hidden, heads), _GATLayer(hidden, out_dim, 1, concat=False), dropout

    def forward(self, x, g):
        x = Fn.dropout(x, self.dropout, self.training)
        x = Fn.elu(self.l1(x, g))
        return self.l2(Fn.dropout(x, self.dropout, self.training), g)


class _HGTLayer(nn.Module):
    def __init__(self, dim, heads, n_type, n_rel):
        super().__init__()
        assert dim % heads == 0
        self.h, self.d, self.n_rel = heads, dim // heads, n_rel
        self.q = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_type)])
        self.k = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_type)])
        self.v = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_type)])
        self.o = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_type)])
        self.W_att = nn.Parameter(torch.stack([torch.eye(self.d).repeat(heads, 1, 1) for _ in range(n_rel)]))
        self.W_msg = nn.Parameter(torch.stack([torch.eye(self.d).repeat(heads, 1, 1) for _ in range(n_rel)]))
        self.mu = nn.Parameter(torch.ones(n_rel, heads))
        self.skip = nn.Parameter(torch.ones(n_type))

    def _typed(self, lins, x, tid):
        out = torch.zeros_like(x)
        for t, lin in enumerate(lins):
            m = tid == t
            if m.any():
                out[m] = lin(x[m])
        return out

    def forward(self, x, g: GraphData):
        n = x.shape[0]
        Q = self._typed(self.q, x, g.node_type).view(n, self.h, self.d)
        K = self._typed(self.k, x, g.node_type).view(n, self.h, self.d)
        Vv = self._typed(self.v, x, g.node_type).view(n, self.h, self.d)
        E = g.src.shape[0]
        score = torch.zeros(E, self.h)
        msg = torch.zeros(E, self.h, self.d)
        for r in range(self.n_rel):
            m = g.rel == r
            if not m.any():
                continue
            s, t = g.src[m], g.dst[m]
            k = torch.einsum("ehd,hdf->ehf", K[s], self.W_att[r])
            score[m] = (Q[t] * k).sum(-1) * self.mu[r] / (self.d ** 0.5)
            msg[m] = torch.einsum("ehd,hdf->ehf", Vv[s], self.W_msg[r])
        a = _scatter_softmax(score, g.dst, n)
        agg = torch.zeros(n, self.h, self.d).index_add_(0, g.dst, a.unsqueeze(-1) * msg).reshape(n, -1)
        out = self._typed(self.o, Fn.gelu(agg), g.node_type)
        alpha = torch.sigmoid(self.skip)[g.node_type].unsqueeze(-1)
        return alpha * out + (1 - alpha) * x


class HGT(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, n_type, n_rel, heads=4, dropout=0.5, **_):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([_HGTLayer(hidden, heads, n_type, n_rel) for _ in range(2)])
        self.out = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, g):
        x = Fn.relu(self.inp(Fn.dropout(x, self.dropout, self.training)))
        for l in self.layers:
            x = Fn.dropout(l(x, g), self.dropout, self.training)
        return self.out(x)


def build_model(kind: str, in_dim: int, out_dim: int, g: GraphData, hidden: int = 64, dropout: float = 0.5,
                layers: int = 2) -> nn.Module:
    if kind == "gcn":
        return GCN(in_dim, hidden, out_dim, layers=layers, dropout=dropout)
    if kind == "gat":
        return GAT(in_dim, hidden, out_dim, dropout=dropout)
    if kind == "hgt":
        return HGT(in_dim, hidden, out_dim, g.n_type, g.n_rel, dropout=dropout)
    raise ValueError(kind)


# ----------------------------------------------------------------------------------------------- training loop
@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 5e-4
    max_epochs: int = 500
    patience: int = 20
    hidden: int = 64
    dropout: float = 0.5
    layers: int = 2
    threshold: float = 0.5


@dataclass
class FitResult:
    model: nn.Module
    best_epoch: int
    epochs_run: int
    epoch_times: List[float]
    val_history: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)

    @property
    def total_time(self) -> float:
        return float(sum(self.epoch_times))


def masked_bce(logits, Y, valid, idx) -> torch.Tensor:
    """Eq. 5 restricted to labelled nodes `idx` and to labels valid for each node's type (mean over entries)."""
    l = Fn.binary_cross_entropy_with_logits(logits[idx], Y[idx], reduction="none")
    v = valid[idx].float()
    return (l * v).sum() / v.sum().clamp(min=1.0)


def fit(kind: str, x: np.ndarray, g: GraphData, Y: np.ndarray, valid: np.ndarray, train_idx: np.ndarray,
        val_idx: np.ndarray, cfg: TrainConfig, seed: int) -> FitResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    xt = torch.as_tensor(x, dtype=torch.float32)
    Yt = torch.as_tensor(Y, dtype=torch.float32)
    vt = torch.as_tensor(valid)
    model = build_model(kind, x.shape[1], Y.shape[1], g, cfg.hidden, cfg.dropout, cfg.layers)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    tr = torch.as_tensor(train_idx, dtype=torch.long)
    best, best_vloss = -1.0, float("inf")
    best_state, best_ep, bad = copy.deepcopy(model.state_dict()), 0, 0
    times, vh, lh = [], [], []
    vi = torch.as_tensor(val_idx, dtype=torch.long)
    for ep in range(1, cfg.max_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        opt.zero_grad()
        loss = masked_bce(model(xt, g), Yt, vt, tr)
        loss.backward()
        opt.step()
        times.append(time.perf_counter() - t0)
        lh.append(float(loss.detach()))
        if len(val_idx):
            model.eval()
            with torch.no_grad():
                logits = model(xt, g)
                vloss = float(masked_bce(logits, Yt, vt, vi))
                pred = (torch.sigmoid(logits) > cfg.threshold).numpy()
            score = multilabel_prf(Y, pred, valid, val_idx)["macro_f1"]          # early stopping on val Macro-F1
        else:
            score, vloss = -float(loss.detach()), float(loss.detach())
        vh.append(score)
        # improvement = higher val Macro-F1; ties (e.g. F1 == 0 early in training) are broken by lower val loss so
        # the patience counter does not expire before the model has started to learn.
        if score > best + 1e-12 or (abs(score - best) <= 1e-12 and vloss < best_vloss - 1e-9):
            best, best_vloss, best_state, best_ep, bad = score, vloss, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return FitResult(model, best_ep, ep, times, vh, lh)


def predict_proba(model: nn.Module, x: np.ndarray, g: GraphData) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(x, dtype=torch.float32), g)).numpy()
