from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def normalize_adjacency(adj: np.ndarray, add_self_loops: bool = True, eps: float = 1e-12) -> np.ndarray:
    """Compute D^{-1/2} (A + I) D^{-1/2}."""
    A = adj.astype(np.float32, copy=True)
    if add_self_loops:
        np.fill_diagonal(A, 1.0)
    deg = A.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(deg + eps)
    D_inv = np.diag(inv_sqrt)
    return D_inv @ A @ D_inv


@dataclass(frozen=True)
class GCNArtifacts:
    Z: np.ndarray  # node embeddings [N, d]
    recon: np.ndarray  # reconstructed adjacency [N, N]
    loss_history: list[float]


def run_gcn_autoencoder(
    adj: np.ndarray,
    X: np.ndarray,
    hidden_dim: int = 64,
    epochs: int = 50,
    lr: float = 1e-3,
    seed: int = 7,
    device: str = "cpu",
) -> GCNArtifacts:
    """A small GCN autoencoder for unsupervised node embeddings.

    - Encoder: 2-layer GCN
    - Decoder: sigmoid(ZZ^T) adjacency reconstruction
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as e:
        raise RuntimeError(
            "PyTorch is required for GCN. Install with: pip install .[gcn]"
        ) from e

    torch.manual_seed(seed)
    A = torch.tensor(normalize_adjacency(adj), dtype=torch.float32, device=device)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    target = torch.tensor((adj > 0).astype(np.float32), dtype=torch.float32, device=device)

    class GCNLayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            self.lin = nn.Linear(in_dim, out_dim)
        def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
            return self.lin(A @ x)

    class GAE(nn.Module):
        def __init__(self, in_dim: int, hid: int, out_dim: int) -> None:
            super().__init__()
            self.g1 = GCNLayer(in_dim, hid)
            self.g2 = GCNLayer(hid, out_dim)
        def encode(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
            h = F.relu(self.g1(x, A))
            z = self.g2(h, A)
            return z
        def forward(self, x: torch.Tensor, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            z = self.encode(x, A)
            recon = torch.sigmoid(z @ z.T)
            return z, recon

    out_dim = X.shape[1]  # keep embeddings aligned to feature dimension
    model = GAE(X.shape[1], hidden_dim, out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses: list[float] = []
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        z, recon = model(X_t, A)
        loss = F.binary_cross_entropy(recon, target)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            log.info("GCN epoch %d/%d | loss=%.4f", ep, epochs, losses[-1])

    model.eval()
    with torch.no_grad():
        z, recon = model(X_t, A)

    return GCNArtifacts(
        Z=z.cpu().numpy(),
        recon=recon.cpu().numpy(),
        loss_history=losses,
    )
