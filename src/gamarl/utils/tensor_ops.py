from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    """Softmax with a {0,1} mask."""
    mask = mask.to(dtype=logits.dtype)
    logits = logits - logits.max(dim=dim, keepdim=True).values
    exp = torch.exp(logits) * mask
    denom = exp.sum(dim=dim, keepdim=True).clamp_min(eps)
    return exp / denom


def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1.0, hard: bool = False, eps: float = 1e-6) -> torch.Tensor:
    """Binary Concrete (a.k.a. Gumbel-Sigmoid).

    For the binary case, the Concrete relaxation uses *Logistic* noise:
        l = log(u) - log(1-u),  u ~ Uniform(0,1)
        y = sigmoid((logits + l) / tau)

    This is equivalent to the difference of two i.i.d. Gumbel variables and is the standard,
    numerically-stable formulation for differentiable Bernoulli sampling.
    """
    u = torch.rand_like(logits).clamp(eps, 1.0 - eps)
    logistic = torch.log(u) - torch.log1p(-u)  # log(u) - log(1-u)
    y = torch.sigmoid((logits + logistic) / tau)
    if not hard:
        return y
    y_hard = (y > 0.5).to(y.dtype)
    # Straight-through estimator
    return y_hard.detach() - y.detach() + y


def add_self_loops(adj: torch.Tensor) -> torch.Tensor:
    n = adj.shape[-1]
    eye = torch.eye(n, device=adj.device, dtype=adj.dtype)
    return torch.clamp(adj + eye, 0, 1)


def gcn_norm(adj: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute D^{-1/2} A D^{-1/2} for (batched) adjacency."""
    # adj: [B,N,N] or [N,N]
    if adj.dim() == 2:
        adj_b = adj.unsqueeze(0)
    else:
        adj_b = adj
    deg = adj_b.sum(dim=-1)  # [B,N]
    deg_inv_sqrt = torch.pow(deg.clamp_min(eps), -0.5)
    D = deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)
    out = adj_b * D
    return out.squeeze(0) if adj.dim() == 2 else out
