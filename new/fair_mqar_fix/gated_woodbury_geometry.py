"""Variable-retention, full-correlation Query geometry via dual Woodbury.

This is an experimental *geometry-only* reference, not a fused recurrent
Delta/GDN operator or a trained 0.79B model.

Assumptions: C_0=0, 0<=rho<=1, 0<=w<=wmax, ||q||_2<=1 when using the bound.
A causal chunk uses only already-observed (q,w,rho) triples. The most recent
chunk's triples must be appended *after* computing that chunk's write directions.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn


class QueryProtectionGate(nn.Module):
    """Predict EMA retention rho in (0,1) per head, initialized to gamma=.95.

    This controls C, NOT the main Delta/GDN memory-state retention alpha.
    """

    def __init__(self, d_model: int, n_heads: int, initial_retention: float = 0.95):
        super().__init__()
        if d_model < 1 or n_heads < 1 or not (0.0 < initial_retention < 1.0):
            raise ValueError("invalid gate shape or initial retention")
        self.linear = nn.Linear(d_model, n_heads)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, math.log(initial_retention / (1.0 - initial_retention)))

    def forward(self, hidden: Tensor) -> Tensor:
        return torch.sigmoid(self.linear(hidden))


def _validate(history_q: Tensor, history_w: Tensor, retention: Tensor, k: Tensor,
              lam: float, mu: float, tau: float) -> None:
    if history_q.ndim != 4 or k.ndim != 4:
        raise ValueError("expected history_q[B,T,H,D], k[B,L,H,D]")
    if history_w.shape != history_q.shape[:-1] or retention.shape != history_w.shape:
        raise ValueError("history_w and retention must be [B,T,H]")
    if history_q.shape[0] != k.shape[0] or history_q.shape[2:] != k.shape[2:]:
        raise ValueError("batch, head and key dimensions disagree")
    if not (lam > 0 and mu >= 0 and tau > 0):
        raise ValueError("require lam,tau>0, mu>=0")
    if torch.any(retention < 0) or torch.any(retention > 1):
        raise ValueError("retention must lie in [0,1]")
    if torch.any(history_w < 0):
        raise ValueError("history Query importance must be nonnegative")


def gated_recent_basis(history_q: Tensor, history_w: Tensor, retention: Tensor,
                       *, rank: int, mu: float = 8.0) -> Tensor:
    """B[B,H,r,D], B^T B equals mu times the last-r contributions to C_t.

    C_t = rho_t C_{t-1} + (1-rho_t) w_t q_t q_t^T; C_0=0.
    """
    if rank < 0 or mu < 0:
        raise ValueError("rank and mu must be nonnegative")
    if history_q.ndim != 4 or history_w.shape != history_q.shape[:-1] or retention.shape != history_w.shape:
        raise ValueError("invalid Query / importance / retention shapes")
    batch, history_len, heads, d = history_q.shape
    n = min(rank, history_len)
    if n == 0:
        return history_q.new_zeros(batch, heads, 0, d)
    q = history_q[:, -n:].permute(0, 2, 1, 3)
    w = history_w[:, -n:].permute(0, 2, 1)
    rho = retention[:, -n:].permute(0, 2, 1)
    inclusive_suffix = torch.flip(torch.cumprod(torch.flip(rho, dims=(-1,)), dim=-1), dims=(-1,))
    future_only = torch.cat((inclusive_suffix[..., 1:], torch.ones_like(inclusive_suffix[..., :1])), dim=-1)
    contribution = mu * (1.0-rho) * w * future_only
    return torch.sqrt(contribution.clamp_min(0.0)).unsqueeze(-1) * q


def _direction(k: Tensor, u: Tensor, tau: float) -> Tensor:
    return u / (tau + (k*u).sum(dim=-1, keepdim=True))


def dense_gated_direction(history_q: Tensor, history_w: Tensor, retention: Tensor,
                          keys: Tensor, *, lam: float = 1.0, mu: float = 8.0,
                          tau: float = 1.0) -> Tensor:
    """Full-d x d Cholesky reference for variable-retention C, C_0=0."""
    _validate(history_q, history_w, retention, keys, lam, mu, tau)
    basis = gated_recent_basis(history_q, history_w, retention,
                               rank=history_q.shape[1], mu=mu)
    d = keys.shape[-1]
    H = lam*torch.eye(d, dtype=keys.dtype, device=keys.device) + basis.transpose(-1,-2)@basis
    u = torch.cholesky_solve(keys.permute(0,2,3,1), torch.linalg.cholesky(H)).permute(0,3,1,2)
    return _direction(keys,u,tau)


def woodbury_gated_direction(history_q: Tensor, history_w: Tensor, retention: Tensor,
                             keys: Tensor, *, rank: int = 64, lam: float = 1.0,
                             mu: float = 8.0, tau: float = 1.0) -> Tensor:
    """Dual rank-r solve, exact for the chosen last-r gated Query covariance."""
    _validate(history_q, history_w, retention, keys, lam, mu, tau)
    basis = gated_recent_basis(history_q, history_w, retention, rank=rank, mu=mu)
    r = basis.shape[-2]
    if r == 0:
        return keys / (tau*lam + keys.square().sum(dim=-1,keepdim=True))
    small = lam*torch.eye(r, device=keys.device, dtype=keys.dtype) + basis@basis.transpose(-1,-2)
    kv = keys.permute(0,2,3,1)
    corr = torch.cholesky_solve(basis@kv, torch.linalg.cholesky(small))
    u = (kv - basis.transpose(-1,-2)@corr).permute(0,3,1,2)/lam
    return _direction(keys,u,tau)


def gated_tail_bound(retention: Tensor, rank: int, *, w_max: float = 2.0) -> Tensor:
    """Bound on ||C_full-C_last_r||_2, assuming C0=0, w<=w_max, ||q||<=1.

    Returns one bound per [B,H].  If rank >= all observed tokens, it is zero.
    """
    if retention.ndim != 3 or rank < 0 or w_max < 0:
        raise ValueError("retention must have shape [B,T,H], rank >=0, w_max>=0")
    if torch.any(retention < 0) or torch.any(retention > 1):
        raise ValueError("retention gate must lie in [0,1]")
    B,T,H = retention.shape
    if rank >= T:
        return retention.new_zeros(B,H)
    if rank == 0:
        return retention.new_full((B,H),w_max)
    return w_max*torch.prod(retention[:, -rank:],dim=1)


def selected_rank(retention: Tensor, *, tolerance: float, max_rank: int,
                  w_max: float = 2.0) -> int | None:
    """Conservative group-wide rank. None requests a dense-C fallback.

    This reference uses a Python reduction and is NOT GPU-graph-friendly.
    Production requires a device-side selection/compaction kernel.
    """
    if tolerance < 0 or max_rank < 0:
        raise ValueError("tolerance and max_rank must be nonnegative")
    T = retention.shape[1]
    for r in range(0,min(T,max_rank)+1):
        if bool(torch.all(gated_tail_bound(retention,r,w_max=w_max) <= tolerance)):
            return r
    return None
