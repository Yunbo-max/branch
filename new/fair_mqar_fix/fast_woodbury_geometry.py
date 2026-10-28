"""Fading-window Query-covariance geometry, using a small dual Woodbury solve.

Preserves ALL off-diagonal correlations among the last rank weighted Queries.
The only approximation is truncating older exponential-EMA contributions.
Pure PyTorch correctness reference, NOT a fused CUDA/Triton operator.

Tensors:
  history_q: [batch, previous_tokens, heads, key_dim]
  history_w: [batch, previous_tokens, heads], nonnegative
  keys:      [batch, chunk_tokens, heads, key_dim]

Only previously observed Queries are allowed in history_q. The chunk's future
Queries must not affect the current chunk's write directions.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _validate(q: Tensor, w: Tensor, k: Tensor,
              gamma: float, mu: float, lam: float, tau: float) -> None:
    if q.ndim != 4 or k.ndim != 4 or w.shape != q.shape[:-1]:
        raise ValueError("expected q[B,T,H,D], w[B,T,H], k[B,L,H,D]")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError("batch/head/key dimensions do not match")
    if q.dtype != k.dtype or w.dtype != q.dtype:
        raise ValueError("use one FP32/FP64 dtype throughout geometry")
    if not (0 <= gamma < 1 and mu >= 0 and lam > 0 and tau > 0):
        raise ValueError("invalid decay, protection, ridge or tau")
    if torch.any(w < 0):
        raise ValueError("Query importance must be nonnegative")


def recent_basis(history_q: Tensor, history_w: Tensor, *,
                 gamma: float, mu: float, rank: int) -> Tensor:
    """Return B[B,H,r,D] with B^T B = mu * truncated Query covariance."""
    if rank < 0:
        raise ValueError("rank must be nonnegative")
    n = min(rank, history_q.shape[1])
    b, _, h, d = history_q.shape
    if n == 0:
        return history_q.new_zeros(b, h, 0, d)
    q = history_q[:, -n:].permute(0, 2, 1, 3)
    w = history_w[:, -n:].permute(0, 2, 1)
    ages = torch.arange(n - 1, -1, -1, device=q.device)
    decay = torch.as_tensor(gamma, dtype=q.dtype, device=q.device)
    coef = mu * (1 - gamma) * torch.pow(decay, ages).view(1, 1, n) * w
    return torch.sqrt(coef).unsqueeze(-1) * q


def _direction(k: Tensor, u: Tensor, tau: float) -> Tensor:
    return u / (tau + torch.sum(k * u, dim=-1, keepdim=True))


def woodbury_direction(history_q: Tensor, history_w: Tensor, k: Tensor, *,
                       gamma: float = 0.95, mu: float = 8.,
                       lam: float = 1., tau: float = 1.,
                       rank: int = 128) -> Tensor:
    """Exact direction for the truncated full-correlation covariance.

    H = lam I + B^T B, so H^-1 K^T =
       (K^T - B^T (lam I + B B^T)^-1 B K^T) / lam.
    Factorizes r x r rather than key_dim x key_dim.
    """
    _validate(history_q, history_w, k, gamma, mu, lam, tau)
    B = recent_basis(history_q, history_w,
                     gamma=gamma, mu=mu, rank=rank)
    r = B.shape[-2]
    if r == 0:
        return k / (tau * lam + torch.sum(k * k, -1, keepdim=True))
    gram = B @ B.transpose(-1, -2)
    gram = gram + lam * torch.eye(r, device=k.device, dtype=k.dtype)
    keys = k.permute(0, 2, 3, 1)
    rhs = B @ keys
    chol = torch.linalg.cholesky(gram)
    correction = torch.cholesky_solve(rhs, chol)
    u = ((keys - B.transpose(-1, -2) @ correction) / lam).permute(0, 3, 1, 2)
    return _direction(k, u, tau)


def dense_ema_direction(history_q: Tensor, history_w: Tensor, k: Tensor, *,
                        gamma: float = 0.95, mu: float = 8.,
                        lam: float = 1., tau: float = 1.) -> Tensor:
    """Expensive exact full-history reference for numerical verification."""
    _validate(history_q, history_w, k, gamma, mu, lam, tau)
    B = recent_basis(history_q, history_w,
                     gamma=gamma, mu=mu, rank=history_q.shape[1])
    d = k.shape[-1]
    H = B.transpose(-1, -2) @ B
    H = H + lam * torch.eye(d, device=k.device, dtype=k.dtype)
    chol = torch.linalg.cholesky(H)
    keys = k.permute(0, 2, 3, 1)
    u = torch.cholesky_solve(keys, chol).permute(0, 3, 1, 2)
    return _direction(k, u, tau)


def truncation_bound(gamma: float, rank: int, max_weight: float = 2.) -> float:
    """Upper bound on ||C_full - C_rank||_2 for ||q||<=1 and 0<=w<=max_weight."""
    if not (0 <= gamma < 1 and rank >= 0 and max_weight >= 0):
        raise ValueError("invalid bound parameters")
    return max_weight * gamma ** rank
