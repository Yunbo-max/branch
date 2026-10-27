"""Exact frozen-geometry GeoDelta expressed as an ordinary Delta recurrence.

For H = L L^T and S in R^{D x V}, define
    k' = L^{-1} k,   q' = L^{-1} q,   W = L^T S,
    beta = 1 / (tau + ||k'||^2).
Then W <- W + beta k' (v - W^T k')^T and o=W^T q' are exactly the
Full-Geometry GeoDelta update/read in real arithmetic.

This is a CPU/PyTorch correctness reference. It provides an algorithmic route
to stock Flash Linear Attention chunk_delta_rule by transforming Q/K/state
on each frozen-C block. BF16 numeric parity and GPU throughput are NOT proven.
"""
from __future__ import annotations

import torch
from torch import Tensor


def geo_reference(
    state: Tensor, k: Tensor, v: Tensor, q: Tensor, H: Tensor, tau: float = 1.
) -> tuple[Tensor, Tensor]:
    """Direct Full-C reference. Shapes S[B,H,D,V], k/q[B,T,H,D], v[B,T,H,V]."""
    L = torch.linalg.cholesky(H)
    u = torch.cholesky_solve(k.permute(0, 2, 3, 1), L).permute(0, 3, 1, 2)
    a = u / (tau + (k * u).sum(-1, keepdim=True))
    s = state
    outputs = []
    for i in range(k.shape[1]):
        e = v[:, i] - torch.einsum("bhkv,bhk->bhv", s, k[:, i])
        s = s + torch.einsum("bhk,bhv->bhkv", a[:, i], e)
        outputs.append(torch.einsum("bhkv,bhk->bhv", s, q[:, i]))
    return torch.stack(outputs, 1), s


def whitened_delta_reference(
    state: Tensor, k: Tensor, v: Tensor, q: Tensor, H: Tensor, tau: float = 1.
) -> tuple[Tensor, Tensor]:
    """Exact GeoDelta via coordinate-whitened ordinary Delta Rule.

    FLA adapters can replace the serial ordinary Delta loop with its
    fused Delta kernel, but must use scale=1, no implicit K normalisation,
    and preserve the chunk-start/final-state basis transforms.
    """
    L = torch.linalg.cholesky(H)
    z_k = torch.linalg.solve_triangular(
        L, k.permute(0, 2, 3, 1), upper=False
    ).permute(0, 3, 1, 2)
    z_q = torch.linalg.solve_triangular(
        L, q.permute(0, 2, 3, 1), upper=False
    ).permute(0, 3, 1, 2)
    W = L.transpose(-1, -2) @ state
    beta = 1. / (tau + z_k.square().sum(-1))
    outputs = []
    for i in range(k.shape[1]):
        e = v[:, i] - torch.einsum("bhkv,bhk->bhv", W, z_k[:, i])
        W = W + torch.einsum(
            "bhk,bhv->bhkv", beta[:, i].unsqueeze(-1) * z_k[:, i], e
        )
        outputs.append(torch.einsum("bhkv,bhk->bhv", W, z_q[:, i]))
    final_state = torch.linalg.solve_triangular(
        L.transpose(-1, -2), W, upper=True
    )
    return torch.stack(outputs, 1), final_state


def gated_geo_reference(
    state: Tensor, k: Tensor, v: Tensor, q: Tensor, H: Tensor,
    gate: Tensor, tau: float = 1.,
) -> tuple[Tensor, Tensor]:
    """GeoDelta with scalar retention gate alpha_t in [0,1] on old state."""
    L = torch.linalg.cholesky(H)
    u = torch.cholesky_solve(k.permute(0, 2, 3, 1), L).permute(0, 3, 1, 2)
    a = u / (tau + (k * u).sum(-1, keepdim=True))
    S = state
    outputs = []
    for i in range(k.shape[1]):
        S = gate[:, i].unsqueeze(-1).unsqueeze(-1) * S
        e = v[:, i] - torch.einsum("bhkv,bhk->bhv", S, k[:, i])
        S = S + torch.einsum("bhk,bhv->bhkv", a[:, i], e)
        outputs.append(torch.einsum("bhkv,bhk->bhv", S, q[:, i]))
    return torch.stack(outputs, 1), S


def whitened_gated_geo_reference(
    state: Tensor, k: Tensor, v: Tensor, q: Tensor, H: Tensor,
    gate: Tensor, tau: float = 1.,
) -> tuple[Tensor, Tensor]:
    """Gated GeoDelta is exactly GDN over whitened keys/queries/state."""
    L = torch.linalg.cholesky(H)
    z_k = torch.linalg.solve_triangular(
        L, k.permute(0, 2, 3, 1), upper=False
    ).permute(0, 3, 1, 2)
    z_q = torch.linalg.solve_triangular(
        L, q.permute(0, 2, 3, 1), upper=False
    ).permute(0, 3, 1, 2)
    W = L.transpose(-1, -2) @ state
    beta = 1. / (tau + z_k.square().sum(-1))
    outputs = []
    for i in range(k.shape[1]):
        W = gate[:, i].unsqueeze(-1).unsqueeze(-1) * W
        e = v[:, i] - torch.einsum("bhkv,bhk->bhv", W, z_k[:, i])
        W = W + torch.einsum(
            "bhk,bhv->bhkv", beta[:, i].unsqueeze(-1) * z_k[:, i], e
        )
        outputs.append(torch.einsum("bhkv,bhk->bhv", W, z_q[:, i]))
    final_state = torch.linalg.solve_triangular(
        L.transpose(-1, -2), W, upper=True
    )
    return torch.stack(outputs, 1), final_state
