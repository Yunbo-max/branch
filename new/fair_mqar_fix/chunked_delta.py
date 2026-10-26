"""Chunked delta-rule update — the algebra that removes the per-token loop.

Given a fixed SPD geometry ``H`` (frozen for the chunk) and fixed write directions, the
recurrence

    S_t = S_{t-1} + (v_t - S_{t-1} k_t) a_t^T,      a_t = u_t / (tau + k_t^T u_t)
    u_t = H^{-1} k_t
    o_t = S_t q_t

is exactly equal to the triangular system

    L = tril(K A^T, -1)
    (I + L) E = V - K S_0^T
    O = Q S_0^T + tril(Q A^T) E
    S_T = S_0 + E^T A

because ``S_{i-1} = S_0 + sum_{s<i} e_s a_s^T`` and ``e_i = v_i - S_{i-1} k_i``.

Verified against the per-token loop to ~1e-15 in float64
(``tests/test_chunked_delta.py``) and against the released MARCH-style read in
``march_mixer`` is NOT implied: this module only covers the write recurrence.

Shape convention: state ``S: (B, H, K, V)`` (key-major, the released convention).
``K, Q: (B, T, H, K)``; ``V: (B, T, H, V)``; ``Hgeom: (B, H, K, K)``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _solve_keys(factor: Tensor, key: Tensor) -> Tensor:
    """H^{-1} k_t for every token at once. factor (B,H,K,K); key (B,T,H,K) -> (B,T,H,K)."""
    rhs = key.permute(0, 2, 3, 1)                       # (B,H,K,T): batched (K,K) x (K,T)
    sol = torch.cholesky_solve(rhs, factor)             # (B,H,K,T)
    return sol.permute(0, 3, 1, 2)                      # (B,T,H,K)


def chunked_delta_update(state0: Tensor, key: Tensor, value: Tensor, query: Tensor,
                         factor: Tensor, tau: float,
                         *, write_mask: Tensor | None = None,
                         return_states: bool = False):
    """Run one chunk. Returns (outputs (B,T,H,V), final_state (B,H,K,V)).

    ``factor`` is the Cholesky factor of the *fixed* geometry ``H`` for this chunk;
    ``state0`` is the state at the chunk start. ``write_mask`` (B,T,H), when given,
    zeroes the write direction of masked tokens, so a state can be updated by a subset
    of the chunk's tokens while the others leave it untouched.
    """
    B, T, H, K = key.shape
    Vd = value.shape[-1]
    if query.shape != key.shape or value.shape[:3] != key.shape[:3]:
        raise ValueError("key/query must be (B,T,H,K) and value (B,T,H,V)")
    if state0.shape != (B, H, K, Vd):
        raise ValueError(f"state0 must be {(B, H, K, Vd)}, got {tuple(state0.shape)}")
    if factor.shape != (B, H, K, K):
        raise ValueError(f"factor must be {(B, H, K, K)}, got {tuple(factor.shape)}")

    dtype = key.dtype
    eye = torch.eye(T, device=key.device, dtype=dtype)

    u = _solve_keys(factor, key)                          # H^{-1} K
    denom = tau + (key * u).sum(-1, keepdim=True)         # (B,T,H,1)
    a = u / denom                                         # write directions
    if write_mask is not None:
        if write_mask.shape != key.shape[:3]:
            raise ValueError(f"write_mask must be {tuple(key.shape[:3])}")
        a = a * write_mask.unsqueeze(-1).to(a.dtype)

    s0 = state0.permute(0, 1, 3, 2)                       # (B,H,V,K) for the K S0^T product
    ks0 = torch.einsum("bthk,bhvk->bthv", key, s0)        # K S_0^T
    rhs = (value - ks0).permute(0, 2, 1, 3)               # (B,H,T,V)

    k_a = torch.einsum("bthk,bshk->bhts", key, a)         # K A^T
    lower = torch.tril(k_a, diagonal=-1)
    # I + L is unit lower triangular: use the triangular solver (BLAS, no MAGMA).
    e = torch.linalg.solve_triangular(lower + eye, rhs, upper=False,
                                      unitriangular=True)  # (B,H,T,V)

    q_k_a = torch.tril(torch.einsum("bthk,bshk->bhts", query, a))  # incl. diagonal
    out = torch.einsum("bthk,bhvk->bthv", query, s0) + \
        torch.einsum("bhts,bhsv->bthv", q_k_a, e)

    final = None
    if return_states:
        final = state0 + torch.einsum("bthk,bhtv->bhkv", a, e)    # S_0 + A^T E
    return out, final


def chunked_delta_from_directions(state0: Tensor, key: Tensor, value: Tensor,
                                   query: Tensor, directions: Tensor, *,
                                   write_mask: Tensor | None = None,
                                   return_states: bool = False):
    """Exact triangular recurrence with externally supplied write directions.

    ``directions`` is (B,T,H,K) and already includes the per-token geometry choice
    (isotropic / Rayleigh / full / selected). The state recurrence and the chunk algebra
    are identical to ``chunked_delta_update``; only the directions are not computed here.
    """
    B, T, H, K = key.shape
    Vd = value.shape[-1]
    if directions.shape != key.shape:
        raise ValueError(f"directions must be {tuple(key.shape)}")
    if state0.shape != (B, H, K, Vd):
        raise ValueError(f"state0 must be {(B, H, K, Vd)}, got {tuple(state0.shape)}")
    dtype = key.dtype
    eye = torch.eye(T, device=key.device, dtype=dtype)
    a = directions
    if write_mask is not None:
        if write_mask.shape != key.shape[:3]:
            raise ValueError(f"write_mask must be {tuple(key.shape[:3])}")
        a = a * write_mask.unsqueeze(-1).to(a.dtype)

    s0 = state0.permute(0, 1, 3, 2)
    ks0 = torch.einsum("bthk,bhvk->bthv", key, s0)
    rhs = (value - ks0).permute(0, 2, 1, 3)
    k_a = torch.einsum("bthk,bshk->bhts", key, a)
    lower = torch.tril(k_a, diagonal=-1)
    e = torch.linalg.solve_triangular(lower + eye, rhs, upper=False,
                                      unitriangular=True)
    q_k_a = torch.tril(torch.einsum("bthk,bshk->bhts", query, a))
    out = torch.einsum("bthk,bhvk->bthv", query, s0) + \
        torch.einsum("bhts,bhsv->bthv", q_k_a, e)
    final = None
    if return_states:
        final = state0 + torch.einsum("bthk,bhtv->bhkv", a, e)
    return out, final


def chunked_delta_reference(state0: Tensor, key: Tensor, value: Tensor, query: Tensor,
                            factor: Tensor, tau: float) -> tuple[Tensor, Tensor]:
    """Per-token loop with the same frozen geometry; used only to check the chunk form."""
    B, T, H, K = key.shape
    S = state0.clone()
    outs = []
    for t in range(T):
        kt = key[:, t]                                   # (B,H,K)
        ut = torch.cholesky_solve(kt.unsqueeze(-1), factor).squeeze(-1)
        at = ut / (tau + (kt * ut).sum(-1, keepdim=True))
        # state is (B,H,K,V): S k_t and S q_t contract the key axis
        et = value[:, t] - torch.einsum("bhkv,bhk->bhv", S, kt)
        S = S + torch.einsum("bhv,bhk->bhkv", et, at)
        outs.append(torch.einsum("bhkv,bhk->bhv", S, query[:, t]))
    return torch.stack(outs, 1), S
