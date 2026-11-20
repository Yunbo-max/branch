"""Exact chunk-frozen Branch Full Geometry using the OFFICIAL GDN-2 kernel.

With default tied gates/zero decay, this is an *exact* Full Geometry
compatibility adapter. Optional untied erase/write and channel-wise decay
extend the model within the protected basis. That extended model is new
and is NOT the same recurrence as the original Branch writer.

For a fixed query-protection metric H = lam I + mu C within a chunk, the
original protected update (key-major S [K,V]) is

  u = H^{-1} k
  a = u / (tau + k^T u)
  S' = S + a (v - S^T k)^T.

Using H = L L^T, define
  Z = L^T S, kh = L^{-1} k, qh = L^{-1} q.
Then
  Z' = Z + beta kh (v - Z^T kh)^T,
  beta = 1/(tau + ||kh||^2),
  Z'^T qh = S'^T q.

The last recurrence is exactly the *tied scalar-gate, zero-decay special case*
of the official NVlabs GDN-2 chunk kernel: b=beta (K channels),
w=beta (V channels), g=0. Optional b_t/w_t/g_t use channel-wise gates
inside the transformed coordinate system, scaling b,w by beta. The
restricted setting recovers the exact Branch update; the general setting
defines a geometry-conditioned GDN-2 with full gate expressivity. The kernel must NOT renormalize kh/qh, and
its key-weighted beta_guard must be disabled: both otherwise alter this
exact equivalence. We do not reimplement any Triton/CUDA kernel.

H and C are updated only at chunk boundaries, using *past* queries. The
state is converted into and out of the chunk's temporary basis. That
conversion, factorization and Python chunk loop have real costs; no speed
claim relative to native GDN-2 is made.

The official NVlabs/GatedDeltaNet-2 source is a separately installed
dependency, subject to its own NVIDIA Source Code License-NC. Do not vendor
or redistribute the kernels through this repository.
"""
from __future__ import annotations

from typing import Callable
import torch
import torch.nn.functional as F
from torch import Tensor


def _official_chunk_gdn2(**kwargs):
    try:
        from lit_gpt.gdn2_ops.chunk_gdn2 import chunk_gdn2
    except ImportError as exc:
        raise RuntimeError(
            "Install NVlabs/GatedDeltaNet-2 separately and make its repository "
            "importable (PYTHONPATH). No GDN-2 code is bundled here."
        ) from exc
    return chunk_gdn2(**kwargs)


def _solve_keys(L: Tensor, x: Tensor) -> Tensor:
    """Apply L^{-1} to key-axis vectors: x [B,T,H,K], L [B,H,K,K]."""
    rhs = x.permute(0, 2, 3, 1).contiguous()
    sol = torch.linalg.solve_triangular(L, rhs, upper=False)
    return sol.permute(0, 3, 1, 2).contiguous()


def update_protection_chunk(
    C: Tensor, queries: Tensor, *, decay: float
) -> Tensor:
    """Causal EMA of query outer products, committed after the chunk."""
    if not 0 <= decay < 1:
        raise ValueError("decay must be in [0,1)")
    n = queries.shape[1]
    if not n:
        return C
    powers = decay ** torch.arange(
        n - 1, -1, -1, dtype=queries.dtype, device=queries.device
    )
    weighted = queries * powers[None, :, None, None]
    increment = torch.einsum("bthk,bthj->bhkj", weighted, queries)
    return decay**n * C + (1 - decay) * increment


def predicted_skip_gate(
    chunk_start_state: Tensor,
    previous_queries: Tensor,
    values: Tensor,
    *,
    gamma: float = 0.25,
) -> Tensor:
    """Optional original Predicted-Skip signal, independent of Full Geometry.

    S_start is [B,H,K,V]; previous_queries is [B,T,H,K] already
    shifted causally by one token. Returns rho [B,T,H]. This function
    DOES NOT select whether to scale only w or both w and b in GDN-2.
    That is an experimental choice, not an established equivalence.
    """
    if gamma < 0:
        raise ValueError("gamma must be nonnegative")
    read = torch.einsum("bhkv,bthk->bthv", chunk_start_state, previous_queries)
    match = F.cosine_similarity(read, values, dim=-1, eps=1e-6).relu()
    return (-gamma * match).exp()


def full_geometry_via_gdn2(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    chunk_size: int = 64,
    lam: float = 1.0,
    mu: float = 8.0,
    tau: float = 1.0,
    protection_decay: float = 0.95,
    initial_state: Tensor | None = None,
    initial_protection: Tensor | None = None,
    detach_protection: bool = True,
    erase_gate: Tensor | None = None,
    write_gate: Tensor | None = None,
    log_decay: Tensor | None = None,
    kernel: Callable | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Execute the exact chunk-frozen Full Geometry recurrence.

    q,k: [B,T,H,K], v: [B,T,H,V]. Pass q,k after whichever
    pre-normalization defines the intended Branch reference equations;
    this adapter deliberately does not change their norms.

    Returns output [B,T,H,V], physical final state [B,H,K,V], and
    updated full covariance [B,H,K,K]. If optional erase_gate [B,T,H,K],
    write_gate [B,T,H,V] or log_decay [B,T,H,K] are provided, GDN-2
    runs with fully channel-wise gating in the protected coordinates.
    In that case this is a new geometric extension, not strict equivalence
    to original Branch. Use the LAST TWO as the streaming
    initial state/protection for the next call.

    kernel: default uses official NVlabs chunk_gdn2; an injectable
    differentiable CPU recurrence can test parity without CUDA.
    """
    if q.shape != k.shape or q.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("Expected q,k [B,T,H,K] and v [B,T,H,V]")
    if not (lam > 0 and mu >= 0 and tau > 0 and chunk_size > 0):
        raise ValueError("lam,tau,chunk_size positive; mu nonnegative")
    B, T, nh, dk = q.shape
    dv = v.shape[-1]
    if dk > 256:
        raise ValueError("The official GDN-2 kernel supports key headdim <= 256")
    if T < 1:
        raise ValueError("At least one token is required")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Use the same dtype for q, k, v (float32 for the PoC)")
    for name, gate, shape in (
        ("erase_gate", erase_gate, (B,T,nh,dk)),
        ("write_gate", write_gate, (B,T,nh,dv)),
        ("log_decay", log_decay, (B,T,nh,dk)),
    ):
        if gate is not None and gate.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if initial_state is None:
        S = q.new_zeros(B, nh, dk, dv)
    else:
        if initial_state.shape != (B, nh, dk, dv):
            raise ValueError("Invalid initial_state shape")
        S = initial_state
    if initial_protection is None:
        C = q.new_zeros(B, nh, dk, dk)
    else:
        if initial_protection.shape != (B, nh, dk, dk):
            raise ValueError("Invalid initial_protection shape")
        C = initial_protection
    runner = kernel or _official_chunk_gdn2
    eye = torch.eye(dk, device=q.device, dtype=q.dtype)
    outputs = []

    for start in range(0, T, chunk_size):
        stop = min(start + chunk_size, T)
        qs, ks, vs = q[:, start:stop], k[:, start:stop], v[:, start:stop]

        Cused = C.detach() if detach_protection else C
        Hmetric = lam * eye + mu * Cused
        L = torch.linalg.cholesky(Hmetric)

        # In the temporary protected basis the physical state becomes Z=L^T S.
        Z0 = torch.matmul(L.transpose(-1, -2), S)
        qh, kh = _solve_keys(L, qs), _solve_keys(L, ks)
        beta = (tau + kh.square().sum(dim=-1)).reciprocal()
        erase_factor = (
            erase_gate[:, start:stop] if erase_gate is not None
            else torch.ones_like(kh)
        )
        write_factor = (
            write_gate[:, start:stop] if write_gate is not None
            else torch.ones_like(vs)
        )
        erase_b = (beta.unsqueeze(-1) * erase_factor).contiguous()
        write_w = (beta.unsqueeze(-1) * write_factor).contiguous()
        gate_g = (
            log_decay[:, start:stop] if log_decay is not None
            else torch.zeros_like(kh)
        )

        y, Zfinal = runner(
            q=qh.contiguous(), k=kh.contiguous(), v=vs.contiguous(),
            g=gate_g.contiguous(), b=erase_b, w=write_w,
            scale=1.0,
            initial_state=Z0.to(torch.float32),
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=False,
            beta_guard=False,
        )
        if Zfinal is None:
            raise RuntimeError("GDN-2 kernel must return a final state")
        # Return from the temporary basis to the original physical state.
        S = torch.linalg.solve_triangular(
            L.transpose(-1, -2), Zfinal.to(L.dtype), upper=True
        )
        outputs.append(y)

        # Query geometry for the NEXT chunk only: causal, no future leakage.
        C = update_protection_chunk(C, qs, decay=protection_decay)

    return torch.cat(outputs, dim=1), S, C
