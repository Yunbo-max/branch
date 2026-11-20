"""CPU-only parity tests; GPU official-kernel smoke is optional.

Run from new/fair_mqar_fix:
  python -m pytest -q test_gdn2_full_geometry.py

These tests do NOT certify GPU throughput or official-kernel behavior.
"""
import math
import torch
import torch.nn.functional as F

from gdn2_full_geometry import (
    full_geometry_via_gdn2,
    predicted_skip_gate,
    update_protection_chunk,
)


def cpu_tied_gdn2(*, q, k, v, g, b, w, initial_state,
                  output_final_state, scale, **kwargs):
    """Independent CPU reproduction of GDN-2's zero-decay tied-gate case."""
    assert output_final_state and scale == 1.0
    assert not kwargs.get("use_qk_l2norm_in_kernel", False)
    assert not kwargs.get("beta_guard", True)
    S = initial_state.to(q.dtype)
    ys = []
    for j in range(q.shape[1]):
        kj, qj, vj, bj, wj = (a[:, j] for a in (k, q, v, b, w))
        S = S * g[:, j].exp().unsqueeze(-1)
        old_read = torch.einsum("bhkv,bhk->bhv", S, bj * kj)
        S = S + torch.einsum("bhk,bhv->bhkv", kj, wj * vj - old_read)
        ys.append(torch.einsum("bhkv,bhk->bhv", S, qj))
    return torch.stack(ys, dim=1), S


def direct_full_geometry(q, k, v, *, chunk_size, lam, mu, tau,
                         protection_decay, initial_state, initial_protection):
    """Direct protected-write optimizer, no whitening and no kernel reuse."""
    B, T, nh, dk = q.shape
    S = initial_state.clone()
    C = initial_protection.clone()
    outs = []
    eye = torch.eye(dk, dtype=q.dtype, device=q.device)
    for start in range(0, T, chunk_size):
        stop = min(start + chunk_size, T)
        L = torch.linalg.cholesky(lam * eye + mu * C)
        for j in range(start, stop):
            kj, qj, vj = k[:, j], q[:, j], v[:, j]
            u = torch.cholesky_solve(kj.unsqueeze(-1), L).squeeze(-1)
            beta = (tau + (kj * u).sum(-1)).reciprocal()
            residual = vj - torch.einsum("bhkv,bhk->bhv", S, kj)
            S = S + torch.einsum("bhk,bhv->bhkv", beta[..., None] * u, residual)
            outs.append(torch.einsum("bhkv,bhk->bhv", S, qj))
        C = update_protection_chunk(C, q[:, start:stop],
                                    decay=protection_decay)
    return torch.stack(outs, dim=1), S, C


def test_full_geometry_equivalence_with_changing_chunk_geometry():
    torch.manual_seed(21)
    B, T, nh, dk, dv = 2, 13, 2, 6, 4
    q = F.normalize(torch.randn(B, T, nh, dk, dtype=torch.float64), dim=-1)
    k = F.normalize(torch.randn(B, T, nh, dk, dtype=torch.float64), dim=-1)
    v = torch.randn(B, T, nh, dv, dtype=torch.float64)
    S0 = 0.02 * torch.randn(B, nh, dk, dv, dtype=torch.float64)
    X = torch.randn(B, nh, dk, dk, dtype=torch.float64)
    C0 = 0.1 * (X @ X.transpose(-1, -2))
    p = dict(chunk_size=4, lam=1.2, mu=6.0, tau=0.8,
             protection_decay=0.91)
    ya, Sa, Ca = full_geometry_via_gdn2(
        q, k, v, initial_state=S0, initial_protection=C0,
        kernel=cpu_tied_gdn2, detach_protection=True, **p
    )
    yr, Sr, Cr = direct_full_geometry(
        q, k, v, initial_state=S0, initial_protection=C0, **p
    )
    assert torch.allclose(ya, yr, atol=3e-7, rtol=1e-6)
    assert torch.allclose(Sa, Sr, atol=3e-7, rtol=1e-6)
    assert torch.allclose(Ca, Cr, atol=1e-12, rtol=1e-12)


def test_no_future_write_can_change_past_outputs():
    torch.manual_seed(42)
    q = F.normalize(torch.randn(1, 11, 2, 8), dim=-1)
    k = F.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn(1, 11, 2, 4)
    args = dict(chunk_size=4, kernel=cpu_tied_gdn2)
    a, *_ = full_geometry_via_gdn2(q, k, v, **args)
    vv = v.clone()
    vv[:, 7:] += 1e3
    b, *_ = full_geometry_via_gdn2(q, k, vv, **args)
    assert torch.allclose(a[:, :7], b[:, :7], atol=1e-5, rtol=1e-5)


def test_predicted_skip_signal_is_not_full_geometry():
    S = torch.zeros(1, 1, 2, 2)
    S[0, 0, 0, 0] = 1.0
    pq = torch.tensor([[[[1., 0.], [0., 0.]]]]).transpose(1, 2)
    # previous_queries [B,T,H,K]
    val = torch.tensor([[[[1., 0.]], [[0., 1.]]]])
    r = predicted_skip_gate(S, pq, val, gamma=2.0)
    assert abs(float(r[0, 0, 0]) - math.exp(-2.0)) < 1e-6
    assert abs(float(r[0, 1, 0]) - 1.0) < 1e-6


def test_full_channel_gates_reduce_to_original_writer_when_unit_and_zero_decay():
    torch.manual_seed(6)
    q = F.normalize(torch.randn(1, 9, 2, 6), dim=-1)
    k = F.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn(1, 9, 2, 4)
    settings = dict(chunk_size=4, kernel=cpu_tied_gdn2)
    exact, *_ = full_geometry_via_gdn2(q, k, v, **settings)
    equal, *_ = full_geometry_via_gdn2(
        q, k, v, erase_gate=torch.ones_like(k),
        write_gate=torch.ones_like(v), log_decay=torch.zeros_like(k),
        **settings
    )
    gated, *_ = full_geometry_via_gdn2(
        q, k, v, erase_gate=torch.sigmoid(torch.randn_like(k)),
        write_gate=torch.sigmoid(torch.randn_like(v)),
        log_decay=-torch.rand_like(k) * 0.1,
        **settings
    )
    assert torch.allclose(exact, equal, atol=1e-6)
    assert not torch.allclose(exact, gated, atol=1e-4)
