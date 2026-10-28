"""GPU-only correctness gate for FLA Full-C whitening adapter.

Requires CUDA + flash-linear-attention/Triton. The PyTorch reference is
mathematically exact but actual BF16 fused-kernel behavior is a separate test.
"""
from __future__ import annotations

import pytest
import torch

from experimental_whitened_fla import geo_via_fla_delta
from whitened_delta_reference import geo_reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/H200")
def test_fla_whitened_delta_matches_geo_at_small_scale():
    pytest.importorskip("fla.ops.delta_rule")
    torch.manual_seed(2026)
    device = "cuda"
    B, L, heads, d, dv = 1, 64, 2, 64, 64
    state = torch.randn(B, heads, d, dv, device=device) * 0.01
    k = torch.nn.functional.normalize(
        torch.randn(B, L, heads, d, device=device), dim=-1
    )
    q = torch.nn.functional.normalize(
        torch.randn(B, L, heads, d, device=device), dim=-1
    )
    v = torch.randn(B, L, heads, dv, device=device).tanh()
    c = torch.randn(B, heads, d, 16, device=device)
    H = torch.eye(d, device=device) + 0.04 * (c @ c.transpose(-1, -2))
    o_ref, s_ref = geo_reference(state, k, v, q, H)
    o_fast, s_fast = geo_via_fla_delta(state, k, v, q, H, chunk_size=64)
    # Explicit accuracy gate, NOT a predeclared success.
    torch.testing.assert_close(o_ref, o_fast.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(s_ref, s_fast.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/H200")
def test_fla_whitened_delta_gradients_finite():
    pytest.importorskip("fla.ops.delta_rule")
    torch.manual_seed(31)
    B, L, Hh, d, dv = 1, 64, 1, 64, 64
    state = torch.randn(B, Hh, d, dv, device="cuda", requires_grad=True)
    k = torch.nn.functional.normalize(
        torch.randn(B, L, Hh, d, device="cuda"), dim=-1
    ).requires_grad_()
    q = torch.nn.functional.normalize(
        torch.randn(B, L, Hh, d, device="cuda"), dim=-1
    ).requires_grad_()
    v = torch.randn(B, L, Hh, dv, device="cuda", requires_grad=True)
    raw = torch.randn(B, Hh, d, 16, device="cuda", requires_grad=True)
    H = torch.eye(d, device="cuda") + 0.1 * (raw @ raw.transpose(-1, -2))
    out, state_end = geo_via_fla_delta(state, k, v, q, H)
    (out.float().square().mean() + state_end.square().mean()).backward()
    for parameter in (state, k, q, v, raw):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/H200")
def test_gated_fla_whitening_matches_gated_geo_reference():
    pytest.importorskip("fla.ops.gated_delta_rule")
    from whitened_delta_reference import gated_geo_reference

    torch.manual_seed(41)
    B, T, Hh, d, dv = 1, 64, 2, 64, 64
    state = torch.randn(B, Hh, d, dv, device="cuda") * .01
    k = torch.nn.functional.normalize(
        torch.randn(B, T, Hh, d, device="cuda"), dim=-1
    )
    q = torch.nn.functional.normalize(
        torch.randn(B, T, Hh, d, device="cuda"), dim=-1
    )
    v = torch.randn(B, T, Hh, dv, device="cuda").tanh()
    raw = torch.randn(B, Hh, d, 16, device="cuda")
    H = torch.eye(d, device="cuda") + .01 * (raw @ raw.transpose(-1, -2))
    log_gate = torch.nn.functional.logsigmoid(
        torch.randn(B, T, Hh, device="cuda")
    )
    ref_out, ref_state = gated_geo_reference(
        state, k, v, q, H, log_gate.exp()
    )
    flash_out, flash_state = geo_via_fla_delta(
        state, k, v, q, H, log_gate=log_gate
    )
    torch.testing.assert_close(ref_out, flash_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(ref_state, flash_state.float(), atol=2e-2, rtol=2e-2)
