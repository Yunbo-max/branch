"""Verify exact Full-C coordinate whitening including gradients wrt H."""
import pytest
import torch

from whitened_delta_reference import geo_reference, whitened_delta_reference


@pytest.mark.parametrize("seed", [0, 11, 42])
def test_whitened_and_original_geo_forward_state_and_backward(seed):
    torch.manual_seed(seed)
    B, T, heads, d, dv = 2, 9, 3, 8, 6
    state = torch.randn(B, heads, d, dv, dtype=torch.float64, requires_grad=True)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, heads, d, dtype=torch.float64), dim=-1
    ).requires_grad_()
    q = torch.nn.functional.normalize(
        torch.randn(B, T, heads, d, dtype=torch.float64), dim=-1
    ).requires_grad_()
    v = torch.randn(B, T, heads, dv, dtype=torch.float64, requires_grad=True)
    c = torch.randn(B, heads, d, d, dtype=torch.float64, requires_grad=True)
    H = torch.eye(d, dtype=torch.float64) + 0.5 * c.transpose(-1, -2) @ c
    o0, s0 = geo_reference(state, k, v, q, H)
    o1, s1 = whitened_delta_reference(state, k, v, q, H)
    torch.testing.assert_close(o0, o1, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(s0, s1, atol=1e-11, rtol=1e-11)
    wrt = (state, k, q, v, c)
    grad0 = torch.autograd.grad(
        o0.square().sum() + s0.square().mean(), wrt, retain_graph=True
    )
    grad1 = torch.autograd.grad(
        o1.square().sum() + s1.square().mean(), wrt
    )
    for x, y in zip(grad0, grad1):
        torch.testing.assert_close(x, y, atol=1e-9, rtol=1e-9)
