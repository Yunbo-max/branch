"""CPU verification for fading-window full-correlation Woodbury geometry."""
from __future__ import annotations

import pytest
import torch

from fast_woodbury_geometry import (
    dense_ema_direction, recent_basis, truncation_bound, woodbury_direction
)


def make(b=2, hist=20, h=3, d=16, l=11, dtype=torch.float64):
    torch.manual_seed(11)
    q = torch.nn.functional.normalize(torch.randn(b, hist, h, d, dtype=dtype), dim=-1)
    w = 1 + torch.rand(b, hist, h, dtype=dtype)
    k = torch.nn.functional.normalize(torch.randn(b, l, h, d, dtype=dtype), dim=-1)
    return q, w, k


@pytest.mark.parametrize("rank", [16, 20, 64])
def test_full_history_is_mathematically_exact(rank):
    q, w, k = make(hist=16)
    a = dense_ema_direction(q, w, k)
    b = woodbury_direction(q, w, k, rank=rank)
    torch.testing.assert_close(a, b, atol=5e-13, rtol=5e-13)


@pytest.mark.parametrize("rank", [8, 16])
def test_truncated_history_is_exact_within_retained_covariance(rank):
    q, w, k = make(hist=30)
    a = dense_ema_direction(q[:, -rank:], w[:, -rank:], k)
    b = woodbury_direction(q, w, k, rank=rank)
    torch.testing.assert_close(a, b, atol=5e-13, rtol=5e-13)


def test_full_history_input_gradients_are_mathematically_exact():
    q, w, k = make(hist=9, d=12, l=6)
    q0, w0, k0 = [v.clone().requires_grad_(True) for v in (q, w, k)]
    q1, w1, k1 = [v.clone().requires_grad_(True) for v in (q, w, k)]
    a = dense_ema_direction(q0, w0, k0)
    b = woodbury_direction(q1, w1, k1, rank=9)
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    a.square().sum().backward()
    b.square().sum().backward()
    for x, y in zip((q0, w0, k0), (q1, w1, k1)):
        torch.testing.assert_close(x.grad, y.grad, atol=1e-11, rtol=1e-11)


def test_discarded_ema_covariance_has_certified_spectral_bound():
    q, w, k = make(hist=200, d=32)
    gamma = 0.95
    B_full = recent_basis(q, w, gamma=gamma, mu=1., rank=200)
    full = B_full.transpose(-1, -2) @ B_full
    rank = 64
    B_recent = recent_basis(q, w, gamma=gamma, mu=1., rank=rank)
    recent = B_recent.transpose(-1, -2) @ B_recent
    discarded = full - recent
    eig = torch.linalg.eigvalsh(discarded)
    assert eig[..., -1].max().item() <= truncation_bound(gamma, rank) + 1e-12
    assert eig[..., 0].min().item() > -1e-12


def test_zero_history_and_negative_importance():
    q, w, k = make(hist=3, d=4)
    actual = woodbury_direction(q[:, :0], w[:, :0], k, rank=0)
    expected = k / (1. + k.square().sum(-1, keepdim=True))
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError):
        woodbury_direction(q, -w, k)


def test_future_queries_cannot_affect_a_past_chunk():
    q, w, k = make(hist=25, d=16)
    original = woodbury_direction(q[:, :20], w[:, :20], k, rank=16)
    other_q, other_w = q.clone(), w.clone()
    other_q[:, 20:] = torch.randn_like(other_q[:, 20:])
    other_w[:, 20:] = 1.5
    changed = woodbury_direction(other_q[:, :20], other_w[:, :20], k, rank=16)
    assert torch.equal(original, changed)
