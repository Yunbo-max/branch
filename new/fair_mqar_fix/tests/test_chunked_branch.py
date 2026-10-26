"""Structural checks for the chunked Branch Memory mixer."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))

from chunked_branch import ChunkedBranchConfig, ChunkedBranchMixer  # noqa: E402
from chunked_delta import chunked_delta_update  # noqa: E402


def cfg(**kw) -> ChunkedBranchConfig:
    base = dict(d_model=16, n_heads=2, d_key=4, d_value=4, d_route=4,
                max_states=4, chunk_len=4)
    base.update(kw)
    return ChunkedBranchConfig(**base)


def test_forward_shape_and_cache():
    m = ChunkedBranchMixer(cfg()).double().eval()
    x = torch.randn(2, 9, 16, dtype=torch.double)
    with torch.no_grad():
        out, cache, stats = m(x)
    assert out.shape == (2, 9, 16)
    assert cache.states.shape == (2, 2, 4, 4, 4)
    assert cache.step == 9
    assert cache.allocated_tensor_bytes() > 0
    assert torch.isfinite(out).all()


def test_causality():
    torch.manual_seed(0)
    m = ChunkedBranchMixer(cfg(chunk_len=4)).double().eval()
    x = torch.randn(1, 12, 16, dtype=torch.double)
    with torch.no_grad():
        a, _, _ = m(x)
        x2 = x.clone()
        x2[:, 7:] += 3.0
        b, _, _ = m(x2)
    torch.testing.assert_close(a[:, :7], b[:, :7], atol=1e-10, rtol=0)
    assert not torch.allclose(a[:, 7:], b[:, 7:])


def test_streaming_matches_single_pass_at_chunk_boundaries():
    torch.manual_seed(1)
    m = ChunkedBranchMixer(cfg(chunk_len=4)).double().eval()
    x = torch.randn(1, 12, 16, dtype=torch.double)
    with torch.no_grad():
        whole, c_whole, _ = m(x)
        cache, parts = None, []
        for s, e in [(0, 4), (4, 8), (8, 12)]:
            z, cache, _ = m(x[:, s:e], cache)
            parts.append(z)
    torch.testing.assert_close(torch.cat(parts, 1), whole, atol=1e-8, rtol=1e-7)
    torch.testing.assert_close(cache.states, c_whole.states, atol=1e-8, rtol=1e-7)


def test_gradients_flow_to_the_mixer():
    m = ChunkedBranchMixer(cfg()).double()
    x = torch.randn(1, 8, 16, dtype=torch.double, requires_grad=True)
    out, _, _ = m(x)
    out.square().mean().backward()
    params = dict(m.named_parameters())
    for name in ("q_proj.weight", "k_proj.weight", "v_proj.weight",
                 "route_q.weight", "value_probes", "content_proj"):
        p = params[name]
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_chunk_len_one_is_a_per_token_run():
    """With chunk_len=1 the mixer must run without chunking and stay finite."""
    m = ChunkedBranchMixer(cfg(chunk_len=1)).double().eval()
    x = torch.randn(1, 6, 16, dtype=torch.double)
    with torch.no_grad():
        out, cache, _ = m(x)
    assert torch.isfinite(out).all() and cache.step == 6


def test_write_mask_leaves_other_states_untouched():
    """A masked token must not change its state (the mechanism the mixer relies on)."""
    torch.manual_seed(2)
    B, T, H, K, V = 1, 5, 1, 3, 3
    S0 = torch.zeros(B, H, K, V, dtype=torch.double)
    R = torch.randn(B, H, K, K, dtype=torch.double)
    factor = torch.linalg.cholesky(R @ R.mT + torch.eye(K, dtype=torch.double))
    Kt = torch.nn.functional.normalize(torch.randn(B, T, H, K, dtype=torch.double), dim=-1)
    Vt = torch.randn(B, T, H, V, dtype=torch.double)
    Q = torch.randn(B, T, H, K, dtype=torch.double)
    mask = torch.zeros(B, T, H, dtype=torch.double)
    mask[:, 2, :] = 1.0
    _, S = chunked_delta_update(S0, Kt, Vt, Q, factor, 1.0,
                                write_mask=mask, return_states=True)
    # only token 2 was allowed to write
    _, S_only2 = chunked_delta_update(S0, Kt[:, 2:3], Vt[:, 2:3], Q[:, 2:3], factor, 1.0,
                                      write_mask=torch.ones(B, 1, H, dtype=torch.double),
                                      return_states=True)
    torch.testing.assert_close(S, S_only2, atol=1e-12, rtol=1e-12)


def test_long_sequence_smoke():
    """1024 tokens at chunk 64: shape, finiteness, and no per-token blow-up."""
    c = ChunkedBranchConfig(d_model=32, n_heads=2, d_key=8, d_value=8, d_route=8,
                            max_states=4, chunk_len=64)
    m = ChunkedBranchMixer(c).double().eval()
    x = torch.randn(1, 1024, 32, dtype=torch.double)
    with torch.no_grad():
        out, cache, _ = m(x)
    assert out.shape == (1, 1024, 32) and torch.isfinite(out).all()
