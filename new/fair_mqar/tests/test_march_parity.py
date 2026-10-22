"""Parity tests for the MARCH mixer against the released implementation's semantics.

The routing helper is imported from the archived official source
(`new/MARCH/code/march_official/ops.py`), so these tests check our port against the
real numerics rather than against a re-derivation.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))

from march_mixer import MARCHCache, MARCHMemoryConfig, MARCHMixer, route_softmax_weights  # noqa: E402
from fair_models import FairLM, FairLMConfig  # noqa: E402


def cfg(**kw) -> MARCHMemoryConfig:
    base = dict(d_model=16, n_heads=1, d_key=4, d_value=4, d_route=4,
                anchor_interval=4, max_anchors=4, read_topk=4)
    base.update(kw)
    return MARCHMemoryConfig(**base)


def test_uses_the_official_routing_helper():
    official = _HERE.parent / "MARCH" / "code" / "march_official" / "ops.py"
    assert official.exists(), "archived official source is missing"
    assert route_softmax_weights.__module__ == "ops"


def test_null_option_absorbs_probability_mass():
    """Official: the null candidate takes softmax mass and is dropped from the output."""
    logits = torch.zeros(1, 1, 1, 3)
    valid = torch.ones(1, 1, 1, 3, dtype=torch.bool)
    without_null = route_softmax_weights(logits, valid)
    assert torch.allclose(without_null.sum(-1), torch.ones(1, 1, 1))

    null_logits = torch.zeros(1, 1, 1, 1)
    with_null = route_softmax_weights(torch.cat([logits, null_logits], -1),
                                      torch.cat([valid, torch.ones_like(valid[..., :1])], -1))
    assert torch.allclose(with_null.sum(-1), torch.ones(1, 1, 1))
    assert with_null[..., :3].sum() < 1.0            # null holds the remainder
    assert torch.allclose(with_null[..., :3].sum(), torch.full((1, 1, 1), 0.75), atol=1e-6)


def test_fully_masked_row_is_zero_not_nan():
    logits = torch.randn(1, 2, 1, 3)
    valid = torch.zeros(1, 2, 1, 3, dtype=torch.bool)
    w = route_softmax_weights(logits, valid)
    assert torch.isfinite(w).all() and torch.allclose(w, torch.zeros_like(w))


def test_read_matches_manual_official_formula():
    torch.manual_seed(0)
    c = cfg()
    m = MARCHMixer(c).double().eval()
    B, T, H, K, V, M = 1, 3, c.n_heads, c.d_key, c.d_value, 3
    states = torch.randn(B, M, H, K, V, dtype=torch.double)
    anchor_k = torch.randn(B, M, H, c.d_route, dtype=torch.double)
    boundaries = torch.tensor([[1, 5, 9]], dtype=torch.long)      # anchor 0 visible from t>=1
    read_q = torch.randn(B, T, H, K, dtype=torch.double)
    router_q = torch.randn(B, T, H, c.d_route, dtype=torch.double)
    null_logits = torch.randn(B, T, H, 1, dtype=torch.double)

    out, weights = m.history_read(read_q, router_q, states, anchor_k, boundaries,
                                  null_logits, start_index=0)

    # manual recomputation straight from the official path
    token_index = torch.arange(T)
    valid = (boundaries.view(B, 1, M) <= token_index.view(1, T, 1)).view(B, T, 1, M).expand(B, T, H, M)
    logits = torch.einsum("bthr,bmhr->bthm", router_q.float(), anchor_k.float()) * m._route_scale()
    if 0 < c.read_topk < M:
        _, idx = logits.masked_fill(~valid, float("-inf")).topk(k=c.read_topk, dim=-1)
        keep = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, idx, True)
        valid = valid & keep
    logits_with_null = torch.cat([logits, null_logits.float()], -1)
    valid_with_null = torch.cat([valid, torch.ones(B, T, H, 1, dtype=torch.bool)], -1)
    expect_w = route_softmax_weights(logits_with_null, valid_with_null, -1)[..., :M]
    assert torch.allclose(weights.float(), expect_w, atol=1e-6)

    reads = torch.einsum("bthk,bmhkv->bthmv", read_q, states) * (c.d_key ** -0.5)
    reads = reads.masked_fill(~valid.unsqueeze(-1), 0)
    expect_out = torch.einsum("bthm,bthmv->bthv", expect_w.to(reads.dtype), reads)
    assert torch.allclose(out, expect_out, atol=1e-6)


def test_topk_limits_the_number_of_read_anchors():
    torch.manual_seed(1)
    c = cfg(read_topk=1, max_anchors=6)
    m = MARCHMixer(c).double().eval()
    B, T, H, K, V, M = 1, 1, c.n_heads, c.d_key, c.d_value, 6
    states = torch.randn(B, M, H, K, V, dtype=torch.double)
    anchor_k = torch.randn(B, M, H, c.d_route, dtype=torch.double)
    boundaries = torch.zeros(B, M, dtype=torch.long)   # all visible
    _, weights = m.history_read(torch.randn(B, T, H, K, dtype=torch.double),
                                torch.randn(B, T, H, c.d_route, dtype=torch.double),
                                states, anchor_k, boundaries, None, start_index=0)
    assert int((weights > 0).sum()) == 1


def test_boundary_visibility_is_inclusive():
    """Official uses `boundary <= token_index`."""
    torch.manual_seed(2)
    c = cfg()
    m = MARCHMixer(c).double().eval()
    B, H, K, V, M = 1, c.n_heads, c.d_key, c.d_value, 2
    states = torch.randn(B, M, H, K, V, dtype=torch.double)
    anchor_k = torch.randn(B, M, H, c.d_route, dtype=torch.double)
    boundaries = torch.tensor([[2, 5]], dtype=torch.long)
    T = 7
    _, weights = m.history_read(torch.randn(B, T, H, K, dtype=torch.double),
                                torch.randn(B, T, H, c.d_route, dtype=torch.double),
                                states, anchor_k, boundaries, None, start_index=0)
    visible_counts = (weights > 0).sum(-1)[0, :, 0].tolist()
    # t=0,1 -> none; t=2,3,4 -> anchor 0 only; t=5,6 -> both (topk=4 keeps both)
    assert visible_counts == [0, 0, 1, 1, 1, 2, 2], visible_counts


def test_causality_and_streaming_equivalence():
    torch.manual_seed(3)
    c = cfg(anchor_interval=3, max_anchors=6)
    m = MARCHMixer(c).double().eval()
    x = torch.randn(2, 9, c.d_model, dtype=torch.double)
    with torch.no_grad():
        full, _, _ = m(x)
        altered = x.clone()
        altered[:, 5:] += 3.0
        partial, _, _ = m(altered)
        assert torch.allclose(full[:, :5], partial[:, :5], atol=1e-10)

        stream_cache = None
        pieces = []
        for s, e in [(0, 2), (2, 5), (5, 9)]:
            z, stream_cache, _ = m(x[:, s:e], stream_cache)
            pieces.append(z)
        assert torch.allclose(full, torch.cat(pieces, 1), atol=1e-9, rtol=1e-8)


def test_no_future_gradient():
    c = cfg(anchor_interval=3, max_anchors=4)
    m = MARCHMixer(c).double()
    x = torch.randn(1, 8, c.d_model, dtype=torch.double, requires_grad=True)
    y, _, _ = m(x)
    g = torch.autograd.grad(y[:, :4].square().sum(), x)[0]
    assert int(torch.count_nonzero(g[:, 4:])) == 0


def test_anchor_capacity_is_respected():
    c = cfg(anchor_interval=2, max_anchors=3)
    m = MARCHMixer(c).double().eval()
    with torch.no_grad():
        _, cache, _ = m(torch.randn(1, 20, c.d_model, dtype=torch.double))
    assert int(cache.counts.max()) <= c.max_anchors


def test_shared_lm_builds_both_mixers():
    for mixer in ("march", "branch"):
        lm = FairLM(FairLMConfig(mixer=mixer, vocab_size=64, d_model=16, n_layers=1,
                                 n_heads=1, d_key=4, d_value=4, d_route=4,
                                 anchor_interval=4, max_anchors=4, max_states=4))
        ids = torch.randint(0, 60, (2, 12))
        logits, _ = lm(ids)
        assert logits.shape == (2, 12, 64)
