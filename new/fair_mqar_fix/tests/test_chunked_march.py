"""Structural checks for the chunked MARCH mixer."""
import sys
from pathlib import Path
import pytest, torch
_HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(_HERE))
from chunked_march import ChunkedMARCHConfig, ChunkedMARCHMixer


def cfg(**kw):
    base = dict(d_model=16, n_heads=2, d_key=4, d_value=4, d_route=4,
                max_states=8, chunk_len=4)
    base.update(kw); return ChunkedMARCHConfig(**base)


def test_forward_and_cache():
    m = ChunkedMARCHMixer(cfg()).double().eval()
    with torch.no_grad():
        out, cache, stats = m(torch.randn(2, 9, 16, dtype=torch.double))
    assert out.shape == (2, 9, 16) and torch.isfinite(out).all()
    assert cache.state.shape == (2, 2, 4, 4)
    assert cache.step == 9


def test_causality():
    torch.manual_seed(0)
    m = ChunkedMARCHMixer(cfg(chunk_len=4)).double().eval()
    x = torch.randn(1, 12, 16, dtype=torch.double)
    with torch.no_grad():
        a, _, _ = m(x)
        x2 = x.clone(); x2[:, 7:] += 3.0
        b, _, _ = m(x2)
    torch.testing.assert_close(a[:, :7], b[:, :7], atol=1e-10, rtol=0)


def test_streaming_matches_single_pass():
    torch.manual_seed(1)
    m = ChunkedMARCHMixer(cfg(chunk_len=4)).double().eval()
    x = torch.randn(1, 12, 16, dtype=torch.double)
    with torch.no_grad():
        whole, _, _ = m(x)
        c, parts = None, []
        for s, e in [(0, 4), (4, 8), (8, 12)]:
            z, c, _ = m(x[:, s:e], c); parts.append(z)
    torch.testing.assert_close(torch.cat(parts, 1), whole, atol=1e-9, rtol=1e-8)


def test_anchor_capacity():
    m = ChunkedMARCHMixer(cfg(chunk_len=2, max_states=3)).double().eval()
    with torch.no_grad():
        _, cache, _ = m(torch.randn(1, 40, 16, dtype=torch.double))
    assert int(cache.counts.max()) <= 3


def test_gradients():
    m = ChunkedMARCHMixer(cfg()).double()
    x = torch.randn(1, 8, 16, dtype=torch.double, requires_grad=True)
    out, _, _ = m(x); out.square().mean().backward()
    p = dict(m.named_parameters())
    for n in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "anchor_k_proj.weight"):
        assert p[n].grad is not None and torch.isfinite(p[n].grad).all(), n


def test_long_smoke():
    c = ChunkedMARCHConfig(d_model=32, n_heads=2, d_key=8, d_value=8, d_route=8,
                           max_states=8, chunk_len=64)
    m = ChunkedMARCHMixer(c).double().eval()
    with torch.no_grad():
        out, _, _ = m(torch.randn(1, 1024, 32, dtype=torch.double))
    assert out.shape == (1, 1024, 32) and torch.isfinite(out).all()
