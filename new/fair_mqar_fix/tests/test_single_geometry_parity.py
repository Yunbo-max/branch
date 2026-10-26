"""T1: single-state geometry parity, external directions, causality, config wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))

from chunked_delta import chunked_delta_from_directions  # noqa: E402
from fair_models import FairLMConfig, build_mixer  # noqa: E402


BASE = dict(
    vocab_size=64,
    d_model=16,
    n_layers=1,
    n_heads=2,
    d_key=4,
    d_value=4,
    d_route=4,
    max_states=1,
    chunk_len=4,
    protect=8.0,
    covariance="full",
    differentiate_covariance=True,
)


def make(mixer: str, **kw):
    return build_mixer(FairLMConfig(mixer=mixer, **BASE, **kw))


def test_full_mode_matches_chunked_m1():
    torch.manual_seed(0)
    ref = make("chunked")
    torch.manual_seed(0)
    new = make("single_geo", geometry_mode="full")
    new.load_state_dict(ref.state_dict())
    ref.eval()
    new.eval()
    x = torch.randn(2, 9, BASE["d_model"])
    a, _, _ = ref(x)
    b, _, _ = new(x)
    # The partial tail (9 = 2*4 + 1) goes through the per-token streaming path in
    # single_geo, which is numerically equivalent but not bit-identical to the batched
    # chunked solve; assert_close with a float32 tolerance covers that.
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)

    ref.zero_grad(set_to_none=True)
    new.zero_grad(set_to_none=True)
    a.sum().backward()
    b.sum().backward()
    for (ka, va), (kb, vb) in zip(ref.named_parameters(), new.named_parameters()):
        assert ka == kb
        assert torch.allclose(va.grad, vb.grad, atol=2e-6, rtol=2e-6), ka


def test_external_directions_match_token_loop():
    torch.manual_seed(0)
    B, T, H, K, V = 2, 7, 2, 4, 3
    S0 = torch.randn(B, H, K, V, dtype=torch.float64)
    key = torch.randn(B, T, H, K, dtype=torch.float64)
    value = torch.randn(B, T, H, V, dtype=torch.float64)
    query = torch.randn(B, T, H, K, dtype=torch.float64)
    directions = torch.randn(B, T, H, K, dtype=torch.float64) * 0.1

    out, final = chunked_delta_from_directions(
        S0, key, value, query, directions, return_states=True
    )

    S = S0.clone()
    outs = []
    for t in range(T):
        e = value[:, t] - torch.einsum("bhkv,bhk->bhv", S, key[:, t])
        S = S + torch.einsum("bhv,bhk->bhkv", e, directions[:, t])
        outs.append(torch.einsum("bhkv,bhk->bhv", S, query[:, t]))
    ref_out = torch.stack(outs, 1)
    assert torch.allclose(out, ref_out, atol=1e-10)
    assert torch.allclose(final, S, atol=1e-10)


def test_causality_and_streaming():
    torch.manual_seed(0)
    m = make("single_geo", geometry_mode="rayleigh")
    m.eval()
    x = torch.randn(2, 10, BASE["d_model"])
    x2 = x.clone()
    x2[:, -2:] = torch.randn(2, 2, BASE["d_model"])
    with torch.no_grad():
        o1, _, _ = m(x)
        o2, _, _ = m(x2)
    assert torch.equal(o1[:, :-2], o2[:, :-2])

    with torch.no_grad():
        full, _, _ = m(x)
        first, cache, _ = m(x[:, :8])
        second, _, _ = m(x[:, 8:], cache)
    streamed = torch.cat([first, second], 1)
    assert torch.allclose(full, streamed, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("geometry_mode", ["full", "rayleigh", "isotropic"])
def test_per_token_streaming_equals_whole(geometry_mode):
    """P0: train (one-shot chunk) must equal per-token streaming decode.

    A partial chunk is deferred across calls (frozen geometry + chunk-start read), so
    feeding one token at a time must reproduce the whole-sequence output. ``certified``
    is excluded: its ``easy`` decision is a chunk-level max over the certificate, which
    is not streamable token-by-token.
    """
    torch.manual_seed(0)
    m = make("single_geo", geometry_mode=geometry_mode).eval()
    x = torch.randn(2, 17, BASE["d_model"])
    with torch.no_grad():
        whole, _, _ = m(x)
        cache = None
        parts = []
        for t in range(17):
            y, cache, _ = m(x[:, t : t + 1], cache)
            parts.append(y)
    streamed = torch.cat(parts, 1)
    assert torch.allclose(whole, streamed, atol=2e-5, rtol=2e-5)


def test_geometry_modes_are_finite_and_distinct():
    torch.manual_seed(0)
    x = torch.randn(2, 9, BASE["d_model"])
    outputs = {}
    for mode in ("isotropic", "rayleigh", "full"):
        torch.manual_seed(0)
        m = make("single_geo", geometry_mode=mode).eval()
        with torch.no_grad():
            o, _, _ = m(x)
        assert torch.isfinite(o).all()
        outputs[mode] = o
    assert not torch.allclose(outputs["isotropic"], outputs["full"])


def test_config_wiring():
    mixer = build_mixer(
        FairLMConfig(
            mixer="single_geo",
            geometry_mode="rayleigh",
            geometry_threshold=0.05,
            **BASE,
        )
    )
    assert mixer.config.geometry_mode == "rayleigh"
    assert mixer.config.geometry_threshold == 0.05
    assert mixer.config.max_states == 1
