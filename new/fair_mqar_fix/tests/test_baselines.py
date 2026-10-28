"""Smoke and causality tests for the two baseline mixers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))

from baseline_mixers import (  # noqa: E402
    BaselineConfig,
    DeltaNetMixer,
    GatedDeltaNetMixer,
    TransformerMixer,
)
from fair_models import FairLM, FairLMConfig, MIXERS  # noqa: E402


def bcfg(**kw) -> BaselineConfig:
    base = dict(d_model=16, n_heads=2, d_key=4, d_value=4, conv_kernel=3)
    base.update(kw)
    return BaselineConfig(**base)


@pytest.mark.parametrize("cls", [DeltaNetMixer, GatedDeltaNetMixer, TransformerMixer])
def test_forward_shape_and_cache(cls):
    m = cls(bcfg()).double().eval()
    x = torch.randn(2, 7, 16, dtype=torch.double)
    with torch.no_grad():
        out, cache, stats = m(x)
    assert out.shape == x.shape
    assert cache.step == 7
    assert cache.allocated_tensor_bytes() > 0
    assert set(stats) >= {"active_states", "fork", "historical_candidates", "write_cost"}


@pytest.mark.parametrize("cls", [DeltaNetMixer, GatedDeltaNetMixer, TransformerMixer])
def test_causality_and_streaming(cls):
    torch.manual_seed(0)
    m = cls(bcfg()).double().eval()
    x = torch.randn(2, 9, 16, dtype=torch.double)
    with torch.no_grad():
        full, _, _ = m(x)
        altered = x.clone()
        altered[:, 5:] += 4.0
        partial, _, _ = m(altered)
        assert torch.allclose(full[:, :5], partial[:, :5], atol=1e-9)

        cache, pieces = None, []
        for s, e in [(0, 2), (2, 5), (5, 9)]:
            z, cache, _ = m(x[:, s:e], cache)
            pieces.append(z)
        assert torch.allclose(full, torch.cat(pieces, 1), atol=1e-8, rtol=1e-7)


def test_transformer_kv_cache_grows_and_attention_is_causal():
    m = TransformerMixer(bcfg()).double().eval()
    with torch.no_grad():
        _, c1, _ = m(torch.randn(1, 4, 16, dtype=torch.double))
        assert c1.k.shape[2] == 4
        _, c2, _ = m(torch.randn(1, 3, 16, dtype=torch.double), c1)
        assert c2.k.shape[2] == 7 and c2.length == 7


def test_delta_state_is_single_and_recurrent():
    m = DeltaNetMixer(bcfg()).double().eval()
    with torch.no_grad():
        _, c, _ = m(torch.randn(3, 5, 16, dtype=torch.double))
    assert c.state.shape == (3, 2, 4, 4)   # B H K V, no anchor/states dimension


def test_gated_delta_gate_changes_memory_behavior():
    m0 = GatedDeltaNetMixer(bcfg()).double().eval()
    m1 = GatedDeltaNetMixer(bcfg()).double().eval()
    m1.load_state_dict(m0.state_dict())
    with torch.no_grad():
        m0.gate_proj.weight.zero_(); m0.gate_proj.bias.fill_(-20.0)  # forget everything
        m1.gate_proj.weight.zero_(); m1.gate_proj.bias.fill_(20.0)   # remember everything
        x = torch.randn(1, 5, 16, dtype=torch.double)
        o0, _, _ = m0(x)
        o1, _, _ = m1(x)
    # full-forgetting vs full-remembering must give different outputs
    assert not torch.allclose(o0, o1, atol=1e-9)


def test_gated_delta_has_gate_parameters():
    m = GatedDeltaNetMixer(bcfg())
    p = dict(m.named_parameters())
    assert "gate_proj.weight" in p and "gate_proj.bias" in p


def test_all_four_arms_build_in_the_shared_lm():
    for mixer in MIXERS:
        lm = FairLM(FairLMConfig(mixer=mixer, vocab_size=64, d_model=16, n_layers=1,
                                 n_heads=2, d_key=4, d_value=4, d_route=4,
                                 anchor_interval=4, max_anchors=4, max_states=4))
        ids = torch.randint(0, 60, (2, 10))
        logits, _ = lm(ids)
        assert logits.shape == (2, 10, 64), mixer


def test_parameter_counts_are_within_the_same_order_of_magnitude():
    counts = {}
    for mixer in MIXERS:
        with torch.device("meta"):
            lm = FairLM(FairLMConfig(mixer=mixer, vocab_size=512, d_model=64, n_layers=2,
                                     n_heads=2, d_key=16, d_value=16, d_route=16,
                                     anchor_interval=8, max_anchors=8, max_states=8))
        counts[mixer] = sum(p.numel() for p in lm.parameters())
    assert max(counts.values()) / min(counts.values()) < 1.5, counts
