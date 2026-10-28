"""Regression: 2-pair seq64 / chunk64 NEVER invokes nonzero Query geometry.

Use the SAME SingleGeometryMixer architecture and weights for Full vs Isotropic;
comparing full-C against standalone DeltaNetMixer also changes the model and
initialization and cannot establish the causal contribution of C.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.testing import assert_close

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fair_models import FairLMConfig, build_mixer  # noqa: E402
from train_fair import add_common_args, default_config  # noqa: E402


def make(mode: str, *, chunk_len: int = 64):
    cfg = FairLMConfig(
        mixer="single_geo", d_model=16, n_heads=2,
        d_key=8, d_value=8, d_route=8, max_states=1,
        chunk_len=chunk_len, geometry_mode=mode,
        read_mode="current_only", protect=8.0,
        differentiate_covariance=True,
    )
    return build_mixer(cfg).double().eval()


def matched_models(*, chunk_len: int):
    torch.manual_seed(1212)
    iso = make("isotropic", chunk_len=chunk_len)
    full = make("full", chunk_len=chunk_len)
    full.load_state_dict(iso.state_dict(), strict=True)
    return iso, full


def test_full_C_is_inactive_for_one_64_token_chunk():
    iso, full = matched_models(chunk_len=64)
    x = torch.randn(2, 64, 16, dtype=torch.float64)
    out_i, cache_i, _ = iso(x)
    out_f, cache_f, _ = full(x)
    # Full C is zero at chunk start, and the whole sequence is exactly one chunk.
    assert_close(out_i, out_f, atol=1e-10, rtol=1e-10)
    assert_close(cache_i.states, cache_f.states, atol=1e-10, rtol=1e-10)
    assert torch.count_nonzero(cache_i.protection) == 0
    assert cache_f.protection.abs().sum() > 0  # only available AFTER the chunk
    grads_i = torch.autograd.grad(out_i.square().mean(), tuple(iso.parameters()),
                                  allow_unused=True)
    grads_f = torch.autograd.grad(out_f.square().mean(), tuple(full.parameters()),
                                  allow_unused=True)
    for g_i, g_f in zip(grads_i, grads_f):
        if g_i is None or g_f is None:
            assert g_i is None and g_f is None
        else:
            assert_close(g_i, g_f, atol=1e-9, rtol=1e-9)


def test_nonzero_C_affects_only_later_chunks():
    iso, full = matched_models(chunk_len=16)
    x = torch.randn(2, 64, 16, dtype=torch.float64)
    y_i, _, _ = iso(x)
    y_f, cf, _ = full(x)
    assert_close(y_i[:, :16], y_f[:, :16], atol=1e-10, rtol=1e-10)
    assert cf.protection.abs().sum() > 0
    assert (y_i[:, 16:] - y_f[:, 16:]).abs().max().item() > 1e-8


def test_train_fair_exposes_same_architecture_ablation():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args([
        "--mixer", "single_geo", "--output", "/tmp/mqar_unused",
        "--geometry-mode", "isotropic", "--read-mode", "current_only",
        "--chunk-len", "16",
    ])
    cfg = default_config(args)
    assert cfg.geometry_mode == "isotropic"
    assert cfg.read_mode == "current_only"
    assert cfg.chunk_len == 16
