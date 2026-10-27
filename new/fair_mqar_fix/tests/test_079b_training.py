"""0.79B preflight: exact tiled GeoDelta, FP32 geometry, DDP sync, 4K sampling.

CPU-size regressions; full 8-GPU H200 benchmarking is a separate gate.
"""
from __future__ import annotations

from contextlib import contextmanager
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pretrain_branch import PackedCorpus, grad_sync_context  # noqa: E402
from fair_models import FairLMConfig, build_mixer  # noqa: E402


def build(tile: int, *, decay: float = 0.95):
    return build_mixer(FairLMConfig(
        mixer="single_geo", vocab_size=64, d_model=16, n_layers=1,
        n_heads=2, d_key=4, d_value=4, d_route=4,
        max_states=1, chunk_len=8, delta_tile_len=tile,
        geometry_mode="full", protect=8.0,
        protection_decay=decay, differentiate_covariance=True,
        grad_checkpoint=False,
    ))


@pytest.mark.parametrize("tile", [1, 2, 4])
def test_tiled_same_forward_cache_backward_as_whole(tile):
    torch.manual_seed(123)
    ref = build(0).train()
    alt = build(tile).train()
    alt.load_state_dict(ref.state_dict())
    x = torch.randn(2, 17, 16)
    x_ref = x.clone().requires_grad_(True)
    x_alt = x.clone().requires_grad_(True)
    y_ref, cache_ref, _ = ref(x_ref)
    y_alt, cache_alt, _ = alt(x_alt)
    torch.testing.assert_close(y_ref, y_alt, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(cache_ref.states, cache_alt.states, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(cache_ref.protection, cache_alt.protection, atol=3e-5, rtol=3e-5)
    (y_ref.square().sum()).backward()
    (y_alt.square().sum()).backward()
    torch.testing.assert_close(x_ref.grad, x_alt.grad, atol=1e-4, rtol=1e-4)
    for (name0, p0), (name1, p1) in zip(ref.named_parameters(), alt.named_parameters()):
        assert name0 == name1
        if p0.grad is None or p1.grad is None:
            assert p0.grad is None and p1.grad is None
        else:
            torch.testing.assert_close(p0.grad, p1.grad, atol=1e-4, rtol=1e-4)


def test_4k_random_crop_reproducible_and_not_prefix(tmp_path):
    data = np.arange(16384, dtype=np.uint16)[None, :]
    np.save(tmp_path / "shard.npy", data)
    corpus = PackedCorpus(tmp_path, 4096, random_crop=True)
    a = corpus.sample(8, torch.Generator().manual_seed(99))
    b = corpus.sample(8, torch.Generator().manual_seed(99))
    assert a.shape == (8, 4096)
    np.testing.assert_array_equal(a, b)
    assert all(np.all(np.diff(row) == 1) for row in a)
    assert len(set(a[:, 0].tolist())) > 1
    assert np.any(a[:, 0] != 0)
    np.testing.assert_array_equal(corpus.gather(np.array([0]))[0], data[0, :4096])


def test_4k_cropping_can_be_disabled(tmp_path):
    data = np.arange(16384, dtype=np.uint16)[None, :]
    np.save(tmp_path / "shard.npy", data)
    corpus = PackedCorpus(tmp_path, 4096, random_crop=False)
    samples = corpus.sample(2, torch.Generator().manual_seed(3))
    np.testing.assert_array_equal(samples[0], data[0, :4096])


def test_invalid_crop_rejected(tmp_path):
    np.save(tmp_path / "shard.npy", np.arange(1024, dtype=np.uint16)[None])
    with pytest.raises(SystemExit):
        PackedCorpus(tmp_path, 2048, random_crop=True)


def test_ddp_no_sync_uses_only_last_microbatch_to_reduce():
    class FakeDDP:
        calls = 0

        @contextmanager
        def no_sync(self):
            self.calls += 1
            yield

    model = FakeDDP()
    with grad_sync_context(model, ddp=True, micro_idx=0, grad_accum=4):
        assert model.calls == 1
    with grad_sync_context(model, ddp=True, micro_idx=2, grad_accum=4):
        assert model.calls == 2
    with grad_sync_context(model, ddp=True, micro_idx=3, grad_accum=4):
        assert model.calls == 2
    with grad_sync_context(model, ddp=False, micro_idx=0, grad_accum=4):
        assert model.calls == 2


def test_custom_protection_decay_is_wired_into_geo():
    model = build(2, decay=0.999)
    assert model.config.protection_decay == pytest.approx(0.999)


def test_bf16_autocast_keeps_geometry_and_state_fp32():
    torch.manual_seed(5)
    model = build(2).eval()
    x = torch.randn(1, 9, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y, cache, _ = model(x)
    assert torch.isfinite(y).all()
    assert cache.states.dtype == torch.float32
    assert cache.protection.dtype == torch.float32
