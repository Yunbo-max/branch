"""Regression tests for the Branch fork, MARCH routing, config wiring and split."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))

from chunked_branch import ChunkedBranchConfig, ChunkedBranchMixer  # noqa: E402
from chunked_march import ChunkedMARCHConfig, ChunkedMARCHMixer  # noqa: E402
from data_split import TRAIN, VAL, RowSplit, build_panel, manifest  # noqa: E402
from fair_models import FairLMConfig, build_mixer  # noqa: E402
from pretrain_branch import PackedCorpus  # noqa: E402


def cfg(**kw) -> ChunkedBranchConfig:
    base = dict(
        d_model=16,
        n_heads=2,
        d_key=4,
        d_value=4,
        d_route=4,
        n_probes=1,
        max_states=4,
        conv_kernel=2,
        chunk_len=4,
    )
    base.update(kw)
    return ChunkedBranchConfig(**base)


def forced_route(parent_slot: int = 0, only: tuple[int, int] | None = None):
    def route(states, factor, active, key, value):
        B, T, H, _ = key.shape
        M = states.shape[2]
        parent = torch.full((B, T, H), parent_slot, dtype=torch.long)
        is_fork = torch.ones(B, T, H, dtype=torch.bool)
        if only is not None:
            is_fork = torch.zeros(B, T, H, dtype=torch.bool)
            is_fork[only[0], :, only[1]] = True
        return parent, is_fork, torch.zeros(B, T, H, 2 * M)

    return route


def test_allocate_children_single_and_multiple():
    M = 8
    active = torch.zeros(1, 1, M, dtype=torch.bool)
    active[..., 0] = True
    fork = torch.zeros(1, 1, M, dtype=torch.bool)
    fork[..., 0] = True
    child = ChunkedBranchMixer._allocate_children(active, fork)
    assert child[0, 0, 0].item() == 1
    assert (child[0, 0, 1:] < 0).all()

    active[..., 2] = True
    fork[..., 2] = True
    child = ChunkedBranchMixer._allocate_children(active, fork)
    assert child[0, 0, 0].item() == 1
    assert child[0, 0, 2].item() == 3


def test_allocate_children_full_pool():
    M = 4
    active = torch.ones(1, 1, M, dtype=torch.bool)
    fork = torch.ones(1, 1, M, dtype=torch.bool)
    assert (ChunkedBranchMixer._allocate_children(active, fork) < 0).all()


def test_route_matches_isotropic_geometry_at_init():
    torch.manual_seed(0)
    m = ChunkedBranchMixer(cfg())
    B, T, H, K, Vd, M = 1, 5, 2, 4, 4, 4
    states = torch.zeros(B, H, M, K, Vd)
    active = torch.zeros(B, H, M, dtype=torch.bool)
    active[..., 0] = True
    key = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1)
    value = torch.tanh(torch.randn(B, T, H, Vd))
    factor = torch.linalg.cholesky(torch.eye(K).expand(B, H, M, K, K))
    _, _, costs = m._route(states, factor, active, key, value)

    replace = costs[..., 0]
    fork = costs[..., M]
    assert torch.isfinite(replace).all()
    assert torch.allclose(fork, replace + m.config.fork_cost, atol=1e-6)


def test_forced_fork_grows_active_and_preserves_parent():
    torch.manual_seed(0)
    m = ChunkedBranchMixer(cfg())
    m._route = forced_route()
    x = torch.randn(2, 8, 16)
    _, cache, stats = m(x)

    assert cache.active.sum(-1).tolist() == [[3, 3], [3, 3]]
    assert stats["fork"].all()
    assert set(torch.unique(stats["destination"]).tolist()) == {1, 2}
    assert torch.isfinite(stats["write_cost"]).all()
    # all tokens forked, so parent 0 must keep its chunk-start value (zero)
    assert torch.count_nonzero(cache.states[:, :, 0]) == 0
    assert torch.count_nonzero(cache.states[:, :, 1]) > 0
    assert torch.count_nonzero(cache.states[:, :, 2]) > 0


def test_forced_fork_isolation_across_batch_and_heads():
    torch.manual_seed(0)
    m = ChunkedBranchMixer(cfg())
    m._route = forced_route(only=(0, 0))
    _, cache, stats = m(torch.randn(2, 8, 16))

    assert cache.active.sum(-1).tolist() == [[3, 1], [1, 1]]
    assert stats["fork"][0, :, 0].all()
    assert not stats["fork"][0, :, 1].any()
    assert not stats["fork"][1].any()


def test_full_pool_falls_back_to_replace():
    torch.manual_seed(0)
    m = ChunkedBranchMixer(cfg(max_states=1))
    m._route = forced_route()
    _, cache, stats = m(torch.randn(2, 8, 16))

    assert cache.active.sum(-1).tolist() == [[1, 1], [1, 1]]
    assert not stats["fork"].any()
    assert (stats["destination"] == 0).all()


def test_march_anchor_keys_are_content_conditioned():
    torch.manual_seed(0)
    m = ChunkedMARCHMixer(
        ChunkedMARCHConfig(
            d_model=16,
            n_heads=2,
            d_key=4,
            d_value=4,
            d_route=4,
            n_probes=1,
            max_states=4,
            conv_kernel=2,
            chunk_len=4,
            read_topk=2,
        )
    )
    x1 = torch.randn(1, 8, 16)
    x2 = torch.randn(1, 8, 16)
    _, c1, _ = m(x1)
    _, c2, _ = m(x2)

    assert c1.counts.item() == 2
    assert not torch.allclose(c1.anchor_k[0, 0], c1.anchor_k[0, 1])
    assert not torch.allclose(c1.anchor_k, c2.anchor_k)


def test_chunked_config_wiring():
    base = dict(
        mixer="chunked",
        vocab_size=64,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_key=4,
        d_value=4,
        d_route=4,
        max_states=4,
        chunk_len=4,
    )
    mixer = build_mixer(
        FairLMConfig(
            **base,
            fork_cost=0.5,
            covariance="diagonal",
            differentiate_covariance=False,
        )
    )
    assert mixer.config.fork_cost == 0.5
    assert mixer.config.covariance == "diagonal"
    assert mixer.config.differentiate_covariance is False

    default = build_mixer(FairLMConfig(**base))
    assert default.config.covariance == "full"
    assert default.config.differentiate_covariance is True


def test_diagonal_geometry_changes_factor():
    torch.manual_seed(0)
    full = ChunkedBranchMixer(cfg(covariance="full"))
    diag = ChunkedBranchMixer(cfg(covariance="diagonal"))
    B, H, M, K = 1, 2, 4, 4
    a = torch.randn(B, H, M, K, K)
    C = a @ a.transpose(-1, -2) + 0.1 * torch.eye(K)
    f_full = full._geometry_factor(C)
    f_diag = diag._geometry_factor(C)
    assert not torch.allclose(f_full, f_diag)

    Cdiag = torch.diag_embed(C.diagonal(dim1=-2, dim2=-1))
    assert torch.allclose(diag._geometry_factor(C), diag._geometry_factor(Cdiag))


def test_differentiate_covariance_controls_gradient():
    for flag in (False, True):
        torch.manual_seed(0)
        m = ChunkedBranchMixer(cfg(differentiate_covariance=flag)).double()
        B, L, H, K, Vd, M = 1, 4, 2, 4, 4, 4
        bank = torch.zeros(B, H, M, K, Vd, dtype=torch.double)
        C = torch.zeros(B, H, M, K, K, dtype=torch.double)
        C[:, :, 0] = torch.eye(K, dtype=torch.double) * 0.5
        C.requires_grad_(True)
        active = torch.zeros(B, H, M, dtype=torch.bool)
        active[..., 0] = True
        kk = torch.randn(B, L, H, K, dtype=torch.double)
        vv = torch.randn(B, L, H, Vd, dtype=torch.double)
        qq = torch.randn(B, L, H, K, dtype=torch.double)
        rr = torch.randn(B, L, H, 4, dtype=torch.double)
        nl = torch.zeros(B, L, H, 1, dtype=torch.double)
        addresses = torch.zeros(B, H, M, 4, dtype=torch.double)
        out, _, _, _, _ = m._chunk_step(
            bank, C, active, kk, vv, qq, rr, nl, addresses, L
        )
        if flag:
            assert out.requires_grad
            out.sum().backward()
            assert C.grad is not None
        else:
            assert not out.requires_grad
            assert C.grad is None


def test_row_split_disjoint_and_deterministic():
    split = RowSplit(seed=7, val_per_mille=200, test_per_mille=100)
    idx = np.arange(20000)
    labels = split.labels(idx)
    assert set(np.unique(labels).tolist()) == {TRAIN, VAL, 2}
    assert (split.labels(idx) == labels).all()
    other = RowSplit(seed=8, val_per_mille=200, test_per_mille=100)
    assert not (other.labels(idx) == labels).all()
    assert not np.intersect1d(idx[labels == VAL], idx[split.labels(idx) == TRAIN]).size
    assert not np.intersect1d(idx[labels == 2], idx[labels == VAL]).size


def test_fixed_panel_is_deterministic_and_held_out(tmp_path: Path):
    rng = np.random.default_rng(0)
    for shard in range(2):
        arr = rng.integers(0, 100, size=(200, 32), dtype=np.uint16)
        np.save(tmp_path / f"shard_{shard}.npy", arr)
    corpus = PackedCorpus(tmp_path, seq_len=16)
    split = RowSplit(seed=11, val_per_mille=500, test_per_mille=100)
    gen1 = torch.Generator().manual_seed(11 + 7)
    gen2 = torch.Generator().manual_seed(11 + 7)
    p1 = build_panel(corpus, 8, split, gen1)
    p2 = build_panel(corpus, 8, split, gen2)

    assert p1.tokens_sha256 == p2.tokens_sha256
    assert p1.indices_sha256 == p2.indices_sha256
    assert (split.labels(p1.indices) == VAL).all()

    train = split.train_indices(32, torch.Generator().manual_seed(1), len(corpus))
    assert (split.labels(train) == TRAIN).all()
    assert not np.intersect1d(train, p1.indices).size

    payload = manifest(split, corpus, p1, 16)
    assert payload["panel"]["tokens_sha256"] == p1.tokens_sha256
    assert payload["split"]["counts"]["val"] > 0
