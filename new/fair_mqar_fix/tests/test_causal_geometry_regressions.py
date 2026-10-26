"""Regression coverage for causal fork allocation and efficient GeoDelta chunks.

All tests are CPU-scale, deterministic, and require no pretraining corpus.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import single_geometry as single_geo_module  # noqa: E402
from chunked_branch import ChunkedBranchConfig, ChunkedBranchMixer  # noqa: E402
from single_geometry import SingleGeometryConfig, SingleGeometryMixer  # noqa: E402


def single_model(*, mode="full", chunk_len=4, checkpoint=False):
    return SingleGeometryMixer(SingleGeometryConfig(
        d_model=16, n_heads=2, d_key=4, d_value=4, d_route=4,
        max_states=1, conv_kernel=2, chunk_len=chunk_len,
        geometry_mode=mode, grad_checkpoint=checkpoint,
    ))


def test_full_prefill_factors_once_per_complete_chunk(monkeypatch):
    torch.manual_seed(0)
    model = single_model().eval()
    x = torch.randn(1, 8, 16)
    calls = []
    original = torch.linalg.cholesky

    def counted(a, *args, **kwargs):
        calls.append(tuple(a.shape))
        return original(a, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky", counted)
    with torch.no_grad():
        model(x)
    assert len(calls) == 2, "full prefill must factor only once per chunk"


def test_full_prefill_with_tail_does_not_prefactor_future_chunk(monkeypatch):
    torch.manual_seed(0)
    model = single_model().eval()
    x = torch.randn(1, 9, 16)  # two complete chunks, one token in next chunk
    calls = []
    original = torch.linalg.cholesky

    def counted(a, *args, **kwargs):
        calls.append(tuple(a.shape))
        return original(a, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky", counted)
    with torch.no_grad():
        model(x)
    assert len(calls) == 3


def test_full_token_streaming_factors_only_on_demand(monkeypatch):
    torch.manual_seed(0)
    model = single_model().eval()
    x = torch.randn(1, 8, 16)
    calls = []
    original = torch.linalg.cholesky

    def counted(a, *args, **kwargs):
        calls.append(tuple(a.shape))
        return original(a, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky", counted)
    cache = None
    with torch.no_grad():
        for t in range(8):
            _, cache, _ = model(x[:, t:t+1], cache)
    assert len(calls) == 2
    assert cache.chunk_pos == 0


@pytest.mark.parametrize("mode", ["isotropic", "rayleigh", "full"])
def test_nonuniform_streaming_and_final_cache_match(mode):
    torch.manual_seed(0)
    model = single_model(mode=mode).eval()
    x = torch.randn(1, 17, 16)
    with torch.no_grad():
        full, full_cache, _ = model(x)
        parts, cache, off = [], None, 0
        for length in (1, 3, 2, 5, 6):
            out, cache, _ = model(x[:, off:off + length], cache)
            parts.append(out)
            off += length
    assert off == x.shape[1]
    torch.testing.assert_close(torch.cat(parts, 1), full, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(cache.states, full_cache.states, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(cache.chunk_protection, full_cache.chunk_protection,
                               atol=3e-5, rtol=3e-5)
    assert cache.chunk_pos == full_cache.chunk_pos == 1


def test_grad_checkpoint_path_called_and_backward_finite(monkeypatch):
    torch.manual_seed(0)
    model = single_model(checkpoint=True).train()
    calls = []
    actual = single_geo_module.checkpoint

    def counted(*args, **kwargs):
        calls.append(True)
        return actual(*args, **kwargs)

    monkeypatch.setattr(single_geo_module, "checkpoint", counted)
    x = torch.randn(1, 8, 16, requires_grad=True)
    out, _, _ = model(x)
    out.square().mean().backward()
    assert len(calls) == 2
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert model.k_proj.weight.grad is not None


def test_certified_is_rejected_for_autoregressive_lm():
    model = single_model(mode="certified")
    with pytest.raises(ValueError, match="non-causal"):
        model(torch.randn(1, 4, 16))


def test_causal_allocator_first_fork_wins():
    active = torch.tensor([[[True, True, False]]])
    any_fork = torch.tensor([[[True, True, False]]])
    # parent 1 first requests at token 0; parent 0 only at token 3.
    first = torch.tensor([[[3, 0, 4]]])
    got = ChunkedBranchMixer._allocate_children(
        active, any_fork, first_request=first
    )
    assert got.tolist() == [[[-1, 2, -1]]]

    # Removing a future fork must not change the earlier allocation.
    only_first = ChunkedBranchMixer._allocate_children(
        active, torch.tensor([[[False, True, False]]]),
        first_request=torch.tensor([[[4, 0, 4]]]),
    )
    assert only_first.tolist() == got.tolist()


def test_causal_allocator_tie_break_and_batch_isolation():
    active = torch.tensor([
        [[True, True, False, False], [True, True, True, False]],
        [[True, True, False, False], [True, True, True, True]],
    ])
    requests = torch.tensor([
        [[True, True, False, False], [True, True, True, False]],
        [[True, True, False, False], [True, True, True, True]],
    ])
    first = torch.tensor([
        [[3, 0, 5, 5], [0, 1, 2, 5]],
        [[0, 3, 5, 5], [0, 1, 2, 3]],
    ])
    got = ChunkedBranchMixer._allocate_children(
        active, requests, first_request=first
    )
    assert got[0, 0].tolist() == [3, 2, -1, -1]
    assert got[0, 1].tolist() == [3, -1, -1, -1]
    assert got[1, 0].tolist() == [2, 3, -1, -1]
    assert (got[1, 1] < 0).all()


def test_future_fork_does_not_change_first_token_output():
    torch.manual_seed(3)
    m = ChunkedBranchMixer(ChunkedBranchConfig(
        d_model=16, n_heads=2, d_key=4, d_value=4, d_route=4,
        max_states=3, chunk_len=4, conv_kernel=2
    )).eval()
    x = torch.randn(1, 4, 16)
    cache = m.initial_cache(x)
    cache.active[:, :, 1] = True
    # Force replace vs fork to have distinct geometries.
    cache.protection[:, :, 1] = 4 * torch.eye(4)

    def force_route(with_later_fork):
        def route(states, factor, active, key, value):
            B, T, H, _ = key.shape
            M = states.shape[2]
            parent = torch.zeros(B, T, H, device=key.device, dtype=torch.long)
            parent[:, 0] = 1
            is_fork = torch.zeros(B, T, H, device=key.device, dtype=torch.bool)
            is_fork[:, 0] = True
            if with_later_fork:
                is_fork[:, -1] = True  # different parent (0) asks later
            return parent, is_fork, torch.zeros(B, T, H, 2*M, device=key.device)
        return route

    with torch.no_grad():
        m._route = force_route(False)
        out0, _, s0 = m(x, cache)
        m._route = force_route(True)
        out1, _, s1 = m(x, cache)
    assert (s0["destination"][:, 0] == 2).all()
    assert (s1["destination"][:, 0] == 2).all()
    torch.testing.assert_close(out0[:, 0], out1[:, 0], atol=1e-7, rtol=0)


def test_allocator_matches_online_reference_over_random_events():
    """The parallel first-event assignment must match online FCFS allocation."""
    torch.manual_seed(13)
    B, H, M, T = 2, 3, 8, 12
    for _ in range(100):
        active = torch.rand(B, H, M) > 0.45
        active[..., 0] = True
        parents = torch.empty(B, H, T, dtype=torch.long)
        for b in range(B):
            for h in range(H):
                options = torch.nonzero(active[b, h], as_tuple=True)[0]
                parents[b, h] = options[torch.randint(len(options), (T,))]
        forks = torch.rand(B, H, T) < 0.5
        events = (
            torch.nn.functional.one_hot(parents, M).bool()
            & forks.unsqueeze(-1)
        )
        first = torch.where(
            events, torch.arange(T).view(1, 1, T, 1), T
        ).amin(dim=2)
        result = ChunkedBranchMixer._allocate_children(
            active, first < T, first_request=first
        )
        for b in range(B):
            for h in range(H):
                free = [j for j in range(M) if not active[b, h, j]]
                online = [-1] * M
                for t in range(T):
                    parent = int(parents[b, h, t])
                    if bool(forks[b, h, t]) and online[parent] < 0 and free:
                        online[parent] = free.pop(0)
                assert result[b, h].tolist() == online


def test_grad_checkpoint_matches_noncheckpoint_gradients():
    """Activation recomputation must not alter parameter or input gradients."""
    torch.manual_seed(9)
    plain = single_model(checkpoint=False).train()
    wrapped = single_model(checkpoint=True).train()
    wrapped.load_state_dict(plain.state_dict())
    original = torch.randn(1, 8, 16)
    xa = original.clone().requires_grad_(True)
    xb = original.clone().requires_grad_(True)
    pa, _, _ = plain(xa)
    pb, _, _ = wrapped(xb)
    torch.testing.assert_close(pa, pb, atol=2e-5, rtol=2e-5)
    pa.square().mean().backward()
    pb.square().mean().backward()
    torch.testing.assert_close(xa.grad, xb.grad, atol=2e-5, rtol=2e-5)
    for (an, ap), (bn, bp) in zip(plain.named_parameters(), wrapped.named_parameters()):
        assert an == bn
        if ap.grad is None or bp.grad is None:
            assert ap.grad is None and bp.grad is None
        else:
            torch.testing.assert_close(ap.grad, bp.grad, atol=2e-5, rtol=2e-5)
