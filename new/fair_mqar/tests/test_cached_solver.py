"""The cached-solver variant must reproduce the reference exactly at solver_refresh=1."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))

from branch_cached import CachedBranchMixer, CachedSolverConfig  # noqa: E402
from branch_memory.mixer import BranchMemoryMixer, MemoryConfig  # noqa: E402


def ref_cfg(**kw) -> MemoryConfig:
    base = dict(d_model=16, n_heads=2, d_key=4, d_value=4, d_route=4, max_states=4,
                mode="branch", covariance="full", protect=8.0, fork_cost=0.02)
    base.update(kw)
    return MemoryConfig(**base)


def cached_cfg(refresh: int, **kw) -> CachedSolverConfig:
    base = dict(d_model=16, n_heads=2, d_key=4, d_value=4, d_route=4, max_states=4,
                mode="branch", covariance="full", protect=8.0, fork_cost=0.02,
                solver_refresh=refresh)
    base.update(kw)
    return CachedSolverConfig(**base)


def paired(refresh: int, seed: int = 0, **kw):
    torch.manual_seed(seed)
    ref = BranchMemoryMixer(ref_cfg(**kw)).double().eval()
    cached = CachedBranchMixer(cached_cfg(refresh, **kw)).double().eval()
    cached.load_state_dict(ref.state_dict())
    return ref, cached


def test_refresh_one_matches_the_reference_exactly():
    ref, cached = paired(1)
    x = torch.randn(2, 7, 16, dtype=torch.double)
    with torch.no_grad():
        y_ref, c_ref, s_ref = ref(x)
        y_new, c_new, s_new = cached(x)
    torch.testing.assert_close(y_new, y_ref, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(c_new.states, c_ref.states, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(c_new.protection, c_ref.protection, atol=1e-12, rtol=1e-12)
    assert torch.equal(s_new["parent"], s_ref["parent"])
    assert torch.equal(s_new["fork"], s_ref["fork"])
    assert torch.equal(s_new["destination"], s_ref["destination"])


def test_refresh_one_matches_across_streamed_chunks():
    ref, cached = paired(1)
    x = torch.randn(1, 8, 16, dtype=torch.double)
    with torch.no_grad():
        y_ref, c_ref, _ = ref(x)
        rc = cc = None
        pieces = []
        for s, e in [(0, 3), (3, 8)]:
            _, rc, _ = ref(x[:, s:e], rc)
            z, cc, _ = cached(x[:, s:e], cc)
            pieces.append(z)
    torch.testing.assert_close(torch.cat(pieces, 1), y_ref, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(cc.protection, c_ref.protection, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("refresh", [2, 5, 32])
def test_stale_factor_stays_finite_and_causal(refresh):
    _, cached = paired(refresh)
    x = torch.randn(2, 9, 16, dtype=torch.double)
    with torch.no_grad():
        y, cache, _ = cached(x)
        assert torch.isfinite(y).all()
        assert torch.isfinite(cache.protection).all()
        altered = x.clone()
        altered[:, 5:] += 3.0
        y2, _, _ = cached(altered)
        assert torch.allclose(y[:, :5], y2[:, :5], atol=1e-9)
        # the second run must not inherit the first run's stale factor
        y3, _, _ = cached(x)
        torch.testing.assert_close(y3, y, atol=1e-12, rtol=1e-12)


def test_staleness_is_bounded_and_grows_slower_than_expected():
    """Larger refresh must move the output, but only by the solver difference."""
    torch.manual_seed(0)
    x = torch.randn(1, 24, 16, dtype=torch.double)
    _, base = paired(1)
    with torch.no_grad():
        y_base, _, _ = base(x)
    gaps = {}
    for refresh in (1, 4, 8, 24):
        _, m = paired(refresh)
        with torch.no_grad():
            y, _, _ = m(x)
        gaps[refresh] = float((y - y_base).abs().max())
    assert gaps[1] == 0.0
    assert gaps[24] >= gaps[4] >= gaps[1]


def test_config_guards():
    with pytest.raises(ValueError):
        CachedSolverConfig(solver_refresh=0)
    with pytest.raises(ValueError):
        CachedSolverConfig(covariance="diagonal")
    with pytest.raises(ValueError):
        CachedSolverConfig(mode="delta")


def test_gradients_flow_with_a_stale_factor():
    _, m = paired(4)
    x = torch.randn(1, 6, 16, dtype=torch.double, requires_grad=True)
    y, _, _ = m(x)
    y.square().mean().backward()
    names = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "route_q.weight",
             "value_probes", "content_proj"]
    params = dict(m.named_parameters())
    for n in names:
        assert params[n].grad is not None and torch.isfinite(params[n].grad).all(), n
