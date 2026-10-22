"""The chunked delta update must equal the per-token recurrence exactly."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))

from chunked_delta import chunked_delta_reference, chunked_delta_update  # noqa: E402


def make_case(B, T, H, K, V, seed=0, dtype=torch.double):
    torch.manual_seed(seed)
    S0 = torch.randn(B, H, K, V, dtype=dtype)
    R = torch.randn(B, H, K, K, dtype=dtype)
    Hm = R @ R.mT + 0.5 * torch.eye(K, dtype=dtype)
    factor = torch.linalg.cholesky(Hm)
    Kt = torch.nn.functional.normalize(torch.randn(B, T, H, K, dtype=dtype), dim=-1)
    Vt = torch.randn(B, T, H, V, dtype=dtype)
    Q = torch.randn(B, T, H, K, dtype=dtype)
    return S0, factor, Kt, Vt, Q


@pytest.mark.parametrize("B,T,H,K,V", [
    (1, 1, 1, 4, 4),
    (2, 7, 2, 5, 3),
    (1, 16, 2, 8, 8),
    (2, 33, 3, 6, 4),
])
def test_chunked_equals_per_token_loop(B, T, H, K, V):
    S0, factor, Kt, Vt, Q = make_case(B, T, H, K, V)
    tau = 1.0
    out_ref, S_ref = chunked_delta_reference(S0, Kt, Vt, Q, factor, tau)
    out, S = chunked_delta_update(S0, Kt, Vt, Q, factor, tau, return_states=True)
    torch.testing.assert_close(out, out_ref, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(S, S_ref, atol=1e-11, rtol=1e-11)


def test_chunks_compose_to_a_long_run():
    """Two chunks chained must equal one long chunk (same frozen geometry)."""
    B, H, K, V = 1, 2, 6, 6
    S0, factor, Kt, Vt, Q = make_case(B, 20, H, K, V, seed=3)
    tau = 0.7
    out_full, S_full = chunked_delta_update(S0, Kt, Vt, Q, factor, tau, return_states=True)
    out1, S_mid = chunked_delta_update(S0, Kt[:, :8], Vt[:, :8], Q[:, :8], factor, tau,
                                       return_states=True)
    out2, S_end = chunked_delta_update(S_mid, Kt[:, 8:], Vt[:, 8:], Q[:, 8:], factor, tau,
                                       return_states=True)
    torch.testing.assert_close(torch.cat([out1, out2], 1), out_full, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(S_end, S_full, atol=1e-10, rtol=1e-10)


def test_causality_inside_a_chunk():
    B, T, H, K, V = 1, 12, 1, 5, 5
    S0, factor, Kt, Vt, Q = make_case(B, T, H, K, V, seed=4)
    out, _ = chunked_delta_update(S0, Kt, Vt, Q, factor, 1.0)
    K2, V2, Q2 = Kt.clone(), Vt.clone(), Q.clone()
    K2[:, 8:] += 2.0
    V2[:, 8:] -= 1.5
    Q2[:, 8:] *= 0.3
    out2, _ = chunked_delta_update(S0, K2, V2, Q2, factor, 1.0)
    torch.testing.assert_close(out[:, :8], out2[:, :8], atol=1e-12, rtol=1e-12)
    assert not torch.allclose(out[:, 8:], out2[:, 8:])


def test_gradients_flow():
    B, T, H, K, V = 1, 6, 1, 4, 4
    S0, factor, Kt, Vt, Q = make_case(B, T, H, K, V, seed=5)
    Kt.requires_grad_(); Vt.requires_grad_(); Q.requires_grad_()
    out, S = chunked_delta_update(S0, Kt, Vt, Q, factor, 1.0, return_states=True)
    (out.square().mean() + S.square().mean()).backward()
    for name, t in (("key", Kt), ("value", Vt), ("query", Q)):
        assert t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0, name


def test_shape_guards():
    B, T, H, K, V = 1, 4, 1, 3, 3
    S0, factor, Kt, Vt, Q = make_case(B, T, H, K, V)
    with pytest.raises(ValueError):
        chunked_delta_update(S0[:, :, :2], Kt, Vt, Q, factor, 1.0)
    with pytest.raises(ValueError):
        chunked_delta_update(S0, Kt, Vt[:, :, :, :2], Q, factor, 1.0)
