"""T0: analytic identities and certificate validity for SG-Delta."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))

from geometry_certificate import (  # noqa: E402
    direction_stats,
    full_direction,
    group_easy,
    isotropic_direction,
    k_is_eigenvector,
    rayleigh_certificate,
    symmetrize,
)


def spd(d: int, *, batch: int = 8, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(batch, d, d, generator=g, dtype=torch.float64)
    return A @ A.transpose(-1, -2) + 0.1 * torch.eye(d, dtype=torch.float64)


def test_bound_random_cases():
    torch.manual_seed(0)
    for d in (2, 4, 8, 32):
        for lam in (0.2, 1.0, 3.0):
            for mu in (0.0, 0.5, 8.0):
                for tau in (0.2, 1.0):
                    C = spd(d, batch=16, seed=d + int(10 * lam) + int(10 * mu))
                    k = torch.randn(16, d, dtype=torch.float64)
                    a_ray, upper, c, rnorm = rayleigh_certificate(
                        k, C, lam=lam, mu=mu, tau=tau
                    )
                    a_full = full_direction(k, C, lam=lam, mu=mu, tau=tau)
                    err = (a_full - a_ray).norm(dim=-1)
                    assert torch.isfinite(upper).all()
                    assert (err <= upper + 1e-9).all(), (
                        d,
                        lam,
                        mu,
                        tau,
                        (err - upper).max(),
                    )


def test_full_direction_matches_direct_solve():
    torch.manual_seed(1)
    d = 8
    C = spd(d, batch=4, seed=3)
    k = torch.randn(4, d, dtype=torch.float64)
    lam, mu, tau = 1.0, 8.0, 0.5
    H = lam * torch.eye(d, dtype=torch.float64) + mu * symmetrize(C)
    A = tau * H + k.unsqueeze(-1) * k.unsqueeze(-2)
    a = torch.linalg.solve(A, k.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(full_direction(k, C, lam=lam, mu=mu, tau=tau), a, atol=1e-11)


def test_non_normalized_keys():
    C = spd(4, batch=3, seed=7)
    k = 3.5 * torch.randn(3, 4, dtype=torch.float64)
    a_ray, upper, c, _ = rayleigh_certificate(k, C, lam=1.0, mu=2.0, tau=1.0)
    k2 = k.square().sum(-1)
    D = 1.0 * (1.0 + 2.0 * c) + k2
    assert torch.allclose(a_ray, k / D.unsqueeze(-1), atol=1e-12)
    a_iso = isotropic_direction(k, lam=1.0, tau=1.0)
    assert not torch.allclose(a_iso, a_ray)


def test_eigenvector_is_exact_rayleigh_case():
    d = 8
    g = torch.Generator().manual_seed(11)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    eig = torch.tensor(
        [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0], dtype=torch.float64
    )
    C = Q @ torch.diag(eig) @ Q.T
    k = Q[:, 7] * 2.3
    a_ray, upper, c, rnorm = rayleigh_certificate(k, C, lam=1.0, mu=8.0, tau=1.0)
    a_full = full_direction(k, C, lam=1.0, mu=8.0, tau=1.0)
    assert rnorm < 1e-10
    assert upper < 1e-10
    assert torch.allclose(a_ray, a_full, atol=1e-10)
    assert bool(k_is_eigenvector(k, C))


def test_zero_covariance_and_zero_mu():
    d = 4
    k = torch.randn(d, dtype=torch.float64)
    C0 = torch.zeros(d, d, dtype=torch.float64)
    a_ray, upper, c, rnorm = rayleigh_certificate(k, C0, lam=1.0, mu=8.0, tau=1.0)
    a_full = full_direction(k, C0, lam=1.0, mu=8.0, tau=1.0)
    a_iso = isotropic_direction(k, lam=1.0, tau=1.0)
    assert upper == 0 and c == 0 and rnorm == 0
    assert torch.allclose(a_ray, a_iso, atol=1e-12)
    assert torch.allclose(a_full, a_iso, atol=1e-12)

    C = spd(d, batch=1, seed=5)[0]
    a_ray, upper, _, _ = rayleigh_certificate(k, C, lam=1.0, mu=0.0, tau=1.0)
    a_full = full_direction(k, C, lam=1.0, mu=0.0, tau=1.0)
    assert upper == 0
    assert torch.allclose(a_ray, a_full, atol=1e-12)


def test_off_diagonal_rotation_gives_positive_residual():
    d = 4
    R = torch.tensor(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    C = R @ torch.diag(torch.tensor([1.0, 5.0, 20.0, 100.0], dtype=torch.float64)) @ R.T
    k = torch.tensor([1.0, 0.3, 0.2, 0.1], dtype=torch.float64)
    a_ray, upper, c, rnorm = rayleigh_certificate(k, C, lam=1.0, mu=8.0, tau=1.0)
    a_full = full_direction(k, C, lam=1.0, mu=8.0, tau=1.0)
    assert rnorm > 0 and upper > 0
    assert not torch.allclose(a_ray, a_full, atol=1e-10)


def test_orthogonal_equivariance():
    torch.manual_seed(2)
    d = 8
    C = spd(d, batch=1, seed=9)[0]
    k = torch.randn(d, dtype=torch.float64)
    R, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))
    C2 = R @ C @ R.T
    k2 = R @ k
    a_full = full_direction(k, C, lam=1.0, mu=8.0, tau=0.5)
    a_full2 = full_direction(k2, C2, lam=1.0, mu=8.0, tau=0.5)
    a_ray, upper, _, _ = rayleigh_certificate(k, C, lam=1.0, mu=8.0, tau=0.5)
    a_ray2, upper2, _, _ = rayleigh_certificate(k2, C2, lam=1.0, mu=8.0, tau=0.5)
    assert torch.allclose(a_full2, R @ a_full, atol=1e-10)
    assert torch.allclose(a_ray2, R @ a_ray, atol=1e-10)
    assert torch.allclose(upper2, upper, atol=1e-10)


def test_validation_and_zero_key():
    k = torch.randn(4, dtype=torch.float64)
    C = spd(4, batch=1, seed=1)[0]
    for kwargs in (
        dict(lam=0.0, mu=1.0, tau=1.0),
        dict(lam=1.0, mu=-1.0, tau=1.0),
        dict(lam=1.0, mu=1.0, tau=0.0),
    ):
        with pytest.raises(ValueError):
            rayleigh_certificate(k, C, **kwargs)
    bad = k.clone()
    bad[0] = float("nan")
    with pytest.raises(ValueError):
        rayleigh_certificate(bad, C, lam=1.0, mu=1.0, tau=1.0)

    z = torch.zeros(4, dtype=torch.float64)
    a_ray, upper, c, rnorm = rayleigh_certificate(z, C, lam=1.0, mu=8.0, tau=1.0)
    a_full = full_direction(z, C, lam=1.0, mu=8.0, tau=1.0)
    assert torch.isfinite(a_ray).all() and a_ray.abs().sum() == 0
    assert upper == 0 and a_full.abs().sum() == 0


def test_fp32_stability_against_fp64():
    torch.manual_seed(3)
    d = 64
    C64 = spd(d, batch=4, seed=13)
    k64 = torch.randn(4, d, dtype=torch.float64)
    a_full64 = full_direction(k64, C64, lam=1.0, mu=8.0, tau=1.0)
    a_ray64, upper64, _, _ = rayleigh_certificate(k64, C64, lam=1.0, mu=8.0, tau=1.0)

    C32, k32 = C64.float(), k64.float()
    a_full32 = full_direction(k32, C32, lam=1.0, mu=8.0, tau=1.0)
    a_ray32, upper32, _, _ = rayleigh_certificate(k32, C32, lam=1.0, mu=8.0, tau=1.0)
    err32 = (a_full32 - a_ray32).norm(dim=-1)
    assert torch.isfinite(a_full32).all() and torch.isfinite(upper32).all()
    assert (err32 <= upper32 + 1e-3).all()
    assert torch.allclose(a_full32, a_full64.float(), atol=2e-2, rtol=2e-2)


def test_group_easy_is_max_over_tokens():
    U = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.2]], dtype=torch.float64)
    assert group_easy(U, 0.15).tolist() == [True, False]
    assert group_easy(U, 0.2).tolist() == [True, True]


def test_direction_stats_reports_valid_certificate():
    C = spd(8, batch=4, seed=21)
    k = torch.randn(4, 8, dtype=torch.float64)
    stats = direction_stats(k, C, lam=1.0, mu=8.0, tau=1.0)
    assert stats["valid"]
    assert (stats["err_ray"] <= stats["upper"] + 1e-10).all()
