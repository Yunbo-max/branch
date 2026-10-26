"""Pure-math pieces for SG-Delta: isotropic / Rayleigh / full directions and certificate.

Conventions match ``chunked_delta.py``: per head the state is key-major ``S in R^{K x V}``,
``k, q in R^K`` and the geometry is ``H = lambda I + mu C`` with ``C`` symmetric PSD.

The certificate bounds the *local one-edit direction error* between the exact full
geometry solve and the no-solve Rayleigh direction. It is not a global loss bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _validate(k: Tensor, C: Tensor, lam: float, mu: float, tau: float) -> None:
    if lam <= 0:
        raise ValueError("lambda must be positive")
    if tau <= 0:
        raise ValueError("tau must be positive")
    if mu < 0:
        raise ValueError("mu must be non-negative")
    if not torch.isfinite(k).all():
        raise ValueError("k contains non-finite values")
    if not torch.isfinite(C).all():
        raise ValueError("C contains non-finite values")
    if C.shape[-1] != C.shape[-2] or C.shape[-1] != k.shape[-1]:
        raise ValueError("C must be (...,K,K) and k must be (...,K)")


def symmetrize(C: Tensor) -> Tensor:
    return 0.5 * (C + C.transpose(-1, -2))


def isotropic_direction(k: Tensor, *, lam: float, tau: float) -> Tensor:
    """a_iso = k / (tau*lambda + ||k||^2)."""
    if lam <= 0 or tau <= 0:
        raise ValueError("lambda and tau must be positive")
    k2 = k.square().sum(-1, keepdim=True)
    return k / (tau * lam + k2)


def rayleigh_certificate(
    k: Tensor, C: Tensor, *, lam: float, mu: float, tau: float
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return ``(a_ray, upper_bound, c, rnorm)``.

    ``a_ray = k / (tau*(lambda + mu*c) + ||k||^2)`` with ``c = k^T C k / ||k||^2``.
    ``upper_bound`` is the proved bound on ``||a_full - a_ray||_2``.
    For exactly zero keys the direction and bound are zero.
    """
    _validate(k, C, lam, mu, tau)
    Cs = symmetrize(C)
    k2 = k.square().sum(-1)
    zero = k2 == 0
    safe_k2 = torch.where(zero, torch.ones_like(k2), k2)
    Ck = torch.matmul(k.unsqueeze(-2), Cs.transpose(-1, -2)).squeeze(-2)
    c = (k * Ck).sum(-1) / safe_k2
    r = Ck - c.unsqueeze(-1) * k
    rnorm = r.norm(dim=-1)
    D = tau * (lam + mu * c) + k2
    a_ray = k / D.unsqueeze(-1)
    upper = mu * rnorm / (lam * D)
    a_ray = torch.where(zero.unsqueeze(-1), torch.zeros_like(a_ray), a_ray)
    upper = torch.where(zero, torch.zeros_like(upper), upper)
    c = torch.where(zero, torch.zeros_like(c), c)
    rnorm = torch.where(zero, torch.zeros_like(rnorm), rnorm)
    return a_ray, upper, c, rnorm


def full_direction(
    k: Tensor, C: Tensor, *, lam: float, mu: float, tau: float
) -> Tensor:
    """a_full = (tau*H + k k^T)^{-1} k, computed with one Cholesky solve."""
    _validate(k, C, lam, mu, tau)
    Cs = symmetrize(C)
    K = k.shape[-1]
    eye = torch.eye(K, device=k.device, dtype=k.dtype)
    H = lam * eye + mu * Cs
    A = tau * H + k.unsqueeze(-1) * k.unsqueeze(-2)
    factor = torch.linalg.cholesky(A)
    rhs = k.unsqueeze(-1)
    return torch.cholesky_solve(rhs, factor).squeeze(-1)


def direction_stats(k: Tensor, C: Tensor, *, lam: float, mu: float, tau: float) -> dict:
    """Compare the three directions and report the certificate on one batch."""
    a_iso = isotropic_direction(k, lam=lam, tau=tau)
    a_ray, upper, c, rnorm = rayleigh_certificate(k, C, lam=lam, mu=mu, tau=tau)
    a_full = full_direction(k, C, lam=lam, mu=mu, tau=tau)
    err_ray = (a_full - a_ray).norm(dim=-1)
    err_iso = (a_full - a_iso).norm(dim=-1)
    return {
        "a_iso": a_iso,
        "a_ray": a_ray,
        "a_full": a_full,
        "upper": upper,
        "c": c,
        "rnorm": rnorm,
        "err_ray": err_ray,
        "err_iso": err_iso,
        "valid": bool((err_ray <= upper + 1e-10).all()),
    }


def group_easy(U: Tensor, threshold: float) -> Tensor:
    """Group is easy iff max token U <= threshold (group axis is the last one)."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    return U.max(dim=-1).values <= threshold


def k_is_eigenvector(k: Tensor, C: Tensor, *, atol: float = 1e-8) -> Tensor:
    """Diagnostic only: ||Ck - c k|| == 0 means k is a C eigenvector."""
    Cs = symmetrize(C)
    k2 = k.square().sum(-1)
    safe = torch.where(k2 == 0, torch.ones_like(k2), k2)
    Ck = torch.matmul(k.unsqueeze(-2), Cs.transpose(-1, -2)).squeeze(-2)
    c = (k * Ck).sum(-1) / safe
    r = Ck - c.unsqueeze(-1) * k
    return r.norm(dim=-1) <= atol
