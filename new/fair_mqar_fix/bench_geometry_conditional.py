#!/usr/bin/env python3
"""SG-Delta stage G3: measure the actual geometry cost and group-easy fraction.

For each (d, L) this measures, on the same device and dtype:

* Cholesky of H = lambda I + mu C (once per group)
* exact H^{-1}K triangular solve for L tokens
* C K for L tokens (needed by the Rayleigh certificate)
* full direction path: Cholesky + solve + scalar
* Rayleigh path: C K + scalar, no Cholesky

It also reports the fraction of groups that would be "easy" under the group-max rule
``max_t U_t <= epsilon`` for thresholds taken from token-level U quantiles. This is the
number that decides whether conditional skipping can save anything in practice.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from geometry_certificate import rayleigh_certificate, symmetrize  # noqa: E402


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--d-grid", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--chunk-grid", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--groups", type=int, default=256)
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--mu", type=float, default=8.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    return p.parse_args()


def timed(fn, *, reps: int, warmup: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / reps


def make_case(d: int, L: int, *, seed: int, device, dtype, lam, mu, gamma):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(L, d, generator=g).to(device=device, dtype=dtype)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = torch.randn(L, d, generator=g).to(device=device, dtype=dtype)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    powers = gamma ** torch.arange(L - 1, -1, -1, device=device, dtype=dtype)
    # C = (1-gamma) * sum_s gamma^{L-1-s} q_s q_s^T
    C = (1.0 - gamma) * ((q * powers.unsqueeze(-1)).T @ q)
    C = symmetrize(C)
    H = lam * torch.eye(d, device=device, dtype=dtype) + mu * C
    return q, k, C, H


def measure_point(a: argparse.Namespace, d: int, L: int, device, dtype) -> dict:
    q, k, C, H = make_case(
        d,
        L,
        seed=1000 + d * 7 + L,
        device=device,
        dtype=dtype,
        lam=a.lam,
        mu=a.mu,
        gamma=a.gamma,
    )
    Kt = k.T.contiguous()
    eye = torch.eye(d, device=device, dtype=dtype)

    def chol():
        return torch.linalg.cholesky(H)

    factor = torch.linalg.cholesky(H)

    def solve():
        return torch.cholesky_solve(Kt, factor)

    def ck():
        return C @ Kt

    def ray():
        return rayleigh_certificate(k, C, lam=a.lam, mu=a.mu, tau=a.tau)

    def full():
        f = torch.linalg.cholesky(H)
        u = torch.cholesky_solve(Kt, f)
        denom = a.tau + (k * u.T).sum(-1, keepdim=True)
        return u / denom.T

    def easy():
        return rayleigh_certificate(k, C, lam=a.lam, mu=a.mu, tau=a.tau)[0]

    t_chol = timed(chol, reps=a.reps, warmup=a.warmup, device=device)
    t_solve = timed(solve, reps=a.reps, warmup=a.warmup, device=device)
    t_ck = timed(ck, reps=a.reps, warmup=a.warmup, device=device)
    t_ray = timed(ray, reps=a.reps, warmup=a.warmup, device=device)
    t_full = timed(full, reps=a.reps, warmup=a.warmup, device=device)
    t_easy = timed(easy, reps=a.reps, warmup=a.warmup, device=device)

    # Group-easy fractions on independent key groups, same C.
    g = torch.Generator(device="cpu").manual_seed(77 + d)
    U = []
    for _ in range(a.groups):
        kk = torch.randn(L, d, generator=g).to(device=device, dtype=dtype)
        kk = kk / kk.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        _, upper, _, _ = rayleigh_certificate(kk, C, lam=a.lam, mu=a.mu, tau=a.tau)
        U.append(upper)
    U = torch.stack(U)  # [G,L]
    flat = U.flatten()
    quantiles = {
        f"p{int(p * 100)}": float(torch.quantile(flat, p))
        for p in (0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
    }
    fractions = {}
    for name, eps in quantiles.items():
        easy_groups = (U.max(dim=-1).values <= eps).float().mean().item()
        exact_tokens = (flat > eps).float().mean().item()
        fractions[name] = {
            "threshold": eps,
            "group_easy_fraction": easy_groups,
            "token_exact_fraction": exact_tokens,
        }
    return {
        "d": d,
        "L": L,
        "ms_per_group": {
            "cholesky": t_chol,
            "solve": t_solve,
            "Ck": t_ck,
            "rayleigh": t_ray,
            "full": t_full,
            "easy": t_easy,
        },
        "speedup_ceiling_easy_vs_full": t_full / max(t_easy, 1e-12),
        "cholesky_share_of_full": t_chol / max(t_full, 1e-12),
        "u_quantiles": quantiles,
        "group_easy": fractions,
        "groups": a.groups,
        "note": "times are per (batch,head,chunk) for one group on this device",
    }


def main() -> int:
    a = parse()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(a.device)
    dtype = torch.float64 if a.dtype == "float64" else torch.float32
    out = {
        "device": torch.cuda.get_device_name(0) if a.device == "cuda" else "cpu",
        "torch": torch.__version__,
        "config": {
            "lam": a.lam,
            "mu": a.mu,
            "tau": a.tau,
            "gamma": a.gamma,
            "dtype": a.dtype,
            "reps": a.reps,
            "groups": a.groups,
        },
        "points": [],
    }
    for d in a.d_grid:
        for L in a.chunk_grid:
            point = measure_point(a, d, L, device, dtype)
            out["points"].append(point)
            print(json.dumps(point), flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "perf.json").write_text(json.dumps(out, indent=2))
    print("PERF_JSON " + str(a.output / "perf.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
