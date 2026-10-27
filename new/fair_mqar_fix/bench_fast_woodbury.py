"""Measure dense Query geometry versus rank-r Woodbury geometry.

Scope: geometry direction construction only, NOT a training step or a comparison
to fused Gated DeltaNet. Run on H200 and report forward + backward separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F

from fast_woodbury_geometry import (
    dense_ema_direction, truncation_bound, woodbury_direction
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--d-key", type=int, default=256)
    p.add_argument("--history", type=int, default=256)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--ranks", nargs="+", type=int, default=[32, 64, 96, 128, 160, 192])
    p.add_argument("--gamma", type=float, default=.95)
    p.add_argument("--mu", type=float, default=8.)
    p.add_argument("--lam", type=float, default=1.)
    p.add_argument("--tau", type=float, default=1.)
    p.add_argument("--train", action="store_true", help="include backward cost")
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--output", type=Path,
                   default=Path("results/bench_woodbury_geometry.json"))
    a = p.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("No CUDA; pass --device cpu for basic profiling.")
    dtype = torch.float32
    torch.manual_seed(2026)
    q = F.normalize(torch.randn(a.batch, a.history, a.heads, a.d_key,
                                device=a.device, dtype=dtype), dim=-1)
    w = 1 + torch.rand(a.batch, a.history, a.heads, device=a.device, dtype=dtype)
    k = F.normalize(torch.randn(a.batch, a.chunk, a.heads, a.d_key,
                                device=a.device, dtype=dtype), dim=-1)
    q.requires_grad_(a.train)
    w.requires_grad_(a.train)
    k.requires_grad_(a.train)

    def synchronize():
        if a.device == "cuda":
            torch.cuda.synchronize()

    rows = []
    configurations = [("full_dense", None)]
    configurations += [(f"woodbury_r{r}", r) for r in a.ranks
                       if 0 < r <= a.history]
    for name, rank in configurations:
        def work():
            if rank is None:
                return dense_ema_direction(
                    q, w, k, gamma=a.gamma, mu=a.mu, lam=a.lam, tau=a.tau
                )
            return woodbury_direction(
                q, w, k, gamma=a.gamma, mu=a.mu, lam=a.lam,
                tau=a.tau, rank=rank
            )
        for _ in range(a.warmup):
            val = work()
            if a.train:
                val.square().mean().backward()
        synchronize()
        milliseconds = []
        for _ in range(a.repeats):
            if a.train:
                for t in (q, w, k):
                    t.grad = None
            start = time.perf_counter()
            val = work()
            if a.train:
                val.square().mean().backward()
            synchronize()
            milliseconds.append((time.perf_counter() - start) * 1000)
        row = {
            "method": name, "rank": rank,
            "median_ms": statistics.median(milliseconds),
            "min_ms": min(milliseconds),
        }
        if rank is not None:
            row["tail_covariance_operator_norm_bound"] = truncation_bound(
                a.gamma, rank, max_weight=2.
            )
        rows.append(row)
        print(row, flush=True)

    dense_ms = rows[0]["median_ms"]
    for row in rows:
        row["geometry_stage_speedup_vs_dense"] = dense_ms / row["median_ms"]
    result = {
        "config": {k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(a).items()},
        "hardware": torch.cuda.get_device_name() if a.device == "cuda" else "CPU",
        "scope": "geometry stage only; no claim of LM/GDN speedup",
        "results": rows,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    print("wrote", a.output)


if __name__ == "__main__":
    main()
