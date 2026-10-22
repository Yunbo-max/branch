#!/usr/bin/env python3
"""Measure step time vs solver_refresh for the cached-factorization variant.

Reports forward+backward seconds per step and the wall-clock ratio against the reference
(refresh=1). Also reports the solver-staleness gap so the speed number is never quoted
without its corresponding quality cost.

    python bench_solver.py --d-model 256 --d-key 64 --states 8 --seq-len 128
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from branch_cached import CachedBranchMixer, CachedSolverConfig  # noqa: E402
from branch_memory.mixer import BranchMemoryMixer, MemoryConfig  # noqa: E402


def build(refresh: int, a):
    kw = dict(d_model=a.d_model, n_heads=a.heads, d_key=a.d_key, d_value=a.d_value,
              d_route=a.d_route, max_states=a.states, mode="branch", covariance="full",
              protect=a.protect, fork_cost=a.fork_cost)
    if refresh == 1:
        return BranchMemoryMixer(MemoryConfig(**kw))
    return CachedBranchMixer(CachedSolverConfig(solver_refresh=refresh, **kw))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d-key", type=int, default=64)
    p.add_argument("--d-value", type=int, default=64)
    p.add_argument("--d-route", type=int, default=32)
    p.add_argument("--states", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--protect", type=float, default=8.0)
    p.add_argument("--fork-cost", type=float, default=0.02)
    p.add_argument("--refreshes", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--output", type=Path, default=Path("results/bench_solver.json"))
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    dtype = torch.float64 if a.dtype == "float64" else torch.float32
    torch.manual_seed(0)

    base_model = build(1, a).to(dtype).eval()
    x = torch.randn(a.batch_size, a.seq_len, a.d_model, dtype=dtype)
    with torch.no_grad():
        y_base, _, _ = base_model(x)

    records = []
    for refresh in a.refreshes:
        torch.manual_seed(0)
        model = build(refresh, a).to(dtype)
        if refresh != 1:
            model.load_state_dict(base_model.state_dict())
        times, gap = [], 0.0
        for r in range(a.repeats):
            model.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            y, cache, stats = model(x)
            loss = y.square().mean()
            loss.backward()
            times.append(time.perf_counter() - t0)
            assert torch.isfinite(loss)
            assert all(q.grad is None or torch.isfinite(q.grad).all() for q in model.parameters())
            if r == 0:
                gap = float((y.detach() - y_base).abs().max())
        rec = {"solver_refresh": refresh,
               "seconds_per_step": statistics.median(times),
               "seconds_all": times,
               "max_output_gap_vs_reference": gap}
        records.append(rec)
        print(json.dumps(rec), flush=True)
        del model
        gc.collect()

    ref = records[0]["seconds_per_step"]
    for r in records:
        r["speedup_vs_reference"] = ref / r["seconds_per_step"]
    report = {"config": vars(a) | {"output": str(a.output)}, "dtype": a.dtype,
              "threads": a.threads, "records": records,
              "note": "speed here is CPU wall clock on a serial reference; the gap column is "
                      "the quality cost of the stale factor and must be reported with it"}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, default=str))

    print("\n" + "=" * 72)
    print(f"{'refresh':>8}{'s/step':>12}{'speedup':>10}{'max |Δy| vs ref':>22}")
    print("-" * 72)
    for r in records:
        print(f"{r['solver_refresh']:>8}{r['seconds_per_step']:>12.4f}"
              f"{r['speedup_vs_reference']:>9.2f}x{r['max_output_gap_vs_reference']:>22.3e}")
    print("=" * 72)
    print(f"wrote {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
