#!/usr/bin/env python3
"""Measure the chunked-vs-per-token cost of Branch Memory.

Both arms use the same ``ChunkedBranchMixer``; only ``chunk_len`` changes (``1`` = one
Python iteration per token, i.e. the per-token path). Reports forward+backward seconds
and the extrapolated cost at 16K context.

    python bench_chunked.py --seq-len 2048 --batch 2 --chunks 1 32 128
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

from chunked_branch import ChunkedBranchConfig, ChunkedBranchMixer  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--d-key", type=int, default=32)
    p.add_argument("--d-value", type=int, default=32)
    p.add_argument("--d-route", type=int, default=32)
    p.add_argument("--states", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--chunks", type=int, nargs="+", default=[1, 32, 128])
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--extrapolate-to", type=int, default=16384)
    p.add_argument("--output", type=Path, default=Path("results/bench_chunked.json"))
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    x = torch.randn(a.batch, a.seq_len, a.d_model)

    records = []
    for cl in a.chunks:
        cfg = ChunkedBranchConfig(d_model=a.d_model, n_heads=a.heads, d_key=a.d_key,
                                  d_value=a.d_value, d_route=a.d_route,
                                  max_states=a.states, chunk_len=cl)
        model = ChunkedBranchMixer(cfg)
        times = []
        for r in range(a.repeats):
            model.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            out, cache, stats = model(x)
            out.square().mean().backward()
            times.append(time.perf_counter() - t0)
            assert torch.isfinite(out).all()
            assert all(q.grad is None or torch.isfinite(q.grad).all() for q in model.parameters())
        med = statistics.median(times)
        rec = {
            "chunk_len": cl,
            "seconds": med,
            "seconds_per_token": med / a.seq_len,
            "extrapolated_16k_seconds": med * a.extrapolate_to / a.seq_len,
            "iterations_per_layer": (a.seq_len + cl - 1) // cl,
        }
        records.append(rec)
        print(json.dumps(rec), flush=True)
        del model, out, cache, stats
        gc.collect()

    base = records[0]["seconds"]
    for r in records:
        r["speedup_vs_per_token"] = base / r["seconds"]
    report = {"config": vars(a) | {"output": str(a.output)},
              "note": "same code path, only chunk_len changes; CPU wall clock, extrapolation "
                      "is linear in sequence length",
              "records": records}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, default=str))

    print("\n" + "=" * 82)
    print(f"{'chunk':>7}{'s/run':>11}{'speedup':>10}{'iters/layer':>13}{'extrap 16K':>13}")
    print("-" * 82)
    for r in records:
        print(f"{r['chunk_len']:>7}{r['seconds']:>11.3f}{r['speedup_vs_per_token']:>9.2f}x"
              f"{r['iterations_per_layer']:>13,}{r['extrapolated_16k_seconds']/60:>10.1f} min")
    print("=" * 82)
    print(f"wrote {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
