#!/usr/bin/env python3
"""Branch Memory on native MQAR — single arm, single seed.

Measures throughput with a short warmup, then sizes the real run from the wall-clock
budget so the job cannot be killed mid-schedule. Logs incrementally.

    python run_branch_only.py --budget-seconds 16200 --target-steps 40000 \
      --output-root results/branch_mqar
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import train_fair as tf  # noqa: E402


def build_args(a, steps: int, out: Path):
    ns = argparse.Namespace(**vars(a))
    ns.mixer = a.mixer
    ns.steps = steps
    ns.output = out
    ns.ff_extra = 0
    return ns


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    probe = argparse.ArgumentParser(add_help=False)
    tf.add_common_args(probe)
    for action in probe._actions:
        if action.dest in {"mixer", "output", "help"}:
            continue
        p._add_action(action)
    p.add_argument("--mixer", choices=["branch", "branch_cached"], default="branch_cached")
    p.add_argument("--budget-seconds", type=float, default=16200.0,
                   help="wall-clock budget for BOTH phases")
    p.add_argument("--warmup-steps", type=int, default=25)
    p.add_argument("--target-steps", type=int, default=40000)
    p.add_argument("--output-root", type=Path, default=Path("results/branch_mqar"))
    a = p.parse_args()

    if a.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    a.output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    # --- phase 1: warmup, also validates the GPU environment ---
    print(f"\n=== warmup ({a.warmup_steps} steps) ===", flush=True)
    warm = tf.run(build_args(a, a.warmup_steps, a.output_root / "warmup"))
    s_per_step = warm["training"]["seconds_per_step"] or 1.0
    elapsed = time.perf_counter() - started
    remaining = a.budget_seconds - elapsed
    planned = int(min(a.target_steps, max(0, remaining) / s_per_step))
    print(json.dumps({"warmup_seconds_per_step": s_per_step,
                      "elapsed_seconds": elapsed,
                      "remaining_budget_seconds": remaining,
                      "planned_steps": planned}), flush=True)

    if planned < 50:
        print("budget too small for a meaningful run; stopping after warmup", flush=True)
        (a.output_root / "run_summary.json").write_text(json.dumps(
            {"status": "budget_exhausted_after_warmup", "warmup": warm,
             "seconds_per_step": s_per_step}, indent=2))
        return 0

    # --- phase 2: the real run ---
    print(f"\n=== main run ({planned} steps) ===", flush=True)
    main_started = time.perf_counter()
    rep = tf.run(build_args(a, planned, a.output_root / "main"))
    wall = time.perf_counter() - main_started

    summary = {
        "arm": "branch",
        "scope": "native MQAR, single arm / single seed; not a paper reproduction",
        "protocol": {"vocab_size": a.vocab_size, "seq_len": a.seq_len,
                     "num_kv_pairs": a.num_kv_pairs, "num_examples": a.num_examples,
                     "seed": a.seed},
        "model": rep["config"],
        "parameters": rep["parameters"],
        "panels": rep["panels"],
        "warmup": {"steps": a.warmup_steps, "seconds_per_step": s_per_step},
        "main": {"planned_steps": planned, "steps": rep["training"]["steps"],
                 "wall_seconds": wall,
                 "seconds_per_step": rep["training"]["seconds_per_step"]},
        "initial": rep["initial"],
        "final": rep["final"],
        "history_tail": rep["history_tail"],
        "chance_accuracy": 1.0 / (a.vocab_size / 2),
        "beats_chance": rep["final"]["test"]["accuracy"] > 3.0 / (a.vocab_size / 2),
        "environment": rep["environment"],
    }
    (a.output_root / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps({"final": rep["final"],
                             "chance": summary["chance_accuracy"],
                             "beats_chance": summary["beats_chance"],
                             "seconds_per_step": rep["training"]["seconds_per_step"]},
                            indent=2), flush=True)
    print(f"wrote {a.output_root / 'run_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
