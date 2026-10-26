#!/usr/bin/env python3
"""Multi-seed MQAR sweep across arms, for a GPU slot.

Runs every (arm, seed) pair with identical data/budget, writes one report per run and a
summary with mean ± std. Results are appended incrementally so a killed job keeps what
already finished.

    python sweep.py --arms march branch delta transformer --seeds 17 23 29 \
      --steps 1500 --batch-size 32 --seq-len 128 --num-kv-pairs 8 --vocab-size 8192 \
      --d-model 256 --layers 2 --heads 4 --d-key 64 --d-value 64 --d-route 32 \
      --anchor-interval 16 --max-anchors 8 --max-states 8 --differentiate-covariance \
      --lr 1e-3 --threads 8 --device cuda --output-root results/fair_gpu_screen
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import train_fair as tf  # noqa: E402
from fair_models import FairLM, MIXERS, count_mixer_parameters  # noqa: E402


def counts_for(a, mixer: str, ff_extra: int = 0) -> dict:
    setattr(a, "mixer", mixer)
    setattr(a, "ff_extra", ff_extra)
    with torch.device("meta"):
        model = FairLM(tf.default_config(a))
    return count_mixer_parameters(model)


def match_parameters(a, arms) -> dict:
    base = {m: counts_for(a, m, 0) for m in arms}
    target = max(c["total"] for c in base.values())
    per_unit = 2 * a.d_model * a.layers
    extra = {m: max(0, round((target - base[m]["total"]) / per_unit)) for m in arms}
    after = {m: counts_for(a, m, extra[m]) for m in arms}
    return {"target_total": target, "ff_extra": extra, "after": after,
            "spread": max(c["total"] for c in after.values())
                      - min(c["total"] for c in after.values())}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    probe = argparse.ArgumentParser(add_help=False)
    tf.add_common_args(probe)
    for action in probe._actions:
        if action.dest in {"mixer", "output", "help"}:
            continue
        p._add_action(action)
    p.add_argument("--arms", nargs="+", choices=list(MIXERS), default=list(MIXERS))
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 29])
    p.add_argument("--output-root", type=Path, default=Path("results/sweep"))
    p.add_argument("--time-budget-seconds", type=float, default=0.0,
                   help="stop starting new runs after this many seconds (0 = no limit)")
    a = p.parse_args()

    a.output_root.mkdir(parents=True, exist_ok=True)
    matching = match_parameters(a, a.arms)
    print("param matching:", json.dumps({"ff_extra": matching["ff_extra"],
                                         "spread": matching["spread"]}), flush=True)

    summary_path = a.output_root / "sweep_summary.json"
    started = time.perf_counter()
    rows = []
    for seed in a.seeds:
        for arm in a.arms:
            if a.time_budget_seconds and time.perf_counter() - started > a.time_budget_seconds:
                print("time budget reached; stopping", flush=True)
                (summary_path).write_text(json.dumps({"partial": True, "matching": matching,
                                                      "rows": rows}, indent=2))
                return 0
            setattr(a, "mixer", arm)
            setattr(a, "seed", seed)
            setattr(a, "ff_extra", matching["ff_extra"][arm])
            setattr(a, "output", a.output_root / f"{arm}_seed{seed}")
            print(f"\n===== {arm} seed={seed} =====", flush=True)
            try:
                rep = tf.run(a)
            except Exception as exc:  # keep the sweep alive
                print(f"[fail] {arm} seed={seed}: {type(exc).__name__}: {exc}", flush=True)
                rows.append({"arm": arm, "seed": seed, "status": "failed",
                             "error": f"{type(exc).__name__}: {exc}"})
                summary_path.write_text(json.dumps({"partial": True, "matching": matching,
                                                    "rows": rows}, indent=2))
                continue
            rows.append({
                "arm": arm, "seed": seed, "status": "ok",
                "params_total": rep["parameters"]["total"],
                "params_mixer": rep["parameters"]["mixer"],
                "valid_loss": rep["final"]["valid"]["loss"],
                "valid_accuracy": rep["final"]["valid"]["accuracy"],
                "test_loss": rep["final"]["test"]["loss"],
                "test_accuracy": rep["final"]["test"]["accuracy"],
                "seconds_per_step": rep["training"]["seconds_per_step"],
                "panel_sha256": rep["panels"]["test"]["sha256"],
            })
            summary_path.write_text(json.dumps({"partial": False, "matching": matching,
                                                "rows": rows}, indent=2))

    agg = {}
    for arm in a.arms:
        ok = [r for r in rows if r["arm"] == arm and r["status"] == "ok"]
        if not ok:
            continue
        agg[arm] = {
            "n_seeds": len(ok),
            "test_accuracy_mean": statistics.fmean(r["test_accuracy"] for r in ok),
            "test_accuracy_std": statistics.pstdev(r["test_accuracy"] for r in ok) if len(ok) > 1 else 0.0,
            "test_loss_mean": statistics.fmean(r["test_loss"] for r in ok),
            "valid_accuracy_mean": statistics.fmean(r["valid_accuracy"] for r in ok),
            "seconds_per_step_mean": statistics.fmean(r["seconds_per_step"] for r in ok),
            "params_total": ok[0]["params_total"],
        }
    summary_path.write_text(json.dumps(
        {"partial": False, "matching": matching, "aggregate": agg, "rows": rows}, indent=2))

    print("\n" + "=" * 76)
    print(f"{'arm':16}{'seeds':>7}{'test acc (mean±std)':>26}{'test loss':>12}{'s/step':>10}")
    print("-" * 76)
    for arm, s in agg.items():
        print(f"{arm:16}{s['n_seeds']:>7}"
              f"{s['test_accuracy_mean']*100:>17.2f}% ± {s['test_accuracy_std']*100:<6.2f}"
              f"{s['test_loss_mean']:>12.4f}{s['seconds_per_step_mean']:>10.3f}")
    print("=" * 76)
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
