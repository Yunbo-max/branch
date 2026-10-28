#!/usr/bin/env python3
"""Aggregate long-context diagnosis with paired interference and competence gates."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def mean_ci(values: list[float], seed: int = 17, rounds: int = 1000):
    if not values:
        return None, None, None
    mu = statistics.fmean(values)
    if len(values) == 1:
        return mu, None, None
    rng = random.Random(seed)
    n = len(values)
    boots = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(rounds)
    )
    return mu, boots[round(rounds * 0.025)], boots[min(rounds - 1, round(rounds * 0.975))]


def summarize_records(records: list[dict], *, min_short_accuracy: float = 0.6) -> dict:
    ok = [r for r in records if r.get("status") == "ok"]
    error = [r for r in records if r.get("status") == "error"]
    if not ok:
        return {"n_ok": 0, "n_error": len(error), "reason": "No valid model evaluations"}

    length_min = min(r["actual_length"] for r in ok)
    short_base = [
        r["correct"] for r in ok
        if r["condition"] == "plain" and r["actual_length"] == length_min
    ]
    short_acc, short_lo, short_hi = mean_ci(short_base)
    competence = (
        "PASS" if short_acc is not None and short_acc >= min_short_accuracy
        else "INSUFFICIENT_SHORT_CONTROL"
    )

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in ok:
        depth = float(r["needle_fraction"])
        depth_bin = "early" if depth < .33 else ("middle" if depth < .67 else "late")
        grouped[(r["actual_length"], r["condition"], r["num_events"], depth_bin)].append(r)
    rows = []
    for (length, condition, n_events, depth_bin), part in sorted(grouped.items()):
        acc, lo, hi = mean_ci([float(r["correct"]) for r in part])
        nll, _, _ = mean_ci([float(r["gold_conditional_nll"]) for r in part])
        margin, _, _ = mean_ci([float(r["margin"]) for r in part])
        latency, _, _ = mean_ci([float(r["latency_ms"]) for r in part])
        rows.append({
            "length": length, "condition": condition,
            "events": n_events, "needle_region": depth_bin,
            "n": len(part),
            "accuracy": acc, "accuracy_ci_low": lo, "accuracy_ci_high": hi,
            "conditional_nll": nll, "margin": margin,
            "median_latency_ms": statistics.median(r["latency_ms"] for r in part),
            "mean_latency_ms": latency,
        })

    # Pair records by target key/value/length/position/event count. As these
    # differ only by the type of intervening records, their comparisons are
    # more causally informative than unpaired length-vs-accuracy differences.
    paired: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in ok:
        paired[r["pair_id"]][r["condition"]] = r
    pair_results: dict[tuple[int, int], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for recs in paired.values():
        if not all(k in recs for k in ("plain", "unrelated", "collision")):
            continue
        plain, unrelated, collision = (recs[k] for k in ("plain", "unrelated", "collision"))
        length, events = plain["actual_length"], plain["num_events"]
        v = pair_results[(length, events)]
        v["plain_vs_unrelated"].append(float(unrelated["correct"] - plain["correct"]))
        v["unrelated_vs_collision"].append(float(collision["correct"] - unrelated["correct"]))
        v["plain_vs_collision"].append(float(collision["correct"] - plain["correct"]))
        v["unrelated_correct_then_collision_wrong"].append(float(
            unrelated["correct"] and not collision["correct"]
        ))
        v["unrelated_correct"].append(float(unrelated["correct"]))
        if unrelated["correct"]:
            v["conditional_collision_failure"].append(float(not collision["correct"]))

    paired_rows = []
    for (length, events), values in sorted(pair_results.items()):
        rec = {"length": length, "events": events,
               "n_triplets": len(values["plain_vs_unrelated"]),
               "n_unrelated_correct": int(sum(values["unrelated_correct"]))}
        for name in ("plain_vs_unrelated", "unrelated_vs_collision",
                     "plain_vs_collision", "conditional_collision_failure"):
            avg, lo, hi = mean_ci(values[name])
            rec[f"{name}_mean"] = avg
            rec[f"{name}_lo"] = lo
            rec[f"{name}_hi"] = hi
        paired_rows.append(rec)

    # Compare latest and earliest retrieval on *the same revision history*.
    revisions = []
    for key, recs in paired.items():
        if "revision_latest" in recs and "revision_earliest" in recs:
            newest, oldest = recs["revision_latest"], recs["revision_earliest"]
            revisions.append({
                "length": newest["actual_length"],
                "events": newest["num_events"],
                "latest_correct": newest["correct"],
                "earliest_correct": oldest["correct"],
                "both_correct": int(newest["correct"] and oldest["correct"]),
            })

    revision_rows = []
    by_rev: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in revisions:
        by_rev[r["length"], r["events"]].append(r)
    for (length, events), group in sorted(by_rev.items()):
        revision_rows.append({
            "length": length, "events": events, "n": len(group),
            "latest_accuracy": statistics.fmean(r["latest_correct"] for r in group),
            "earliest_accuracy": statistics.fmean(r["earliest_correct"] for r in group),
            "both_correct": statistics.fmean(r["both_correct"] for r in group),
        })

    return {
        "n_ok": len(ok), "n_error": len(error),
        "short_control": {
            "min_tested_length": length_min,
            "plain_n": len(short_base),
            "accuracy": short_acc, "ci_low": short_lo, "ci_high": short_hi,
            "competence_gate": competence,
            "interpretation": (
                "Do NOT attribute long-context failure to memory erasure "
                "when short plain lookup is not mastered."
                if competence != "PASS" else
                "Short basic lookup passes threshold. Paired interference "
                "tests can now be interpreted more strongly."
            ),
        },
        "groups": rows,
        "paired": paired_rows,
        "revisions": revision_rows,
        "context_extrapolation_n": sum(bool(r["is_context_extrapolated"]) for r in ok),
        "warnings": [
            "Synthetic diagnostic, not official RULER/NoLiMa benchmark.",
            "Comparing differently pretrained model families is descriptive, not a matched causal claim.",
            "Results beyond each checkpoint's training/config context are extrapolation.",
            "Zero-shot base models may fail instruction following; require short-context competence gate.",
            "Paired edits change text tokens; exact lengths and target positions are controlled, "
            "but content distribution is not identical.",
            "Errors/OOM are excluded from accuracy denominators and reported separately.",
        ],
    }


def _save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--min-short-accuracy", type=float, default=.60)
    args = ap.parse_args()
    result_path = args.run / "results.jsonl"
    if not result_path.exists():
        raise SystemExit(f"missing {result_path} (run run.py first)")
    rows = []
    with result_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    result = summarize_records(rows, min_short_accuracy=args.min_short_accuracy)
    (args.run / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for name in ("groups", "paired", "revisions"):
        _save_csv(args.run / f"{name}.csv", result.get(name, []))
    print(f"OK {result['n_ok']} / errors {result['n_error']}")
    ctrl = result.get("short_control", {})
    print("Short-control gate:", ctrl.get("competence_gate"),
          "accuracy:", ctrl.get("accuracy"))
    for row in result.get("paired", []):
        print(
            f"{row['length']}t {row['events']} events "
            f"near-vs-unrelated={row['unrelated_vs_collision_mean']:+.4f}; "
            f"conditional failure={row['conditional_collision_failure_mean']}"
        )
    print(f"Saved {args.run / 'summary.json'} plus CSV tables")


if __name__ == "__main__":
    main()
