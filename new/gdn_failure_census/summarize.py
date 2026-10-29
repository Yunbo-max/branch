#!/usr/bin/env python3
"""Summarize paired GDN diagnostics with seed-clustered uncertainty."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _ratio_ci(numerators, denominators, clusters, *, seed=17, rounds=1000):
    """Row-weighted ratio; resample whole data-seed clusters.

    Unknown seed identity gives a descriptive estimate without an inferential
    CI. For conditional failure include zero-denominator rows to preserve the
    original seed population instead of treating selected successes as IID.
    """
    if not (len(numerators) == len(denominators) == len(clusters)):
        raise ValueError("metric and cluster lengths differ")
    if rounds < 100:
        raise ValueError("bootstrap_rounds must be >=100")
    den = sum(denominators)
    mean = sum(numerators) / den if den else None
    if mean is None or any(x is None for x in clusters):
        return mean, None, None
    grouped = defaultdict(lambda: [0.0, 0.0])
    for num, d, cluster in zip(numerators, denominators, clusters):
        grouped[cluster][0] += num
        grouped[cluster][1] += d
    values = list(grouped.values())
    n = len(values)
    if n < 2:
        return mean, None, None
    rng = random.Random(seed)
    boots = []
    for _ in range(rounds):
        chosen = [values[rng.randrange(n)] for _ in range(n)]
        sample_den = sum(v[1] for v in chosen)
        if sample_den:
            boots.append(sum(v[0] for v in chosen) / sample_den)
    if len(boots) < rounds // 2:
        return mean, None, None
    boots.sort()
    return mean, boots[int((len(boots) - 1) * .025)], boots[int((len(boots) - 1) * .975)]


def mean_ci(values: list[float], seed: int = 17, rounds: int = 1000):
    """Compatibility helper for genuinely independent observations only."""
    return _ratio_ci(values, [1] * len(values), list(range(len(values))), seed=seed, rounds=rounds)


def _cluster(row):
    value = row.get("seed")
    return None if value is None else json.dumps(value, sort_keys=True)


def _metric(rows, field, rounds):
    selected = [r for r in rows if r.get(field) is not None]
    values = [float(r[field]) for r in selected]
    if any(not math.isfinite(x) for x in values):
        raise ValueError(f"non-finite {field}")
    return _ratio_ci(values, [1] * len(values), [_cluster(r) for r in selected], rounds=rounds)


def _mean(rows, field):
    values = [float(r[field]) for r in rows if r.get(field) is not None]
    if any(not math.isfinite(x) for x in values):
        raise ValueError(f"non-finite {field}")
    return statistics.fmean(values) if values else None


def _n_seeds(rows):
    return len({_cluster(r) for r in rows if _cluster(r) is not None})


def _is_oom(row):
    text = f"{row.get('error_type', '')} {row.get('error', '')}".lower()
    return row.get("status") == "oom" or "outofmemory" in text or "out of memory" in text


def _is_unsupported(row):
    text = f"{row.get('error_type', '')} {row.get('error', '')}".lower()
    return row.get("status") == "unsupported" or "unsupported context" in text or "allow-extrapolation" in text


def summarize_records(
    records: list[dict], *, min_short_accuracy: float = .6,
    short_max_length: int = 1024, min_independent_seeds: int = 20,
    bootstrap_rounds: int = 1000, planned_cases: int | None = None,
    expected_probe_ids: list[str] | None = None,
) -> dict:
    if not 0 <= min_short_accuracy <= 1:
        raise ValueError("min_short_accuracy must be in [0,1]")
    if not 1 <= short_max_length <= 1024:
        raise ValueError("short_max_length must be in [1,1024]")
    if min_independent_seeds < 2 or bootstrap_rounds < 100:
        raise ValueError("at least 2 independent seeds and 100 bootstrap rounds required")
    ids = [r.get("probe_id") for r in records]
    if any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("every result must have a nonempty probe_id")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate probe_id; do not concatenate overlapping runs/retries")
    if expected_probe_ids is not None:
        if len(set(expected_probe_ids)) != len(expected_probe_ids):
            raise ValueError("duplicate probe_id in case plan")
        expected = set(expected_probe_ids)
        if set(ids) - expected:
            raise ValueError("result contains probe_id outside the case plan")
        if planned_cases is not None and planned_cases != len(expected):
            raise ValueError("planned_cases disagrees with case plan")
        planned_cases = len(expected)
    if planned_cases is not None and planned_cases < len(records):
        raise ValueError("more results than planned cases")

    evidence_types = {r.get("evidence_type", "pretrained_lm") for r in records}
    if len(evidence_types) > 1:
        raise ValueError("mixed evidence types: mock/diagnostic and model results must stay separate")
    diagnostic_only = bool(evidence_types and evidence_types != {"pretrained_lm"})
    statuses = Counter(r.get("status", "unknown") for r in records)
    all_ok = [r for r in records if r.get("status") == "ok"]
    ok = [] if diagnostic_only else all_ok
    failed = [r for r in records if r.get("status") != "ok"]
    for r in ok:
        if r.get("correct") not in (0, 1):
            raise ValueError("correct must be 0 or 1 for an ok result")
        if r.get("preedit_correct") not in (None, 0, 1):
            raise ValueError("preedit_correct must be null, 0 or 1")
        if r.get("actual_length", 0) <= 0 or r.get("num_events", -1) < 0:
            raise ValueError("invalid result length or event count")
        if not 0 <= float(r.get("needle_fraction", -1)) <= 1:
            raise ValueError("invalid needle_fraction")

    # Predeclared rule: shortest eligible plain control at <=1024 tokens,
    # explicitly marked as within config context. Never use an 8K minimum.
    eligible = [r for r in ok if r["condition"] == "plain"
                and r["actual_length"] <= short_max_length
                and r.get("is_context_extrapolated") is False]
    control_length = min((r["actual_length"] for r in eligible), default=None)
    short = [r for r in eligible if r["actual_length"] == control_length]
    acc, lo, hi = _metric(short, "correct", bootstrap_rounds)
    n_short_seeds = _n_seeds(short)
    missing_short_seed = sum(_cluster(r) is None for r in short)
    if acc is None or acc < min_short_accuracy:
        competence = "INSUFFICIENT_SHORT_CONTROL"
    elif (n_short_seeds < min_independent_seeds or missing_short_seed
          or lo is None or lo < min_short_accuracy):
        competence = "PROVISIONAL_SHORT_CONTROL"
    else:
        competence = "PASS"
    if diagnostic_only:
        competence = "DIAGNOSTIC_ONLY"

    no_target = [r for r in ok if r["condition"] == "no_target"]
    absent_acc, absent_lo, absent_hi = _metric(no_target, "correct", bootstrap_rounds)
    absent_seeds = _n_seeds(no_target)
    absent_missing_seed = sum(_cluster(r) is None for r in no_target)
    if diagnostic_only:
        absent_flag = "DIAGNOSTIC_ONLY"
    elif not no_target:
        absent_flag = "NOT_MEASURED"
    elif absent_seeds < min_independent_seeds or absent_missing_seed or absent_lo is None:
        absent_flag = "PROVISIONAL"
    else:
        absent_flag = "ABOVE_CHANCE" if absent_lo > .25 else "NO_CLEAR_ABOVE_CHANCE"
    no_target_control = {
        "n": len(no_target), "n_seeds": absent_seeds,
        "n_independent_seeds": absent_seeds, "n_missing_seed": absent_missing_seed,
        "accuracy": absent_acc, "ci_low": absent_lo, "ci_high": absent_hi,
        "chance_accuracy": .25, "flag": absent_flag, "answerable_control": False,
        "interpretation": (
            "Evidence-removed label-predictability check, not retrieval ability. Above-chance "
            "performance requires inspection for leakage or answer-label cues; an inconclusive "
            "interval does not prove their absence. Pooled over observed cells; inspect groups too."
        ),
    }

    grouped = defaultdict(list)
    for r in ok:
        depth = float(r["needle_fraction"])
        region = "early" if depth < .33 else ("middle" if depth < .67 else "late")
        grouped[(r["actual_length"], r["condition"], r["num_events"], region)].append(r)
    rows = []
    for (length, condition, events, region), part in sorted(grouped.items()):
        accuracy, lower, upper = _metric(part, "correct", bootstrap_rounds)
        nll = _mean(part, "gold_conditional_nll")
        margin = _mean(part, "margin")
        latency = _mean(part, "latency_ms")
        latencies = [r["latency_ms"] for r in part if r.get("latency_ms") is not None]
        rows.append({
            "length": length, "condition": condition, "events": events,
            "needle_region": region, "n": len(part),
            "n_independent_seeds": _n_seeds(part),
            "n_missing_seed": sum(_cluster(r) is None for r in part),
            "accuracy": accuracy, "accuracy_ci_low": lower, "accuracy_ci_high": upper,
            "conditional_nll": nll, "margin": margin,
            "gold_full_nll": _mean(part, "gold_full_nll"),
            "n_gold_full_nll": sum(r.get("gold_full_nll") is not None for r in part),
            "answerable_control": condition != "no_target",
            "interpretation": ("evidence-removed label predictability; not retrieval accuracy"
                               if condition == "no_target" else "answerable task accuracy"),
            "median_latency_ms": statistics.median(latencies) if latencies else None,
            "mean_latency_ms": latency,
        })

    paired_all = defaultdict(dict)
    for r in records:
        if not r.get("pair_id") or not r.get("condition"):
            raise ValueError("every result must have pair_id and condition")
        part = paired_all[r["pair_id"]]
        if r["condition"] in part:
            raise ValueError("duplicate pair_id/condition; possible mixed runs")
        if part:
            ref = next(iter(part.values()))
            for name in ("actual_length", "num_events", "seed"):
                if r.get(name) != ref.get(name):
                    raise ValueError(f"mismatched paired {name}")
        part[r["condition"]] = r
    paired_ok = {key: {c: r for c, r in part.items()
                      if r.get("status") == "ok" and not diagnostic_only}
                 for key, part in paired_all.items()}
    triplet_groups = defaultdict(list)
    triplet_seen = 0
    for part in paired_all.values():
        if diagnostic_only:
            continue
        if any(c in part for c in ("plain", "unrelated", "collision")):
            triplet_seen += 1
        if not all(c in part and part[c].get("status") == "ok"
                   for c in ("plain", "unrelated", "collision")):
            continue
        p, u, c = (part[name] for name in ("plain", "unrelated", "collision"))
        triplet_groups[p["actual_length"], p["num_events"]].append({
            "seed": p.get("seed"),
            "plain_vs_unrelated": float(u["correct"] - p["correct"]),
            "unrelated_vs_collision": float(c["correct"] - u["correct"]),
            "plain_vs_collision": float(c["correct"] - p["correct"]),
            "unrelated_correct": int(u["correct"]),
            "conditional_numerator": int(u["correct"] and not c["correct"]),
        })
    paired_rows = []
    for (length, events), part in sorted(triplet_groups.items()):
        row = {"length": length, "events": events, "n_triplets": len(part),
               "n_unrelated_correct": sum(r["unrelated_correct"] for r in part),
               "n_independent_seeds": _n_seeds(part),
               "n_missing_seed": sum(_cluster(r) is None for r in part)}
        for name in ("plain_vs_unrelated", "unrelated_vs_collision", "plain_vs_collision"):
            avg, lower, upper = _metric(part, name, bootstrap_rounds)
            row[f"{name}_mean"], row[f"{name}_lo"], row[f"{name}_hi"] = avg, lower, upper
        avg, lower, upper = _ratio_ci(
            [r["conditional_numerator"] for r in part],
            [r["unrelated_correct"] for r in part],
            [_cluster(r) for r in part], rounds=bootstrap_rounds,
        )
        row["conditional_collision_failure_mean"] = avg
        row["conditional_collision_failure_lo"] = lower
        row["conditional_collision_failure_hi"] = upper
        paired_rows.append(row)

    preedit_groups = defaultdict(list)
    for r in ok:
        if r.get("preedit_correct") is not None:
            preedit_groups[r["actual_length"], r["condition"], r["num_events"]].append(r)
    preedit_rows = []
    for (length, condition, events), part in sorted(preedit_groups.items()):
        n_pre = sum(r["preedit_correct"] for r in part)
        same_target = condition not in ("revision_latest", "no_target") and all(
            r.get("answer") == r.get("initial_answer")
            for r in part if r.get("answer") is not None and r.get("initial_answer") is not None
        )
        avg, lower, upper = (None, None, None)
        if same_target:
            avg, lower, upper = _ratio_ci(
                [int(r["preedit_correct"] and not r["correct"]) for r in part],
                [r["preedit_correct"] for r in part],
                [_cluster(r) for r in part], rounds=bootstrap_rounds,
            )
        pre_acc, pre_lo, pre_hi = _metric(part, "preedit_correct", bootstrap_rounds)
        post_acc, post_lo, post_hi = _metric(part, "correct", bootstrap_rounds)
        preedit_rows.append({
            "length": length, "condition": condition, "events": events,
            "n": len(part), "n_independent_seeds": _n_seeds(part),
            "n_correct_before_edit": n_pre, "preedit_accuracy": pre_acc,
            "preedit_accuracy_ci_low": pre_lo, "preedit_accuracy_ci_high": pre_hi,
            "postedit_accuracy": post_acc,
            "postedit_accuracy_ci_low": post_lo, "postedit_accuracy_ci_high": post_hi,
            "prepost_same_target": same_target,
            "lost_after_edit_given_preedit_correct": avg,
            "lost_after_edit_ci_low": lower, "lost_after_edit_ci_high": upper,
            "interpretation": ("same-target query-insertion contrast" if same_target else
                               "changed/absent target; not same-fact memory loss"),
        })

    revision_groups = defaultdict(list)
    for part in paired_ok.values():
        if "revision_latest" not in part or "revision_earliest" not in part:
            continue
        latest, earliest = part["revision_latest"], part["revision_earliest"]
        revision_groups[latest["actual_length"], latest["num_events"]].append({
            "seed": latest.get("seed"), "latest_correct": latest["correct"],
            "earliest_correct": earliest["correct"],
            "both_correct": int(latest["correct"] and earliest["correct"]),
            "target_label_changed": (
                latest["latest_answer"] != latest["initial_answer"]
                if latest.get("latest_answer") is not None and latest.get("initial_answer") is not None
                else None
            ),
        })
    revision_rows = []
    for (length, events), part in sorted(revision_groups.items()):
        row = {"length": length, "events": events, "n": len(part),
               "n_independent_seeds": _n_seeds(part)}
        for name, key in (("latest_accuracy", "latest_correct"),
                          ("earliest_accuracy", "earliest_correct"), ("both_correct", "both_correct")):
            avg, lower, upper = _metric(part, key, bootstrap_rounds)
            row[name], row[f"{name}_ci_low"], row[f"{name}_ci_high"] = avg, lower, upper
        for label, changed in (("changed", True), ("unchanged", False)):
            subset = [r for r in part if r["target_label_changed"] is changed]
            row[f"n_{label}_target_labels"] = len(subset)
            row[f"{label}_target_latest_accuracy"] = _mean(subset, "latest_correct")
            row[f"{label}_target_earliest_accuracy"] = _mean(subset, "earliest_correct")
        revision_rows.append(row)

    complete_triplets = sum(len(x) for x in triplet_groups.values())
    result = {
        "schema": "gdn-long-context-summary.v2",
        "evidence_type": next(iter(evidence_types), "pretrained_lm"),
        "diagnostic_only": diagnostic_only,
        "n_mock": sum(r.get("evidence_type") == "plumbing_mock" for r in records),
        "n_mock_ok": sum(r.get("evidence_type") == "plumbing_mock" for r in all_ok),
        "n_nonmodel": len(records) if diagnostic_only else 0,
        "n_ok": len(ok), "n_error": len(failed),
        "n_oom": sum(_is_oom(r) for r in failed),
        "n_unsupported": sum(_is_unsupported(r) for r in failed),
        "n_planned": planned_cases,
        "n_missing": planned_cases - len(records) if planned_cases is not None else None,
        "status_counts": dict(sorted(statuses.items())),
        "error_type_counts": dict(sorted(Counter(r.get("error_type", r.get("status", "unknown"))
                                                   for r in failed).items())),
        "chance_accuracy": .25,
        "uncertainty": {
            "unit": "data-generation seed (whole cluster)",
            "method": "paired seed-cluster percentile bootstrap, row-weighted estimand",
            "rounds": bootstrap_rounds, "confidence": .95,
            "minimum_seeds_for_short_pass": min_independent_seeds,
            "scope": "conditional on this fixed checkpoint; not model-training uncertainty",
        },
        "short_control": {
            "min_tested_length": min((r["actual_length"] for r in ok), default=None),
            "max_allowed_length": short_max_length, "evaluated_control_length": control_length,
            "plain_n": len(short), "n_independent_seeds": n_short_seeds,
            "n_missing_seed": missing_short_seed, "accuracy": acc,
            "ci_low": lo, "ci_high": hi, "threshold": min_short_accuracy,
            "competence_gate": competence,
            "interpretation": (
                "Short basic lookup passes the prespecified seed-count and CI gates; "
                "this alone does not establish GDN causality."
                if competence == "PASS" else
                "Do not attribute long-context errors to memory erasure: an in-context "
                "<=1024-token plain control has not passed the competence evidence gate."
            ),
        },
        "no_target_control": no_target_control,
        "groups": rows, "paired": paired_rows,
        "revisions": revision_rows, "preedit": preedit_rows,
        "pair_coverage": {"triplets_observed_any_condition": triplet_seen,
                          "complete_valid_triplets": complete_triplets,
                          "incomplete_or_failed_triplets": triplet_seen - complete_triplets},
        "context_extrapolation_n": sum(bool(r.get("is_context_extrapolated")) for r in ok),
        "warnings": [
            "Synthetic four-choice diagnostic; random-label baseline is 25%, not an official benchmark score.",
            "CIs resample data seeds, keeping repeated positions/load/conditions together; missing seed metadata disables CIs.",
            "Small seed counts and boundary bootstrap intervals warrant caution; a smoke test is provisional.",
            "Differently pretrained model families are descriptive comparisons, not matched architectural causality.",
            "Config-context extrapolation is separate from verified training-context exposure.",
            "Errors/OOM/unsupported and missing planned cases are separate from accuracy; inspect attrition before comparing.",
            "Paired contrasts require all three valid conditions; incomplete triplets are reported, not imputed.",
            "Pre-edit probing changes the queried prefix; it is not a direct hidden-state trace.",
            "Revision-latest changes the target after updates and is never labeled loss of the same fact.",
            "No-target accuracy measures answer-label predictability without evidence, not successful retrieval.",
            "Intervals are exploratory across many cells; confirm selected contrasts on held-out seeds.",
        ],
    }
    if not ok:
        result["reason"] = ("Diagnostic/mock plumbing rows excluded from model accuracy and inference"
                            if diagnostic_only else "No valid model evaluations")
    return result


def _save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        if path.exists():
            path.unlink()
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--min-short-accuracy", type=float, default=.60)
    parser.add_argument("--short-max-length", type=int, default=1024)
    parser.add_argument("--min-independent-seeds", type=int, default=20)
    parser.add_argument("--bootstrap-rounds", type=int, default=1000)
    args = parser.parse_args()
    result_path = args.run / "results.jsonl"
    if not result_path.exists():
        raise SystemExit(f"missing {result_path} (run run.py first)")
    with result_path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    manifest_path, plan_path = args.run / "manifest.json", args.run / "cases.jsonl"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    expected = None
    if plan_path.exists():
        with plan_path.open(encoding="utf-8") as fh:
            expected = [json.loads(line)["probe_id"] for line in fh if line.strip()]
    result = summarize_records(
        rows, min_short_accuracy=args.min_short_accuracy,
        short_max_length=args.short_max_length, min_independent_seeds=args.min_independent_seeds,
        bootstrap_rounds=args.bootstrap_rounds, planned_cases=manifest.get("planned_cases"),
        expected_probe_ids=expected,
    )
    result["skipped_invalid_grid_cases"] = manifest.get("skipped_invalid_grid_cases")
    (args.run / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name in ("groups", "paired", "revisions", "preedit"):
        _save_csv(args.run / f"{name}.csv", result[name])
    print(f"OK {result['n_ok']} / errors {result['n_error']} / missing {result['n_missing']}")
    ctrl = result["short_control"]
    print("Short-control gate:", ctrl["competence_gate"], "accuracy:", ctrl["accuracy"],
          "independent seeds:", ctrl["n_independent_seeds"])
    for row in result["paired"]:
        print(f"{row['length']}t {row['events']} events "
              f"near-vs-unrelated={row['unrelated_vs_collision_mean']:+.4f}; "
              f"conditional failure={row['conditional_collision_failure_mean']}")
    print(f"Saved {args.run / 'summary.json'} plus CSV tables")


if __name__ == "__main__":
    main()
