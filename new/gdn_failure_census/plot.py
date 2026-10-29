#!/usr/bin/env python3
"""Optional descriptive diagnostic plots from one finished census summary.

No GPU, model loading or external data is needed. Uses default matplotlib
colors; no fabricated performance observations are part of this script.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def build_plot_metadata(payload: dict) -> dict:
    """Expose changing cell composition instead of treating a pooled slope as causal."""
    by_curve = defaultdict(list)
    for row in payload.get("groups", []):
        by_curve[row["condition"], row["length"]].append({
            "events": row["events"], "needle_region": row["needle_region"],
            "n": row["n"], "n_independent_seeds": row.get("n_independent_seeds"),
        })
    by_condition = defaultdict(list)
    coverage = []
    for (condition, length), cells in sorted(by_curve.items()):
        cells.sort(key=lambda x: (x["events"], x["needle_region"]))
        by_condition[condition].append(tuple((x["events"], x["needle_region"], x["n"]) for x in cells))
        coverage.append({"condition": condition, "length": length,
                         "n": sum(x["n"] for x in cells), "cells": cells})
    return {
        "schema": "gdn-census-plots.v2", "descriptive_only": True,
        "accuracy_pooling": "valid-row weighted average across event loads and needle regions",
        "length_curve_is_causal": False,
        "cell_composition_varies_by_length": {
            condition: len(set(signatures)) > 1 for condition, signatures in sorted(by_condition.items())
        },
        "coverage": coverage,
        "warnings": [
            "Feasibility exclusions and evaluation failures can change load, position, or seed coverage by length.",
            "A pooled accuracy drop is not an isolated length effect; compare matched strata and underlying cases.",
            "Paired collision contrasts compare conditions within length; their cross-length trend remains descriptive.",
            "The no_target curve measures label predictability without evidence and is not retrieval ability.",
        ],
    }


def draw(run: Path) -> None:

    payload = json.loads((run / "summary.json").read_text())
    group = payload.get("groups", [])
    if payload.get("diagnostic_only"):
        raise ValueError("Mock/diagnostic rows cannot produce model-performance plots")
    if not group:
        raise ValueError("No valid groups to plot; run pretrained evaluation first")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metadata = build_plot_metadata(payload)

    # Accuracies pooled over event count/needle position with per-row weighting.
    pooled = defaultdict(lambda: [0., 0])
    for row in group:
        key = (row["condition"], row["length"])
        pooled[key][0] += row["accuracy"] * row["n"]
        pooled[key][1] += row["n"]
    conditions = sorted({k[0] for k in pooled})
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for condition in conditions:
        pairs = sorted(
            (length, value[0] / value[1]) for (name, length), value in pooled.items()
            if name == condition and value[1] > 0
        )
        if not pairs:
            continue
        ax.plot([x for x, y in pairs], [y for x, y in pairs], marker="o",
                linestyle="--" if condition == "no_target" else "-",
                label="no_target (evidence removed)" if condition == "no_target" else condition)
    ax.axhline(.25, linestyle=":", linewidth=1, color="gray", label="random label: 25%")
    ax.set(xlabel="Prompt tokens", ylabel="Constrained 4-choice accuracy",
           title="GDN census: descriptive pooled accuracy")
    ax.set_xscale("log", base=2)
    ax.set_ylim(-.02, 1.02)
    ax.grid(alpha=.25)
    ax.legend(ncol=2)
    fig.text(.5, .015, "Cell coverage can change with length; pooled slopes are not isolated length effects.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(run / "retrieval_accuracy.png", dpi=180)
    plt.close(fig)

    rows = payload.get("paired", [])
    if rows:
        by_length = defaultdict(list)
        for row in rows:
            if row["unrelated_vs_collision_mean"] is not None:
                by_length[row["length"]].append(
                    (row["events"], row["unrelated_vs_collision_mean"])
                )
        fig, ax = plt.subplots(figsize=(9, 5))
        for n in sorted({e for es in by_length.values() for e, _ in es}):
            pairs = sorted(
                (length, value)
                for length, evs in by_length.items()
                for ev, value in evs if ev == n
            )
            if pairs:
                ax.plot([a for a, b in pairs], [b for a, b in pairs],
                        marker="o", label=f"{n} interfering records")
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set(xlabel="Prompt tokens",
               ylabel="Accuracy(collision) − Accuracy(unrelated)",
               title="Within-length paired contrasts (cross-length trend descriptive)")
        ax.grid(alpha=.25)
        ax.legend()
        fig.text(.5, .015, "Only complete valid pairs; retained positions and seeds may differ across lengths.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .05, 1, 1))
        fig.savefig(run / "paired_collision_effect.png", dpi=180)
        plt.close(fig)
    (run / "plot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved descriptive plots and coverage metadata in {run}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    draw(args.run)

