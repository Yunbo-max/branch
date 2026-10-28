#!/usr/bin/env python3
"""Optional paper-ready diagnostic plots from one finished census summary.

No GPU, model loading or external data is needed. Uses default matplotlib
colors; no fabricated performance observations are part of this script.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def draw(run: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads((run / "summary.json").read_text())
    group = payload.get("groups", [])
    if not group:
        raise ValueError("No valid groups to plot; run pretrained evaluation first")

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
        ax.plot([x for x, y in pairs], [y for x, y in pairs],
                marker="o", label=condition)
    ax.set(xlabel="Prompt tokens", ylabel="Constrained 4-choice accuracy",
           title="GDN long-context failure census")
    ax.set_xscale("log", base=2)
    ax.set_ylim(-.02, 1.02)
    ax.grid(alpha=.25)
    ax.legend(ncol=2)
    fig.tight_layout()
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
               title="Paired effect of similar-key interference")
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run / "paired_collision_effect.png", dpi=180)
        plt.close(fig)
    print(f"Saved plots in {run}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    draw(args.run)
