#!/usr/bin/env python3
"""Paired MQAR comparison across all arms on identical data and budget.

Arms: ``march``, ``branch`` (Branch Memory), ``delta`` (Gated DeltaNet), ``transformer``.

All arms share: backbone (d_model/layers/heads), data panels (same SHA-256), optimiser,
learning rate, batch size, step count and seed. Only the mixer differs. Total parameter
counts are equalised automatically by giving the smaller arms extra FFN hidden units;
the residual gap is printed, not hidden.

    python compare.py --steps 500 --batch-size 16 --output-root results/fair
    python compare.py --arms delta transformer --steps 200 --output-root results/base
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
    per_unit = 2 * a.d_model * a.layers       # one extra FFN hidden unit, per layer
    extra = {m: max(0, round((target - base[m]["total"]) / per_unit)) for m in arms}
    achieved = {m: counts_for(a, m, extra[m]) for m in arms}
    return {"arms": list(arms), "before": base, "target_total": target,
            "ff_extra": extra, "after": achieved, "params_per_ff_unit": per_unit,
            "residual_gap": {m: achieved[m]["total"] - target for m in arms}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    probe = argparse.ArgumentParser(add_help=False)
    tf.add_common_args(probe)
    for action in probe._actions:
        if action.dest in {"mixer", "output", "help", "ff_extra"}:
            continue
        p._add_action(action)
    p.add_argument("--arms", nargs="+", choices=list(MIXERS), default=list(MIXERS))
    p.add_argument("--output-root", type=Path, default=Path("results/fair"))
    p.add_argument("--no-match-params", action="store_true")
    a = p.parse_args()

    if a.no_match_params:
        matching = {"arms": list(a.arms), "ff_extra": {m: 0 for m in a.arms},
                    "residual_gap": {}, "target_total": None}
    else:
        matching = match_parameters(a, a.arms)
    print("param matching ff_extra:", json.dumps(matching["ff_extra"]), flush=True)
    if matching.get("residual_gap"):
        print("residual gaps:", json.dumps(matching["residual_gap"]), flush=True)

    reports = {}
    for mixer in a.arms:
        setattr(a, "mixer", mixer)
        setattr(a, "output", a.output_root / mixer)
        setattr(a, "ff_extra", matching["ff_extra"][mixer])
        print(f"\n===== {mixer} (ff_extra={a.ff_extra}) =====", flush=True)
        reports[mixer] = tf.run(a)

    a.output_root.mkdir(parents=True, exist_ok=True)
    (a.output_root / "paired_report.json").write_text(
        json.dumps({"matching": matching, "arms": reports}, indent=2), encoding="utf-8")

    width = max(20, *(len(m) for m in a.arms))
    print("\n" + "=" * (26 + (width + 4) * len(a.arms)))
    print(f"{'metric':24}" + "".join(f"{m:>{width + 4}}" for m in a.arms))
    print("-" * (26 + (width + 4) * len(a.arms)))
    rows = [
        ("mixer params", lambda r: f"{r['parameters']['mixer']:,}"),
        ("total params", lambda r: f"{r['parameters']['total']:,}"),
        ("valid loss", lambda r: f"{r['final']['valid']['loss']:.4f}"),
        ("valid accuracy", lambda r: f"{r['final']['valid']['accuracy']:.4f}"),
        ("test loss", lambda r: f"{r['final']['test']['loss']:.4f}"),
        ("test accuracy", lambda r: f"{r['final']['test']['accuracy']:.4f}"),
        ("seconds/step", lambda r: f"{r['training']['seconds_per_step']:.3f}"),
    ]
    for label, fn in rows:
        print(f"{label:24}" + "".join(f"{fn(reports[m]):>{width + 4}}" for m in a.arms))
    print("=" * (26 + (width + 4) * len(a.arms)))

    first = reports[a.arms[0]]
    same_data = all(all(reports[m]["panels"][k]["sha256"] == first["panels"][k]["sha256"]
                        for k in first["panels"]) for m in a.arms)
    totals = [reports[m]["parameters"]["total"] for m in a.arms]
    print(f"identical data panels: {same_data}")
    print(f"parameter spread: {max(totals) - min(totals):,} params "
          f"({(max(totals) - min(totals)) / max(totals) * 100:.3f}%)")
    print(f"identical budget: steps={a.steps} batch={a.batch_size} lr={a.lr} seed={a.seed}")
    print(f"wrote {a.output_root / 'paired_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
