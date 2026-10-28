#!/usr/bin/env python3
"""Measure failure modes of PRETRAINED GDN models, zero-shot and tokenizer-exact."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from backend import PretrainedCausalLM, single_token_choices
from spec import CONDITIONS, make_probe, suite_size


def csv_ints(text: str) -> list[int]:
    out = [int(x) for x in text.split(",") if x.strip()]
    if not out or any(x < 0 for x in out):
        raise argparse.ArgumentTypeError("expected a comma-separated list of nonnegative integers")
    return out


def csv_floats(text: str) -> list[float]:
    out = [float(x) for x in text.split(",") if x.strip()]
    if not out or any(not (0.02 <= x <= 0.95) for x in out):
        raise argparse.ArgumentTypeError("needle positions must be in [0.02,0.95]")
    return out


def csv_conditions(text: str) -> list[str]:
    out = [x.strip() for x in text.split(",") if x.strip()]
    if not out or any(x not in CONDITIONS for x in out):
        raise argparse.ArgumentTypeError("conditions must be from " + ", ".join(CONDITIONS))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="m-a-p/340M-20B-GatedDeltaNet-pure-baseline")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--lengths", type=csv_ints, default=[512, 1024, 2048, 4096])
    p.add_argument("--seed-start", type=int, default=1000)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--positions", type=csv_floats, default=[0.12, 0.5, 0.88])
    p.add_argument("--events", type=csv_ints, default=[8, 32])
    p.add_argument("--conditions", type=csv_conditions,
                   default=["plain", "unrelated", "collision", "revision_latest",
                            "revision_earliest", "alias", "two_hop"])
    p.add_argument("--event-placement", choices=["early", "late"], default="early")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--allow-extrapolation", action="store_true",
                   help="explicitly evaluate lengths beyond model config position limit")
    p.add_argument("--preedit-probes", action="store_true",
                   help="also test the SAME fact immediately before interfering records")
    p.add_argument("--dry-run", action="store_true",
                   help="prepare an auditable case manifest without loading model weights")
    p.add_argument("--resume", action="store_true",
                   help="resume an interrupted run with exactly the same parameters")
    p.add_argument("--max-samples", type=int, default=0,
                   help="debugging limit; 0=all cases")
    p.add_argument("--error-policy", choices=["stop", "record"], default="stop")
    p.add_argument("--log-every", type=int, default=25)
    a = p.parse_args(argv)
    if a.seeds < 1 or a.seed_start < 0 or a.max_samples < 0 or a.log_every < 1:
        p.error("invalid seed/max-samples/log-every")
    if any(l < 128 for l in a.lengths):
        p.error("lengths must be >=128")
    return a


def _config(a) -> dict:
    fields = (
        "model", "tokenizer", "revision", "dtype", "device", "lengths",
        "seed_start", "seeds", "positions", "events", "conditions",
        "event_placement", "allow_extrapolation", "max_samples",
        "preedit_probes",
    )
    return {k: getattr(a, k) for k in fields}


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _cases(a, tok):
    # Only emit COMPLETE matched groups. If short context cannot fit 32/64
    # distractors, discard every condition for that cell (including plain);
    # otherwise comparisons would silently use unequal case populations.
    for length in a.lengths:
        for seed in range(a.seed_start, a.seed_start + a.seeds):
            for depth in a.positions:
                for n in a.events:
                    group = []
                    invalid = False
                    for condition in a.conditions:
                        try:
                            case = make_probe(
                                tok, length=length, seed=seed,
                                needle_fraction=depth, num_events=n,
                                condition=condition,
                                event_placement=a.event_placement,
                            )
                        except ValueError as exc:
                            if "required text" not in str(exc):
                                raise
                            invalid = True
                            break
                        # The demanded needle location must actually be realized.
                        # Short texts with many events cannot be forced to .88
                        # without clipping, which would confound the position study.
                        if abs(case.needle_fraction - depth) > 0.10:
                            invalid = True
                            break
                        group.append(case)
                    if invalid:
                        continue
                    if len({x.needle_token for x in group}) != 1:
                        # Same target at different positions is not a paired
                        # comparison: reject it rather than silently average.
                        continue
                    yield from group


def main(argv=None) -> int:
    a = parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    config = _config(a)
    manifest_path = a.out / "manifest.json"
    plan_path = a.out / "cases.jsonl"
    results_path = a.out / "results.jsonl"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if not a.resume:
            raise SystemExit(f"{manifest_path} exists; use --resume or a new --out")
        if saved["config"] != config:
            raise SystemExit("resume refused: existing manifest uses a different configuration")
        if a.dry_run != bool(saved.get("dry_run")):
            raise SystemExit("resume refused: cannot mix dry-run with GPU evaluation")
    elif a.resume:
        raise SystemExit("--resume requires an existing manifest.json")

    # Importing the tokenizer does not allocate model weights. Case generation
    # is pinned to the MODEL tokenizer, so requested lengths are exact tokens.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        a.tokenizer or a.model, revision=a.revision, trust_remote_code=False
    )

    probes = list(_cases(a, tok))
    if a.max_samples:
        probes = probes[:a.max_samples]
    # Explicitly verify the token budget and the case ID uniqueness.
    if len({p.probe_id for p in probes}) != len(probes):
        raise RuntimeError("duplicated probe IDs; use unique seeds/positions/events")
    if any(p.actual_length != p.target_length for p in probes):
        raise RuntimeError("a probe violated the token budget")
    if not probes:
        raise RuntimeError("empty census")

    # Validate actual checkpoint-tokenizer response labels *before* loading
    # potentially several GB of pretrained model weights. Each query type
    # shares the same Answer: boundary; 40 distinct probes cover lexical IDs.
    for case in probes[:40]:
        single_token_choices(tok, case.question_suffix)

    if not manifest_path.exists():
        manifest = {
            "schema": "gdn-long-context-census.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "dry_run": bool(a.dry_run),
            "planned_cases": len(probes),
            "full_grid_cases": suite_size(
                a.lengths, list(range(a.seeds)), a.positions, a.events, a.conditions
            ),
            "skipped_invalid_grid_cases": (
                suite_size(a.lengths, list(range(a.seeds)), a.positions, a.events, a.conditions)
                - len(probes)
            ) if not a.max_samples else None,
            "skipping_policy": (
                "Discard entire matched group when required text does not fit, "
                "needle placement is off by >0.10 fraction, or matched target offsets differ."
            ),
            "notes": (
                "Synthetic probes are NOT official RULER. GDN checkpoints with "
                "max_position_embeddings=2048 are stressed out of distribution "
                "at longer lengths. Answer space is four next-token labels."
            ),
        }
        _write_json(manifest_path, manifest)
        with plan_path.open("w", encoding="utf-8") as fh:
            for p in probes:
                fh.write(json.dumps(p.to_meta(), ensure_ascii=False) + "\n")
    print(f"Prepared {len(probes)} probes; requested context lengths: {a.lengths}", flush=True)
    if a.dry_run:
        for p in probes[:3]:
            print(json.dumps(p.to_meta(), ensure_ascii=False), flush=True)
        print(f"Saved {plan_path} and {manifest_path} (NO model loaded)", flush=True)
        return 0

    if results_path.exists() and not a.resume:
        raise SystemExit("results.jsonl already exists; use --resume")
    completed: set[str] = set()
    if results_path.exists():
        with results_path.open(encoding="utf-8") as fh:
            for ln in fh:
                rec = json.loads(ln)
                if rec.get("status") in ("ok", "error"):
                    completed.add(rec["probe_id"])
    evaluator = PretrainedCausalLM(
        a.model, tokenizer_id=a.tokenizer, revision=a.revision,
        device=a.device, dtype=a.dtype, allow_extrapolation=a.allow_extrapolation,
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["loaded_model"] = evaluator.info.__dict__
    _write_json(manifest_path, manifest)
    # Deduplicate choice-token validation by actual QUESTION text, not model only.
    choice_cache: dict[str, dict[str, int]] = {}
    evaluated = 0
    with results_path.open("a", encoding="utf-8") as fh:
        for i, p in enumerate(probes, 1):
            if p.probe_id in completed:
                continue
            suffix = p.question_suffix
            if suffix not in choice_cache:
                choice_cache[suffix] = single_token_choices(tok, suffix)
            try:
                r = evaluator.score(p.token_ids, choice_cache[suffix])
                before = None
                if a.preedit_probes and p.event_token is not None:
                    # No future-event tokens enter this probe. The query is
                    # inserted immediately before the first interfering write.
                    # Revision probes ask for the most-recent/earliest value:
                    # before all revisions both should equal initial_answer.
                    pre_ids = (
                        p.token_ids[:p.event_token]
                        + tuple(tok.encode(suffix, add_special_tokens=False))
                    )
                    before = evaluator.score(pre_ids, choice_cache[suffix])
                margin = (
                    r["log_prob_choice"][p.answer]
                    - max(v for k, v in r["log_prob_choice"].items() if k != p.answer)
                )
                record = {
                    **p.to_meta(), "status": "ok",
                    "prediction": r["prediction"],
                    "correct": int(r["prediction"] == p.answer),
                    "margin": margin,
                    "gold_conditional_nll": -r["log_prob_choice"][p.answer],
                    "gold_full_nll": -r["log_prob_full"][p.answer],
                    "choice_log_probs": r["log_prob_choice"],
                    "latency_ms": r["latency_ms"],
                    "peak_allocated_bytes": r["peak_allocated_bytes"],
                    "is_context_extrapolated": bool(
                        evaluator.context is not None and p.actual_length > evaluator.context
                    ),
                    "preedit_correct": (
                        int(before["prediction"] == p.initial_answer)
                        if before is not None else None
                    ),
                    "preedit_gold_conditional_nll": (
                        -before["log_prob_choice"][p.initial_answer]
                        if before is not None else None
                    ),
                    "preedit_prompt_tokens": len(pre_ids) if before is not None else None,
                }
            except Exception as exc:
                if a.error_policy == "stop":
                    raise
                record = {
                    **p.to_meta(), "status": "error",
                    "error_type": type(exc).__name__, "error": str(exc),
                }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            evaluated += 1
            if evaluated % a.log_every == 0 or i == len(probes):
                print(
                    f"{i}/{len(probes)} attempted; {evaluated} newly evaluated; "
                    f"latest={p.condition} {p.actual_length}t; status={record['status']}",
                    flush=True,
                )
    print(
        f"Saved: {results_path}\n"
        f"Next: python summarize.py --run {a.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
