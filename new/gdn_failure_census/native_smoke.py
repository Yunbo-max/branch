#!/usr/bin/env python3
"""Native pretrained-GDN smoke: strict FLA load + ONE real token-logit forward.

GPU required. No training, no accuracy claim, no large-context sweep.
Run this BEFORE the smoke/competence/screen stages. Compatible with 4090 as
well as H100/H200 if a working CUDA/FLA environment is installed.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from backend import PretrainedCausalLM, single_token_choices
from provenance import hub_revision, resolve_inputs
from spec import make_probe


def audit_scores(result: dict) -> None:
    """Check the observable result is finite and a complete four-choice distribution."""
    probs = result.get("log_prob_choice")
    if set(probs or {}) != set("ABCD"):
        raise ValueError("Native model must score exactly four A/B/C/D candidates")
    values = [probs[x] for x in "ABCD"]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        raise ValueError("Native model yielded nonfinite four-choice log probabilities")
    norm = sum(math.exp(v) for v in values)
    if not math.isclose(norm, 1.0, abs_tol=2e-5, rel_tol=2e-5):
        raise ValueError(f"four-choice probabilities do not sum to one: {norm}")
    if not isinstance(result.get("prediction"), str) or result["prediction"] not in ("A", "B", "C", "D"):
        raise ValueError("Native model produced invalid choice")
    latency = result.get("latency_ms")
    if type(latency) not in (float, int) or not math.isfinite(latency) or latency < 0:
        raise ValueError("Native smoke requires finite nonnegative timing")


def run_native(model: str, *, tokenizer_id: str | None = None,
               revision: str | None = None, tokenizer_revision: str | None = None,
               device: str = "cuda", dtype: str = "bf16",
               length: int = 512) -> dict:
    if length < 512:
        raise ValueError("native smoke expects at least 512 tokens for the full task")
    # Use the same identity policy as the census runner: immutable Hub commits
    # or a fingerprint of local checkpoint/tokenizer bytes. Local fingerprints
    # belong in the receipt, never in the Hugging Face revision argument.
    inputs = resolve_inputs(argparse.Namespace(
        backend="hf", model=model, tokenizer=tokenizer_id,
        revision=revision, tokenizer_revision=tokenizer_revision,
    ))
    evaluator = PretrainedCausalLM(
        inputs["model_id"], tokenizer_id=inputs["tokenizer_id"],
        revision=hub_revision(inputs["model_revision"]),
        tokenizer_revision=hub_revision(inputs["tokenizer_revision"]),
        device=device, dtype=dtype,
        allow_extrapolation=False,
    )
    tok = evaluator.tok
    probe = make_probe(
        tok, condition="plain", length=length, seed=2718,
        needle_fraction=0.25, num_events=0,
    )
    choices = single_token_choices(tok, probe.question_suffix)
    result = evaluator.score(probe.token_ids, choices)
    audit_scores(result)
    return {
        "schema": "gdn-native-smoke.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "native_forward_success": True,
        "model_info": evaluator.info.__dict__,
        "input_provenance": inputs,
        "python": platform.python_version(),
        "probe": {
            "tokens": length, "condition": "plain",
            "prompt_hash": probe.prompt_hash, "choice_ids": choices,
            "gold": probe.answer, "predicted": result["prediction"],
            "correct": bool(result["prediction"] == probe.answer),
            "conditional_gold_nll": -result["log_prob_choice"][probe.answer],
        },
        "runtime": {
            "forward_plus_scoring_ms": result["latency_ms"],
            "peak_allocated_bytes": result["peak_allocated_bytes"],
        },
        "limitations": [
            "This is one successful strict pretrained model load and forward, not a long-context benchmark.",
            "The 512-token gold-choice accuracy is a single anecdotal example, not a scientific result.",
            "No causal comparison with MARCH, GDN-2 or GeoDelta.",
        ],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="puigde/gated-deltanet-360M-15B-slimpajama")
    ap.add_argument("--tokenizer")
    ap.add_argument("--revision")
    ap.add_argument("--tokenizer-revision")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    ap.add_argument("--length", default=512, type=int)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    report = run_native(
        args.model, tokenizer_id=args.tokenizer, revision=args.revision,
        tokenizer_revision=args.tokenizer_revision, device=args.device,
        dtype=args.dtype, length=args.length,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
