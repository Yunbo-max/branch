"""One-block reference acceptance evaluation across held-out prompt-only inputs.

The frozen teacher is recomputed on each prefix: this is NOT a production
cached speculative-decoding benchmark. Results assess acceptance behaviour,
not speedup or target task accuracy.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from .training import load_prompts, verify_one


@torch.no_grad()
def evaluate_acceptance(drafter, teacher, *, data_path: str,
                        report_dir: str, horizon: int, samples: int, seed: int):
    if horizon < 1 or samples < 1:
        raise ValueError("horizon and samples must be positive")
    root = Path(report_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("refuse to overwrite evaluation output")
    root.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(data_path)
    rows = []
    for item in prompts:
        prefix = teacher.encode_prompt(item["prompt"])
        for sample in range(samples):
            identity = f"{seed}|{item['id']}|{sample}".encode()
            sample_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % (2**63 - 1)
            generator = torch.Generator(device=prefix.device).manual_seed(sample_seed)
            result = verify_one(drafter, teacher, prefix, horizon, generator=generator)
            rows.append({"prompt_id": item["id"], "replicate": sample,
                         "sample_seed": sample_seed, "accepted": result["accepted"],
                         "emitted_count": len(result["emitted"]),
                         "rejected_at": result["rejected_at"]})
    with (root / "one_block_rows.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    result = {"status": "reference_acceptance_not_speed_benchmark",
              "prompt_count": len(prompts), "samples_per_prompt": samples,
              "draft_horizon": horizon, "seed": seed,
              "mean_accepted_draft_tokens": sum(x["accepted"] for x in rows) / len(rows),
              "mean_emitted_tokens": sum(x["emitted_count"] for x in rows) / len(rows),
              "fully_accepted_fraction": sum(x["accepted"] == horizon for x in rows) / len(rows),
              "teacher_backend": "full_prefix_recompute_not_cached",
              "task_accuracy": None, "tokens_per_second": None}
    (root / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
