#!/usr/bin/env python3
"""Evaluate checkpoints on one fixed validation panel.

Supports current Branch checkpoints and the legacy MARCH port, whose checkpoint contains
the removed ``anchor_embed`` parameter. Legacy MARCH is loaded through the archived
``new/fair_mqar/chunked_march.py`` module so the comparison is against the model that was
actually trained, not a re-interpretation with the fixed content-conditioned keys.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict, fields
from pathlib import Path
import sys

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from data_split import RowSplit, build_panel  # noqa: E402
from fair_models import FairLM, FairLMConfig  # noqa: E402
from pretrain_branch import PackedCorpus  # noqa: E402


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packed-dir", type=Path, required=True)
    p.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="name=path; may be repeated",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--old-code-dir",
        type=Path,
        default=_HERE.parent / "fair_mqar",
        help="directory containing the legacy chunked_march.py",
    )
    p.add_argument("--seq-len", type=int, default=16384)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--split-seed", type=int, default=12345)
    p.add_argument("--val-per-mille", type=int, default=20)
    p.add_argument("--test-per-mille", type=int, default=10)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    return p.parse_args()


def load_legacy_march(old_dir: Path):
    path = old_dir / "chunked_march.py"
    spec = importlib.util.spec_from_file_location("legacy_chunked_march", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["legacy_chunked_march"] = module
    spec.loader.exec_module(module)
    return module


def build_from_checkpoint(path: Path, old_dir: Path, device) -> tuple[FairLM, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw = ckpt["config"]
    names = {f.name for f in fields(FairLMConfig)}
    cfg = FairLMConfig(**{k: v for k, v in raw.items() if k in names})
    model = FairLM(cfg)
    if any(k.endswith("anchor_embed") for k in ckpt["model"]):
        legacy = load_legacy_march(old_dir)
        mcfg = legacy.ChunkedMARCHConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_key=cfg.d_key,
            d_value=cfg.d_value,
            d_route=cfg.d_route,
            max_states=cfg.max_states,
            conv_kernel=cfg.conv_kernel,
            tau=cfg.tau,
            ridge=cfg.ridge,
            read_topk=cfg.read_topk,
            chunk_len=cfg.chunk_len,
        )
        for block in model.blocks:
            block.mixer = legacy.ChunkedMARCHMixer(mcfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, ckpt


def evaluate(model: FairLM, tokens: torch.Tensor, batch_size: int) -> dict:
    loss = 0.0
    correct = 0
    count = 0
    batches = 0
    with torch.no_grad():
        for start in range(0, tokens.shape[0], batch_size):
            ids = tokens[start : start + batch_size]
            logits, _ = model(ids)
            flat_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
            flat_ids = ids[:, 1:].reshape(-1)
            loss += float(torch.nn.functional.cross_entropy(flat_logits, flat_ids))
            correct += int((flat_logits.argmax(-1) == flat_ids).sum())
            count += flat_ids.numel()
            batches += 1
    return {
        "loss": loss / max(batches, 1),
        "accuracy": correct / max(count, 1),
        "tokens": count,
    }


def main() -> int:
    a = parse()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(a.device)
    corpus = PackedCorpus(a.packed_dir, a.seq_len)
    split = RowSplit(
        seed=a.split_seed,
        val_per_mille=a.val_per_mille,
        test_per_mille=a.test_per_mille,
    )
    panel = build_panel(
        corpus,
        a.eval_batches * a.batch_size,
        split,
        torch.Generator().manual_seed(a.split_seed + 7),
    )
    tokens = panel.tensor(device)
    print(f"panel={len(panel.indices)} rows sha256={panel.tokens_sha256}", flush=True)

    results = {"panel_sha256": panel.tokens_sha256, "checkpoints": {}}
    for item in a.checkpoint:
        name, _, raw_path = item.partition("=")
        path = Path(raw_path)
        model, ckpt = build_from_checkpoint(path, a.old_code_dir, device)
        metrics = evaluate(model, tokens, a.batch_size)
        step = int(ckpt.get("step", -1))
        results["checkpoints"][name] = {
            "path": str(path),
            "step": step,
            "config": {k: str(v) for k, v in asdict(model.config).items()},
            **metrics,
        }
        print(f"[{name}] step={step} {json.dumps(metrics)}", flush=True)
        del model
        if a.device == "cuda":
            torch.cuda.empty_cache()

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(results, indent=2, default=str))
    print("EVAL_JSON " + str(a.output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
