#!/usr/bin/env python3
"""Honest single-state LM efficiency: optimized isotropic vs full geometry.

Runs one mode per process (use the accompanying shell wrapper) so peak CUDA memory is
measured cleanly. The isotropic arm is a real efficient baseline: it never builds the
covariance or a Cholesky factor. The full arm is the corrected full-geometry path.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from data_split import RowSplit, build_panel  # noqa: E402
from fair_models import FairLM, FairLMConfig  # noqa: E402
from pretrain_branch import PackedCorpus  # noqa: E402


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packed-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", required=True, choices=["isotropic", "full", "rayleigh"])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seq-len", type=int, default=16384)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d-key", type=int, default=64)
    p.add_argument("--d-value", type=int, default=64)
    p.add_argument("--d-route", type=int, default=32)
    p.add_argument("--chunk-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=12345)
    p.add_argument("--val-per-mille", type=int, default=20)
    p.add_argument("--test-per-mille", type=int, default=10)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--vocab-size", type=int, default=32000)
    return p.parse_args()


def make_model(a: argparse.Namespace) -> FairLM:
    torch.manual_seed(a.seed)
    cfg = FairLMConfig(
        mixer="single_geo",
        geometry_mode=a.mode,
        vocab_size=a.vocab_size,
        d_model=a.d_model,
        n_layers=a.layers,
        n_heads=a.heads,
        d_key=a.d_key,
        d_value=a.d_value,
        d_route=a.d_route,
        ff_mult=4,
        max_states=1,
        protect=8.0,
        covariance="full",
        differentiate_covariance=True,
        chunk_len=a.chunk_len,
        grad_checkpoint=True,
    )
    return FairLM(cfg).to(a.device)


def evaluate(model: FairLM, panel: torch.Tensor, batch_size: int) -> dict:
    model.eval()
    loss = 0.0
    correct = 0
    tokens = 0
    batches = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, panel.shape[0], batch_size):
            ids = panel[start : start + batch_size]
            logits, _ = model(ids)
            flat = logits[:, :-1].reshape(-1, logits.shape[-1])
            target = ids[:, 1:].reshape(-1)
            loss += float(F.cross_entropy(flat, target))
            correct += int((flat.argmax(-1) == target).sum())
            tokens += target.numel()
            batches += 1
    if panel.device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "loss": loss / max(batches, 1),
        "accuracy": correct / max(tokens, 1),
        "tokens": tokens,
        "seconds": time.perf_counter() - t0,
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
    panel_ids = panel.tensor(device)

    model = make_model(a)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    gen = torch.Generator().manual_seed(a.seed + 1)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    model.train()
    t0 = time.perf_counter()
    losses = []
    for step in range(1, a.steps + 1):
        ids = torch.from_numpy(corpus.sample_split(a.batch_size, gen, split)).to(device)
        opt.zero_grad(set_to_none=True)
        logits, _ = model(ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if step % 100 == 0:
            print(
                f"[{a.mode}] step {step} loss {np.mean(losses[-100:]):.4f}", flush=True
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    reserved = torch.cuda.max_memory_reserved() / 1e9 if device.type == "cuda" else 0.0
    eval_metrics = evaluate(model, panel_ids, a.batch_size)

    result = {
        "mode": a.mode,
        "config": asdict(model.config),
        "steps": a.steps,
        "train_loss_last100": float(np.mean(losses[-100:])),
        "seconds": wall,
        "seconds_per_step": wall / a.steps,
        "peak_allocated_gb": peak,
        "peak_reserved_gb": reserved,
        "eval": eval_metrics,
        "panel_sha256": panel.tokens_sha256,
    }
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / f"{a.mode}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
