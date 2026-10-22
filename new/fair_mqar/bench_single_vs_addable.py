#!/usr/bin/env python3
"""Controlled benchmark: single-state Branch vs fork/addable Branch.

Both arms use the same backbone, parameters, data order, optimizer and fixed evaluation
panel. Only ``max_states`` differs (1 vs N). Reports training time, peak GPU memory,
evaluation loss and token accuracy, plus fork/active-state telemetry for the addable arm.
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
    p.add_argument("--seq-len", type=int, default=16384)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d-key", type=int, default=64)
    p.add_argument("--d-value", type=int, default=64)
    p.add_argument("--d-route", type=int, default=32)
    p.add_argument("--chunk-len", type=int, default=256)
    p.add_argument("--single-states", type=int, default=1)
    p.add_argument("--addable-states", type=int, default=64)
    p.add_argument("--fork-cost", type=float, default=0.02)
    p.add_argument("--protect", type=float, default=8.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--split-seed", type=int, default=12345)
    p.add_argument("--val-per-mille", type=int, default=20)
    p.add_argument("--test-per-mille", type=int, default=10)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--arm", default="both", choices=["single", "addable", "both"],
                   help="run one arm per process to isolate CUDA memory")
    p.add_argument("--vocab-size", type=int, default=32000)
    return p.parse_args()


def make_model(a: argparse.Namespace, states: int) -> FairLM:
    torch.manual_seed(a.seed)
    cfg = FairLMConfig(
        mixer="chunked",
        vocab_size=a.vocab_size,
        d_model=a.d_model,
        n_layers=a.layers,
        n_heads=a.heads,
        d_key=a.d_key,
        d_value=a.d_value,
        d_route=a.d_route,
        ff_mult=4,
        max_states=states,
        protect=a.protect,
        fork_cost=a.fork_cost,
        covariance="full",
        differentiate_covariance=True,
        chunk_len=a.chunk_len,
        grad_checkpoint=True,
    )
    return FairLM(cfg).to(a.device)


def evaluate(model: FairLM, batches, device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    tokens = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for ids in batches:
            ids = torch.from_numpy(ids).to(device)
            logits, _ = model(ids)
            flat_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
            flat_ids = ids[:, 1:].reshape(-1)
            total_loss += float(F.cross_entropy(flat_logits, flat_ids))
            correct += int((flat_logits.argmax(-1) == flat_ids).sum())
            tokens += flat_ids.numel()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (
        total_loss / max(len(batches), 1),
        correct / max(tokens, 1),
        time.perf_counter() - t0,
    )


def train_arm(model: FairLM, batches, a: argparse.Namespace) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    model.train()
    if a.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    losses = []
    last_stats = None
    for ids in batches:
        ids = torch.from_numpy(ids).to(a.device)
        opt.zero_grad(set_to_none=True)
        logits, stats = model(ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        last_stats = stats[-1] if stats else None
    if a.device == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if a.device == "cuda" else 0.0
    reserved = torch.cuda.max_memory_reserved() / 1e9 if a.device == "cuda" else 0.0
    telemetry = {}
    if last_stats is not None:
        active = last_stats.get("active_states")
        fork = last_stats.get("fork")
        if active is not None:
            heads = max(model.config.n_heads, 1)
            telemetry["active_mean_per_head"] = float(active.mean()) / heads
            telemetry["active_max_per_head"] = float(active.max()) / heads
        if fork is not None:
            telemetry["fork_fraction"] = float(fork.float().mean())
    return {
        "steps": len(losses),
        "loss_first": losses[0] if losses else None,
        "loss_last": float(np.mean(losses[-10:])) if losses else None,
        "seconds": wall,
        "seconds_per_step": wall / max(len(losses), 1),
        "peak_allocated_gb": peak,
        "peak_reserved_gb": reserved,
        "telemetry": telemetry,
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
    eval_batches = [
        panel.tokens[i : i + a.batch_size]
        for i in range(0, len(panel.tokens), a.batch_size)
    ]
    gen = torch.Generator().manual_seed(a.seed)
    train_batches = [
        corpus.sample_split(a.batch_size, gen, split) for _ in range(a.steps)
    ]
    print(
        f"corpus={len(corpus):,} rows | train_batches={len(train_batches)} "
        f"eval_batches={len(eval_batches)} panel_sha={panel.tokens_sha256[:12]} "
        f"seq_len={a.seq_len} chunk={a.chunk_len}",
        flush=True,
    )

    config_model = make_model(a, a.addable_states)
    config = asdict(config_model.config)
    del config_model
    if a.device == "cuda":
        torch.cuda.empty_cache()
    a.output.mkdir(parents=True, exist_ok=True)
    results = {"config": config, "panel_sha256": panel.tokens_sha256}
    arms = (("single", a.single_states), ("addable", a.addable_states))
    if a.arm != "both":
        arms = tuple(item for item in arms if item[0] == a.arm)
    for name, states in arms:
        model = make_model(a, states)
        train = train_arm(model, train_batches, a)
        loss, acc, eval_seconds = evaluate(model, eval_batches, device)
        results[name] = {
            **train,
            "eval_loss": loss,
            "eval_accuracy": acc,
            "eval_seconds": eval_seconds,
            "active_states_config": states,
        }
        print(f"[{name}] {json.dumps(results[name], default=str)}", flush=True)
        (a.output / f"{name}.json").write_text(
            json.dumps({**results, "arm": name}, indent=2, default=str)
        )
        del model
        if a.device == "cuda":
            torch.cuda.empty_cache()

    if a.arm == "both":
        (a.output / "bench.json").write_text(json.dumps(results, indent=2, default=str))
        print("BENCH_JSON " + str(a.output / "bench.json"), flush=True)
    else:
        print("BENCH_ARM_JSON " + str(a.output / f"{a.arm}.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
