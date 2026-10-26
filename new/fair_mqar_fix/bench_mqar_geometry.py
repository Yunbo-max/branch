#!/usr/bin/env python3
"""Associative-recall diagnostic for the single-state geometry ablation.

Trains two otherwise identical single-state Branch models on MQAR:

* ``isotropic``: H = lambda I          (protect = 0)
* ``full``:      H = lambda I + mu C   (protect > 0)

Then evaluates both on a grid that changes the number of key-value associations and the
retention gap. This separates the geometry effect from fork/read/routing/state-count
changes.
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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from fair_models import FairLM, FairLMConfig  # noqa: E402
from mqar import MQARConfig, mqar_loss_acc  # noqa: E402


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-n", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--train-pairs", type=int, default=8)
    p.add_argument("--train-gap", type=int, default=32)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--d-key", type=int, default=32)
    p.add_argument("--d-value", type=int, default=32)
    p.add_argument("--d-route", type=int, default=16)
    p.add_argument("--chunk-len", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--eval-pairs", type=int, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--eval-gaps", type=int, nargs="+", default=[0, 32, 64, 128])
    return p.parse_args()


def make_model(a: argparse.Namespace, protect: float, vocab: int) -> FairLM:
    torch.manual_seed(a.seed)
    cfg = FairLMConfig(
        mixer="chunked",
        vocab_size=vocab,
        d_model=a.d_model,
        n_layers=a.layers,
        n_heads=a.heads,
        d_key=a.d_key,
        d_value=a.d_value,
        d_route=a.d_route,
        ff_mult=2,
        max_states=1,
        protect=protect,
        fork_cost=0.0,
        protection_decay=0.95,
        covariance="full",
        differentiate_covariance=True,
        chunk_len=a.chunk_len,
        grad_checkpoint=False,
    )
    return FairLM(cfg).to(a.device)


def evaluate_grid(model: FairLM, a: argparse.Namespace, vocab: int) -> list[dict]:
    model.eval()
    out = []
    with torch.no_grad():
        for pairs in a.eval_pairs:
            for gap in a.eval_gaps:
                if 2 * pairs + gap + 2 > a.seq_len:
                    continue
                cfg = MQARConfig(
                    seq_len=a.seq_len,
                    num_pairs=pairs,
                    gap=gap,
                    seed=9000 + pairs * 100 + gap,
                )
                tokens, _ = cfg.make_fixed(a.eval_n, cfg.seed)
                loss = 0.0
                correct = 0
                batches = 0
                for start in range(0, a.eval_n, a.eval_batch_size):
                    ids = torch.from_numpy(
                        tokens[start : start + a.eval_batch_size]
                    ).to(a.device)
                    logits, _ = model(ids)
                    batch_loss, batch_correct = mqar_loss_acc(logits, ids, ids[:, -1])
                    loss += float(batch_loss)
                    correct += batch_correct
                    batches += 1
                out.append(
                    {
                        "num_pairs": pairs,
                        "gap": gap,
                        "loss": loss / max(batches, 1),
                        "accuracy": correct / max(a.eval_n, 1),
                        "chance": 1.0 / cfg.n_values,
                    }
                )
    model.train()
    return out


def run_arm(
    a: argparse.Namespace, name: str, protect: float, batches, vocab: int
) -> dict:
    model = make_model(a, protect, vocab)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    if a.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    losses = []
    model.train()
    for ids_np in batches:
        ids = torch.from_numpy(ids_np).to(a.device)
        opt.zero_grad(set_to_none=True)
        logits, _ = model(ids)
        loss, _ = mqar_loss_acc(logits, ids, ids[:, -1])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    if a.device == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if a.device == "cuda" else 0.0
    grid = evaluate_grid(model, a, vocab)
    result = {
        "protect": protect,
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": float(np.mean(losses[-50:])) if losses else None,
        "seconds_per_step": wall / max(len(losses), 1),
        "peak_allocated_gb": peak,
        "grid": grid,
    }
    print(f"[{name}] {json.dumps(result, default=str)}", flush=True)
    return result


def main() -> int:
    a = parse()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    train_cfg = MQARConfig(
        seq_len=a.seq_len,
        num_pairs=a.train_pairs,
        gap=a.train_gap,
        seed=a.seed,
    )
    rng = np.random.default_rng(a.seed)
    batches = [train_cfg.make_batch(a.batch_size, rng)[0] for _ in range(a.steps)]
    print(
        f"train_pairs={a.train_pairs} train_gap={a.train_gap} seq_len={a.seq_len} "
        f"batches={len(batches)} vocab={train_cfg.vocab_size}",
        flush=True,
    )
    results = {
        "train_config": asdict(train_cfg),
        "args": {k: str(v) for k, v in vars(a).items()},
        "arms": {},
    }
    for name, protect in (("isotropic", 0.0), ("full", 8.0)):
        results["arms"][name] = run_arm(a, name, protect, batches, train_cfg.vocab_size)
        if a.device == "cuda":
            torch.cuda.empty_cache()
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "bench_mqar.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    print("BENCH_MQAR_JSON " + str(a.output / "bench_mqar.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
