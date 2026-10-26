#!/usr/bin/env python3
"""G2: isotropic / Rayleigh / full single-state geometry on MQAR.

All arms use the same backbone, initial weights per seed, batch stream, chunk length and
fixed evaluation samples. Only the write-direction rule changes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
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
    p.add_argument("--modes", nargs="+", default=["isotropic", "rayleigh", "full"])
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 42])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-n", type=int, default=1024)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--train-pairs", type=int, default=8)
    p.add_argument("--train-gap", type=int, default=32)
    p.add_argument("--eval-pairs", type=int, nargs="+", default=[2, 4, 8, 16, 24])
    p.add_argument("--eval-gaps", type=int, nargs="+", default=[0, 32, 64, 128])
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--d-key", type=int, default=32)
    p.add_argument("--d-value", type=int, default=32)
    p.add_argument("--d-route", type=int, default=16)
    p.add_argument("--chunk-len", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    return p.parse_args()


def make_model(a: argparse.Namespace, mode: str, vocab: int, seed: int) -> FairLM:
    torch.manual_seed(seed)
    cfg = FairLMConfig(
        mixer="single_geo",
        geometry_mode=mode,
        vocab_size=vocab,
        d_model=a.d_model,
        n_layers=a.layers,
        n_heads=a.heads,
        d_key=a.d_key,
        d_value=a.d_value,
        d_route=a.d_route,
        ff_mult=2,
        max_states=1,
        protect=8.0,
        covariance="full",
        differentiate_covariance=True,
        chunk_len=a.chunk_len,
        grad_checkpoint=False,
    )
    return FairLM(cfg).to(a.device)


def evaluate(model: FairLM, a: argparse.Namespace, vocab: int) -> list[dict]:
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


def run(a: argparse.Namespace, mode: str, seed: int, batches, vocab: int) -> dict:
    model = make_model(a, mode, vocab, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
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
    grid = evaluate(model, a, vocab)
    print(
        f"[{mode} seed={seed}] train_last={np.mean(losses[-50:]):.4f} "
        f"grid={json.dumps(grid)}",
        flush=True,
    )
    return {
        "seed": seed,
        "train_loss_last": float(np.mean(losses[-50:])),
        "grid": grid,
    }


def main() -> int:
    a = parse()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    train_cfg = MQARConfig(
        seq_len=a.seq_len,
        num_pairs=a.train_pairs,
        gap=a.train_gap,
        seed=a.seeds[0],
    )
    results = {
        "train_config": asdict(train_cfg),
        "args": {k: str(v) for k, v in vars(a).items()},
        "runs": {},
    }
    for seed in a.seeds:
        rng = np.random.default_rng(seed)
        batches = [train_cfg.make_batch(a.batch_size, rng)[0] for _ in range(a.steps)]
        for mode in a.modes:
            key = f"{mode}_seed{seed}"
            results["runs"][key] = run(a, mode, seed, batches, train_cfg.vocab_size)
            if a.device == "cuda":
                torch.cuda.empty_cache()

    # mean / range over seeds for each (mode, pairs, gap)
    summary = {}
    for mode in a.modes:
        cells = {}
        grids = [results["runs"][f"{mode}_seed{seed}"]["grid"] for seed in a.seeds]
        for i, cell in enumerate(grids[0]):
            accs = [g[i]["accuracy"] for g in grids]
            cells[f"{cell['num_pairs']}/{cell['gap']}"] = {
                "num_pairs": cell["num_pairs"],
                "gap": cell["gap"],
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "accuracy": accs,
            }
        summary[mode] = cells
    results["summary"] = summary

    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "mqar_modes.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    print("MQAR_MODES_JSON " + str(a.output / "mqar_modes.json"), flush=True)

    print("\nmode      pairs/gap   mean acc   std")
    for mode in a.modes:
        for key, cell in summary[mode].items():
            print(
                f"{mode:<9} {key:<10} {cell['mean_accuracy']:.4f}   "
                f"{cell['std_accuracy']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
