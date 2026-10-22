#!/usr/bin/env python3
"""Train Branch Memory on native MQAR (Zoology generator, aligned labels).

Scope: a research pilot. It reports query-position accuracy/loss on held-out panels;
it is not a claim that this budget matches the Zoology paper.

Example (CPU resource calibration):
    python train_mqar.py --steps 200 --batch-size 16 --seq-len 128 --num-kv-pairs 8 \
      --vocab-size 8192 --d-model 128 --heads 2 --layers 2 --d-key 32 --d-value 32 \
      --d-route 16 --max-states 8 --mode branch --covariance full \
      --differentiate-covariance --lr 3e-4 --threads 4 --output results/mqar_pilot
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from branch_memory import LMConfig, TinyMemoryLM
from mqar_adapter import MQARBatch, build_panels, mqar_accuracy, mqar_loss


def make_model(a, vocab_size: int) -> TinyMemoryLM:
    cfg = LMConfig(
        vocab_size=vocab_size,
        d_model=a.d_model,
        n_layers=a.layers,
        n_heads=a.heads,
        d_key=a.d_key,
        d_value=a.d_value,
        d_route=a.d_route,
        max_states=a.max_states,
        mode=a.mode,
        tau=a.tau,
        ridge=a.ridge,
        protect=a.protect,
        fork_cost=a.fork_cost,
        covariance=a.covariance,
        differentiate_covariance=a.differentiate_covariance,
        protection_decay=a.protection_decay,
    )
    return TinyMemoryLM(cfg)


def sample(panel: MQARBatch, batch_size: int, generator: torch.Generator, device) -> tuple:
    idx = torch.randint(panel.inputs.shape[0], (batch_size,), generator=generator)
    return panel.inputs[idx].to(device), panel.labels[idx].to(device)


@torch.no_grad()
def evaluate(model, panel: MQARBatch, batch_size: int, device) -> dict:
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for start in range(0, panel.inputs.shape[0], batch_size):
        x = panel.inputs[start:start + batch_size].to(device)
        y = panel.labels[start:start + batch_size].to(device)
        logits, _ = model(x)
        loss = mqar_loss(logits, y)
        acc, n = mqar_accuracy(logits, y)
        total_loss += float(loss) * n
        total_correct += round(acc * n)
        total_n += n
    model.train()
    return {
        "loss": total_loss / max(total_n, 1),
        "accuracy": total_correct / max(total_n, 1),
        "labeled_positions": total_n,
        "examples": int(panel.inputs.shape[0]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # MQAR protocol
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--num-kv-pairs", type=int, default=8)
    p.add_argument("--power-a", type=float, default=0.01)
    p.add_argument("--num-passes", type=int, default=1)
    p.add_argument("--num-examples", type=int, default=4096)
    p.add_argument("--seed", type=int, default=17)
    # Model
    p.add_argument("--mode", choices=["branch", "snapshot", "delta"], default="branch")
    p.add_argument("--covariance", choices=["full", "diagonal"], default="full")
    p.add_argument("--differentiate-covariance", action="store_true")
    p.add_argument("--protection-decay", type=float, default=0.95)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--d-key", type=int, default=32)
    p.add_argument("--d-value", type=int, default=32)
    p.add_argument("--d-route", type=int, default=16)
    p.add_argument("--max-states", type=int, default=8)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--protect", type=float, default=8.0)
    p.add_argument("--fork-cost", type=float, default=0.02)
    # Optimisation
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--output", type=Path, default=Path("results/mqar_pilot"))
    # Length / load extrapolation panels
    p.add_argument("--extrapolate-seq-lens", type=int, nargs="*", default=[])
    p.add_argument("--extrapolate-kv", type=int, nargs="*", default=[])
    a = p.parse_args()

    if a.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)

    panels = build_panels(
        vocab_size=a.vocab_size, seq_len=a.seq_len, num_kv_pairs=a.num_kv_pairs,
        num_examples=a.num_examples, seed=a.seed, power_a=a.power_a,
        num_passes=a.num_passes,
    )
    train_panel, valid_panel, test_panel = panels["train"], panels["valid"], panels["test"]

    extrapolation: dict[str, MQARBatch] = {}
    for L in a.extrapolate_seq_lens:
        extrapolation[f"len{L}"] = build_panels(
            vocab_size=a.vocab_size + 2 * L, seq_len=L, num_kv_pairs=a.num_kv_pairs,
            num_examples=512, seed=a.seed + L,
            power_a=a.power_a, num_passes=a.num_passes,
        )["test"]
    for K in a.extrapolate_kv:
        extrapolation[f"kv{K}"] = build_panels(
            vocab_size=a.vocab_size, seq_len=a.seq_len, num_kv_pairs=K,
            num_examples=512, seed=a.seed + 1000 + K,
            power_a=a.power_a, num_passes=a.num_passes,
        )["test"]

    model = make_model(a, a.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    generator = torch.Generator().manual_seed(a.seed + 1)

    a.output.mkdir(parents=True, exist_ok=True)
    initial = {
        "valid": evaluate(model, valid_panel, a.eval_batch_size, device),
        "test": evaluate(model, test_panel, a.eval_batch_size, device),
    }

    start = time.perf_counter()
    history = []
    with (a.output / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for step in range(1, a.steps + 1):
            model.train()
            x, y = sample(train_panel, a.batch_size, generator, device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = mqar_loss(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), a.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            acc, n = mqar_accuracy(logits.detach(), y)
            record = {
                "step": step,
                "train_loss": float(loss.detach()),
                "train_accuracy": acc,
                "grad_norm": float(grad_norm),
                "elapsed_seconds": time.perf_counter() - start,
            }
            if step == 1 or step % a.eval_every == 0 or step == a.steps:
                record["valid"] = evaluate(model, valid_panel, a.eval_batch_size, device)
                record["test"] = evaluate(model, test_panel, a.eval_batch_size, device)
            history.append(record)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step == 1 or step % a.log_every == 0 or step == a.steps:
                print(json.dumps({k: v for k, v in record.items() if k != "elapsed_seconds"}),
                      flush=True)

    wall = time.perf_counter() - start
    final = {
        "valid": evaluate(model, valid_panel, a.eval_batch_size, device),
        "test": evaluate(model, test_panel, a.eval_batch_size, device),
    }
    extrapolated = {name: evaluate(model, panel, a.eval_batch_size, device)
                    for name, panel in extrapolation.items()}

    report = {
        "scope": "native MQAR pilot; aligned labels; not a Zoology-paper reproduction",
        "protocol": {
            "vocab_size": a.vocab_size, "seq_len": a.seq_len,
            "num_kv_pairs": a.num_kv_pairs, "power_a": a.power_a,
            "num_passes": a.num_passes, "random_non_queries": True,
            "num_examples": a.num_examples, "seed": a.seed,
            "panels": {name: {"sha256": panel.sha256, "examples": int(panel.inputs.shape[0]),
                              "alignment": panel.alignment}
                       for name, panel in panels.items()},
        },
        "model": asdict(model.config),
        "parameters": sum(p.numel() for p in model.parameters()),
        "training": {"steps": a.steps, "batch_size": a.batch_size, "lr": a.lr,
                     "wall_seconds": wall,
                     "seconds_per_step": wall / max(a.steps, 1)},
        "initial": initial,
        "final": final,
        "extrapolation": extrapolated,
        "history_tail": history[-5:],
        "environment": {"python": __import__("platform").python_version(),
                        "torch": torch.__version__, "device": str(device),
                        "threads": a.threads},
        "limitations": [
            "aligned-label CE; do not reuse next_token_loss",
            "CPU pilot budget; not matched to MARCH training compute",
            "full-covariance write path is O(T*M*K^3) without fused kernels",
            "extrapolation panels are separate sample draws, not a matched-length study",
        ],
    }
    (a.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save({"config": asdict(model.config), "model": model.state_dict(),
                "step": a.steps, "seed": a.seed,
                "panels": {k: v.sha256 for k, v in panels.items()}},
               a.output / "checkpoint.pt")
    print(json.dumps({"initial": initial, "final": final,
                      "extrapolation": extrapolated, "wall_seconds": wall}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
