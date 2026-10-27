"""Fair MQAR trainer: one backbone, one data path, one budget; only the mixer differs.

    python train_fair.py --mixer march  --output results/fair_march
    python train_fair.py --mixer branch --output results/fair_branch
    python compare.py                  # runs both and prints the paired table
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path
import sys

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "branch_memory"))

from mqar_adapter import MQARBatch, build_panels, build_segment, mqar_accuracy, mqar_loss  # noqa: E402
from fair_models import MIXERS, FairLM, FairLMConfig, count_mixer_parameters  # noqa: E402


def default_config(a) -> FairLMConfig:
    return FairLMConfig(
        mixer=a.mixer, vocab_size=a.vocab_size, d_model=a.d_model, n_layers=a.layers,
        n_heads=a.heads, d_key=a.d_key, d_value=a.d_value, ff_mult=a.ff_mult,
        ff_extra=a.ff_extra,
        conv_kernel=a.conv_kernel, anchor_interval=a.anchor_interval,
        max_anchors=a.max_anchors, read_topk=a.read_topk, d_route=a.d_route,
        max_states=a.max_states, tau=a.tau, ridge=a.ridge, protect=a.protect,
        fork_cost=a.fork_cost, protection_decay=a.protection_decay,
        covariance=a.covariance, differentiate_covariance=a.differentiate_covariance,
        solver_refresh=a.solver_refresh, chunk_len=a.chunk_len,
    )


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mixer", choices=list(MIXERS), required=True)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--num-kv-pairs", type=int, default=8)
    p.add_argument("--power-a", type=float, default=0.01)
    p.add_argument("--num-examples", type=int, default=4096)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--d-key", type=int, default=32)
    p.add_argument("--d-value", type=int, default=32)
    p.add_argument("--d-route", type=int, default=32)
    p.add_argument("--ff-mult", type=int, default=2)
    p.add_argument("--ff-extra", type=int, default=0)
    p.add_argument("--conv-kernel", type=int, default=3)
    # MARCH
    p.add_argument("--anchor-interval", type=int, default=16)
    p.add_argument("--max-anchors", type=int, default=16)
    p.add_argument("--read-topk", type=int, default=4)
    # Branch Memory
    p.add_argument("--max-states", type=int, default=8)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--protect", type=float, default=8.0)
    p.add_argument("--fork-cost", type=float, default=0.02)
    p.add_argument("--protection-decay", type=float, default=0.95)
    p.add_argument("--covariance", choices=["full", "diagonal"], default="full")
    p.add_argument("--chunk-len", type=int, default=64,
                   help="chunked arm: tokens per Python iteration")
    p.add_argument("--solver-refresh", type=int, default=1,
                   help="branch_cached only: reuse the SPD factorization for N tokens")
    p.add_argument("--differentiate-covariance", action="store_true")
    # Optimisation
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--eval-examples", type=int, default=0,
                   help="cap valid/test at N examples (0 = full panel); keeps a large\n                          training pool from inflating the step-time measurement")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--online-train", action="store_true",
                   help="generate a fresh, deterministic MQAR training batch every step "
                        "instead of repeatedly sampling the fixed train panel")
    p.add_argument("--online-seed-offset", type=int, default=3000001,
                   help="training-example stream seed offset, separate from held-out panel seeds")
    p.add_argument("--output", type=Path, required=True)


@torch.no_grad()
def evaluate(model: FairLM, panel, batch_size: int, device) -> dict:
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for s in range(0, panel.inputs.shape[0], batch_size):
        x = panel.inputs[s:s + batch_size].to(device)
        y = panel.labels[s:s + batch_size].to(device)
        logits, _ = model(x)
        _, k = mqar_accuracy(logits, y)
        loss_sum += float(mqar_loss(logits, y)) * k
        acc, _ = mqar_accuracy(logits, y)
        correct += round(acc * k)
        n += k
    model.train()
    return {"loss": loss_sum / max(n, 1), "accuracy": correct / max(n, 1),
            "labeled_positions": n}


def run(a) -> dict:
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)

    panels = build_panels(vocab_size=a.vocab_size, seq_len=a.seq_len,
                          num_kv_pairs=a.num_kv_pairs, num_examples=a.num_examples,
                          seed=a.seed, power_a=a.power_a)
    if a.eval_examples and a.eval_examples > 0:
        for name in ("valid", "test"):
            p = panels[name]
            if p.inputs.shape[0] > a.eval_examples:
                # Subset for evaluation speed only; the recorded SHA-256 still refers to the
                # full generated panel, which both arms share.
                panels[name] = MQARBatch(p.inputs[:a.eval_examples], p.labels[:a.eval_examples],
                                         p.spec, p.sha256, p.alignment)

    cfg = default_config(a)
    model = FairLM(cfg).to(device)
    counts = count_mixer_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    gen = torch.Generator().manual_seed(a.seed + 1)

    a.output.mkdir(parents=True, exist_ok=True)
    initial = {name: evaluate(model, panel, a.eval_batch_size, device)
               for name, panel in panels.items() if name != "train"}

    start = time.perf_counter()
    history = []
    with (a.output / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for step in range(1, a.steps + 1):
            model.train()
            if a.online_train:
                # New KV pairings every step. The training stream is deterministic
                # across arms, independent of the fixed valid/test panels.
                online_spec = replace(
                    panels["train"].spec,
                    num_examples=a.batch_size,
                    seed=a.seed + a.online_seed_offset + step,
                )
                online_batch = build_segment(online_spec)
                x = online_batch.inputs.to(device)
                y = online_batch.labels.to(device)
            else:
                idx = torch.randint(
                    panels["train"].inputs.shape[0], (a.batch_size,), generator=gen
                )
                x = panels["train"].inputs[idx].to(device)
                y = panels["train"].labels[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = mqar_loss(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip,
                                                       error_if_nonfinite=True)
            optimizer.step()
            acc, _ = mqar_accuracy(logits.detach(), y)
            rec = {"step": step, "train_loss": float(loss.detach()), "train_accuracy": acc,
                   "grad_norm": float(grad_norm), "elapsed_seconds": time.perf_counter() - start}
            if step == 1 or step % a.eval_every == 0 or step == a.steps:
                rec["valid"] = evaluate(model, panels["valid"], a.eval_batch_size, device)
                rec["test"] = evaluate(model, panels["test"], a.eval_batch_size, device)
            history.append(rec)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step == 1 or step % a.log_every == 0 or step == a.steps:
                print(json.dumps({k: v for k, v in rec.items() if k != "elapsed_seconds"}),
                      flush=True)
    wall = time.perf_counter() - start
    final = {name: evaluate(model, panel, a.eval_batch_size, device)
             for name, panel in panels.items() if name != "train"}

    report = {
        "mixer": a.mixer,
        "scope": "fair paired MQAR pilot; identical backbone, data, optimiser and budget",
        "config": asdict(cfg),
        "parameters": counts,
        "panels": {name: {"sha256": p.sha256, "examples": int(p.inputs.shape[0]),
                          "alignment": p.alignment} for name, p in panels.items()},
        "initial": initial,
        "final": final,
        "training": {
            "steps": a.steps, "batch_size": a.batch_size, "lr": a.lr,
            "online_train": a.online_train,
            "online_seed_offset": a.online_seed_offset if a.online_train else None,
            "online_examples_generated": a.steps * a.batch_size
                if a.online_train else None,
            "wall_seconds": wall, "seconds_per_step": wall / max(a.steps, 1),
        },
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": str(device), "threads": a.threads},
        "history_tail": history[-3:],
    }
    (a.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save({"config": asdict(cfg), "model": model.state_dict(), "step": a.steps},
               a.output / "checkpoint.pt")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    a = p.parse_args()
    report = run(a)
    print(json.dumps({"mixer": report["mixer"], "parameters": report["parameters"],
                      "final": report["final"], "wall_seconds": report["training"]["wall_seconds"]}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
