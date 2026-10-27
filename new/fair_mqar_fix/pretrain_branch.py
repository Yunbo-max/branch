#!/usr/bin/env python3
"""Pretrain on the packed 16K corpus with the fair backbone.

Reads the ``.npy`` shards produced by ``new/MARCH/preprocess/pack_pretrain.py``
(shape ``(n_sequences, packed_len)``, uint16), samples rows, and trains with the standard
next-token objective. The mixer is chosen with ``--mixer`` (``chunked`` for the chunked
Branch Memory, ``branch_cached``/``branch`` for the per-token variants).

    python pretrain_branch.py --packed-dir ../MARCH/data/packed_16k \
      --mixer chunked --chunk-len 64 --seq-len 16384 --batch-size 8 --steps 2000 \
      --d-model 256 --layers 2 --heads 4 --d-key 64 --d-value 64 --d-route 32 \
      --output results/pretrain_chunked
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from fair_models import MIXERS, FairLM, FairLMConfig  # noqa: E402
from data_split import (  # noqa: E402
    RowSplit,
    build_panel,
    manifest as split_manifest,
    write_manifest,
)


class PackedCorpus:
    """Random access over memmapped shards; no shard is fully loaded into RAM."""

    def __init__(self, root: Path, seq_len: int, *, random_crop: bool = False) -> None:
        self.files = sorted(root.glob("*.npy"))
        if not self.files:
            raise SystemExit(f"no .npy shards under {root}")
        self.maps = [np.load(f, mmap_mode="r") for f in self.files]
        self.packed_len = int(self.maps[0].shape[1])
        if seq_len > self.packed_len:
            raise SystemExit(
                f"seq_len {seq_len} exceeds packed length {self.packed_len}"
            )
        self.seq_len = seq_len
        self.random_crop = random_crop
        self.lengths = np.array([m.shape[0] for m in self.maps], dtype=np.int64)
        self.cum = np.cumsum(self.lengths)

    def __len__(self) -> int:
        return int(self.cum[-1])

    def gather(self, idx: np.ndarray, *, offsets: np.ndarray | None = None) -> np.ndarray:
        """Read packed sub-sequences; validation uses a deterministic prefix."""
        idx = np.asarray(idx, dtype=np.int64)
        shard = np.searchsorted(self.cum, idx, side="right")
        within = idx - np.where(shard > 0, self.cum[shard - 1], 0)
        if offsets is None:
            offsets = np.zeros(len(idx), dtype=np.int64)
        else:
            offsets = np.asarray(offsets, dtype=np.int64)
            if offsets.shape != idx.shape or np.any(offsets < 0) or np.any(
                offsets + self.seq_len > self.packed_len
            ):
                raise ValueError("invalid packed-row crop offsets")
        rows = np.stack(
            [self.maps[sh][row][off:off + self.seq_len]
             for sh, row, off in zip(shard, within, offsets)]
        )
        return rows.astype(np.int64)

    def _offsets(self, n: int, gen: torch.Generator) -> np.ndarray | None:
        if not self.random_crop or self.seq_len == self.packed_len:
            return None
        return torch.randint(
            0, self.packed_len - self.seq_len + 1, (n,), generator=gen
        ).numpy()

    def sample(self, n: int, gen: torch.Generator) -> np.ndarray:
        idx = torch.randint(len(self), (n,), generator=gen).numpy()
        return self.gather(idx, offsets=self._offsets(n, gen))

    def sample_split(self, n: int, gen: torch.Generator, split: RowSplit) -> np.ndarray:
        idx = split.train_indices(n, gen, len(self))
        return self.gather(idx, offsets=self._offsets(n, gen))


def grad_sync_context(model, *, ddp: bool, micro_idx: int, grad_accum: int):
    """Reduce gradients only in the last microbatch; wrap forward AND backward."""
    if ddp and micro_idx < grad_accum - 1:
        return model.no_sync()
    return nullcontext()

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--packed-dir", type=Path, required=True)
    p.add_argument("--seq-len", type=int, default=16384)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--split-seed",
        type=int,
        default=12345,
        help="seed for the deterministic row-level train/val/test split",
    )
    p.add_argument(
        "--val-per-mille",
        type=int,
        default=20,
        help="validation rows per 1000 (row-level hash split)",
    )
    p.add_argument(
        "--test-per-mille",
        type=int,
        default=10,
        help="held-out test rows per 1000 (never used for evaluation here)",
    )
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--amp", choices=["none", "bf16"], default="none",
                   help="bf16 autocast for the LM forward/backward")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="micro-batches accumulated per optimizer step")
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                   help="allow TF32 matmuls in FP32 training")
    p.add_argument("--ddp-timeout-min", type=int, default=60)
    p.add_argument("--lr-schedule", choices=["constant", "wsd"], default="constant")
    p.add_argument("--warmup-frac", type=float, default=0.01)
    p.add_argument("--decay-frac", type=float, default=0.2)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=10000,
        help="write checkpoint_<step>.pt every N steps (0 disables)",
    )
    p.add_argument(
        "--keep-checkpoints",
        choices=["last", "all"],
        default="last",
        help=(
            "keep only the newest numbered checkpoint (default), or keep every "
            "checkpoint; at the end only checkpoint.pt remains in 'last' mode"
        ),
    )
    p.add_argument(
        "--auto-chunk",
        type=int,
        nargs="+",
        default=None,
        help="probe these chunk_len values, keep the fastest that fits, then train",
    )
    p.add_argument("--time-budget-seconds", type=float, default=0.0)
    # model / mixer
    p.add_argument("--mixer", choices=list(MIXERS), default="chunked")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d-key", type=int, default=64)
    p.add_argument("--d-value", type=int, default=64)
    p.add_argument("--d-route", type=int, default=32)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--ff-extra", type=int, default=0,
                   help="extra FFN hidden units per layer for parameter matching")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--max-states", type=int, default=8)
    p.add_argument("--chunk-len", type=int, default=64)
    p.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="recompute each chunk in backward; cuts activation memory",
    )
    p.add_argument("--geometry-mode", choices=["isotropic", "rayleigh", "full", "certified"], default="full")
    p.add_argument("--geometry-threshold", type=float, default=0.1)
    p.add_argument("--read-mode", choices=["chunk_start", "current_only"], default="chunk_start")
    p.add_argument("--anchor-interval", type=int, default=512)
    p.add_argument("--read-topk", type=int, default=4)
    p.add_argument("--solver-refresh", type=int, default=8)
    p.add_argument("--protect", type=float, default=8.0)
    p.add_argument("--fork-cost", type=float, default=0.02)
    p.add_argument("--covariance", choices=["full", "diagonal"], default="full")
    p.add_argument(
        "--differentiate-covariance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="let gradients flow through the protection matrix (default: on)",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "resume from checkpoint_<step>.pt or checkpoint.pt: restores model, "
            "optimizer, step, RNG states and the data generator state; --steps is "
            "the total number of steps to reach"
        ),
    )
    a = p.parse_args()

    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if a.grad_accum < 1:
        raise SystemExit("--grad-accum must be >= 1")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp = world_size > 1
    if ddp:
        if a.device != "cuda":
            raise SystemExit("DDP requires --device cuda")
        dist.init_process_group(
            "nccl", timeout=timedelta(minutes=a.ddp_timeout_min)
        )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(a.device)
    is_main = rank == 0

    torch.set_num_threads(a.threads)
    # Identical initialization on every rank; the data generator is rank-specific.
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    if device.type == "cuda" and a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    corpus = PackedCorpus(a.packed_dir, a.seq_len)
    print(
        f"corpus: {len(corpus):,} sequences x {a.seq_len} tokens "
        f"= {len(corpus) * a.seq_len / 1e9:.2f} B tokens available",
        flush=True,
    )

    def build(chunk_len: int):
        cfg = FairLMConfig(
            mixer=a.mixer,
            vocab_size=a.vocab_size,
            d_model=a.d_model,
            n_layers=a.layers,
            n_heads=a.heads,
            d_key=a.d_key,
            d_value=a.d_value,
            ff_mult=a.ff_mult,
            ff_extra=a.ff_extra,
            anchor_interval=a.anchor_interval,
            max_anchors=8,
            read_topk=a.read_topk,
            d_route=a.d_route,
            max_states=a.max_states,
            protect=a.protect,
            fork_cost=a.fork_cost,
            covariance=a.covariance,
            differentiate_covariance=a.differentiate_covariance,
            geometry_mode=a.geometry_mode,
            geometry_threshold=a.geometry_threshold,
            read_mode=a.read_mode,
            solver_refresh=a.solver_refresh,
            chunk_len=chunk_len,
            grad_checkpoint=a.grad_checkpoint,
        )
        return FairLM(cfg).to(device), cfg

    gen = torch.Generator().manual_seed(a.seed + 1 + rank)
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

    if a.auto_chunk:
        probe_rows = []
        for cl in a.auto_chunk:
            try:
                m, _ = build(cl)
                opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
                probe_gen = torch.Generator().manual_seed(a.seed + 5)
                ids = torch.from_numpy(
                    corpus.sample_split(a.batch_size, probe_gen, split)
                ).to(device)
                t0 = time.perf_counter()
                logits, _ = m(ids)
                F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
                ).backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                elapsed = time.perf_counter() - t0
                probe_rows.append({"chunk_len": cl, "seconds_per_step": elapsed})
                print(f"[auto-chunk] {cl}: {elapsed:.3f} s/step", flush=True)
                del m, opt, logits, ids
                torch.cuda.empty_cache()
            except Exception as exc:  # OOM, MAGMA assert, or any backend failure
                print(
                    f"[auto-chunk] {cl}: {type(exc).__name__}: {str(exc)[:120]}",
                    flush=True,
                )
                import gc as _gc

                _gc.collect()
                torch.cuda.empty_cache()
                probe_rows.append({"chunk_len": cl, "seconds_per_step": None})
        usable = [r for r in probe_rows if r["seconds_per_step"]]
        if not usable:
            raise SystemExit("no chunk size fit in memory")
        best = min(usable, key=lambda r: r["seconds_per_step"])
        a.chunk_len = best["chunk_len"]
        print(
            f"[auto-chunk] selected chunk_len={a.chunk_len} "
            f"({best['seconds_per_step']:.3f} s/step)",
            flush=True,
        )
        del probe_rows, usable, best

    model, cfg = build(a.chunk_len)
    mixer_cfg = getattr(model.blocks[0].mixer, "config", None)
    if is_main and mixer_cfg is not None:
        print(
            "[mixer-config] " + json.dumps(asdict(mixer_cfg), default=str), flush=True
        )
    if ddp:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if ddp else model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=a.lr, weight_decay=a.weight_decay
    )

    if is_main:
        a.output.mkdir(parents=True, exist_ok=True)
    if ddp:
        dist.barrier()

    panel_ids = panel.tensor(device)
    if is_main:
        write_manifest(
            a.output / "split_manifest.json",
            split_manifest(split, corpus, panel, a.seq_len),
        )
        split_counts = split.counts(len(corpus))
        print(
            f"[split] train={split_counts['train']:,} "
            f"val={split_counts['val']:,} "
            f"test={split_counts['test']:,} | "
            f"panel={len(panel.indices)} rows sha256={panel.tokens_sha256[:12]}",
            flush=True,
        )

    def save_checkpoint(path: Path, step: int, elapsed: float) -> None:
        local_rng = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        local_cuda = (
            torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        )
        if ddp:
            gen_states = [None] * world_size
            rng_states = [None] * world_size
            cuda_states = [None] * world_size
            dist.all_gather_object(gen_states, gen.get_state())
            dist.all_gather_object(rng_states, local_rng)
            dist.all_gather_object(cuda_states, local_cuda)
        else:
            gen_states = [gen.get_state()]
            rng_states = [local_rng]
            cuda_states = [local_cuda]
        if not is_main:
            return
        payload = {
            "config": asdict(cfg),
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "elapsed_seconds": elapsed,
            "gen_state": gen_states[0],
            "gen_states": gen_states,
            "rng_states": rng_states,
            "cuda_rng_states": cuda_states,
            "torch_rng_state": rng_states[0]["torch"],
            "numpy_rng_state": rng_states[0]["numpy"],
            "python_rng_state": rng_states[0]["python"],
            "split": {
                "seed": split.seed,
                "val_per_mille": split.val_per_mille,
                "test_per_mille": split.test_per_mille,
            },
            "panel_sha256": panel.tokens_sha256,
        }
        torch.save(payload, path)

    def evaluate() -> float:
        model.eval()
        total = 0.0
        n_batches = 0
        with torch.no_grad():
            for start in range(0, len(panel_ids), a.batch_size):
                ids = panel_ids[start : start + a.batch_size]
                if ids.shape[0] == 0:
                    continue
                logits, _ = model(ids)
                total += float(
                    F.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.shape[-1]),
                        ids[:, 1:].reshape(-1),
                    )
                )
                n_batches += 1
        model.train()
        return total / max(n_batches, 1)

    start_step = 0
    prior_elapsed = 0.0
    history = []
    resume_info = None
    if a.resume is not None:
        ckpt = torch.load(a.resume, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        prior_elapsed = float(ckpt.get("elapsed_seconds", 0.0))
        if "gen_states" in ckpt:
            gen.set_state(ckpt["gen_states"][rank])
        elif "gen_state" in ckpt:
            gen.set_state(ckpt["gen_state"])
        if "rng_states" in ckpt:
            rng = ckpt["rng_states"][rank]
            torch.set_rng_state(rng["torch"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        else:
            if "torch_rng_state" in ckpt:
                torch.set_rng_state(ckpt["torch_rng_state"])
            if "numpy_rng_state" in ckpt:
                np.random.set_state(ckpt["numpy_rng_state"])
            if "python_rng_state" in ckpt:
                random.setstate(ckpt["python_rng_state"])
        if device.type == "cuda" and "cuda_rng_states" in ckpt:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_states"][rank])
        elif device.type == "cuda" and "cuda_rng_state" in ckpt:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        metrics_path = a.output / "metrics.jsonl"
        if is_main and metrics_path.exists():
            old = [
                json.loads(line)
                for line in metrics_path.read_text().splitlines()
                if line.strip()
            ]
            history = [r for r in old if int(r.get("step", 0)) <= start_step]
            metrics_path.write_text("".join(json.dumps(r) + "\n" for r in history))
        resume_info = {
            "from": str(a.resume),
            "start_step": start_step,
            "data_stream_continuous": "gen_state" in ckpt,
            "optimizer_restored": "optimizer" in ckpt,
            "rng_restored": all(
                key in ckpt
                for key in (
                    "torch_rng_state",
                    "numpy_rng_state",
                    "python_rng_state",
                )
            ),
        }
        if is_main:
            print("[resume] " + json.dumps(resume_info), flush=True)
        if ddp:
            dist.barrier()

    def learning_rate_at(step: int) -> float:
        if a.lr_schedule == "constant":
            return a.lr
        warm = max(1, int(a.steps * a.warmup_frac))
        decay_start = max(warm, int(a.steps * (1.0 - a.decay_frac)))
        if step <= warm:
            return a.lr * step / warm
        if step <= decay_start:
            return a.lr
        frac = (step - decay_start) / max(1, a.steps - decay_start)
        return a.lr * (1.0 - (1.0 - a.min_lr_ratio) * frac)

    start = time.perf_counter()
    mode = "a" if a.resume is not None else "w"
    log_path = (a.output / "metrics.jsonl") if is_main else Path(os.devnull)
    with log_path.open(mode, encoding="utf-8") as log:
        for step in range(start_step + 1, a.steps + 1):
            reached = bool(
                a.time_budget_seconds
                and prior_elapsed + time.perf_counter() - start
                > a.time_budget_seconds
            )
            if ddp:
                flag = torch.tensor([int(reached)], device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                reached = bool(flag.item())
            if reached:
                if is_main:
                    print("time budget reached", flush=True)
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            micro_losses = []
            for micro_idx in range(a.grad_accum):
                ids = torch.from_numpy(
                    corpus.sample_split(a.batch_size, gen, split)
                ).to(device)
                with grad_sync_context(
                    model, ddp=ddp, micro_idx=micro_idx, grad_accum=a.grad_accum
                ):
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=(a.amp == "bf16" and device.type == "cuda"),
                    ):
                        logits, _ = model(ids)
                        loss = F.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.shape[-1]),
                            ids[:, 1:].reshape(-1),
                        )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss at step {step}")
                    (loss / a.grad_accum).backward()
                micro_losses.append(float(loss.detach()))
            lr_now = learning_rate_at(step)
            for group in optimizer.param_groups:
                group["lr"] = lr_now
            gn = torch.nn.utils.clip_grad_norm_(
                model.parameters(), a.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            tokens_seen = (
                step
                * a.batch_size
                * a.seq_len
                * world_size
                * a.grad_accum
            )
            rec = {
                "step": step,
                "train_loss": float(sum(micro_losses) / len(micro_losses)),
                "grad_norm": float(gn),
                "lr": lr_now,
                "tokens_seen": tokens_seen,
                "elapsed_seconds": prior_elapsed + time.perf_counter() - start,
            }
            if step == 1 or step % a.eval_every == 0 or step == a.steps:
                rec["eval_loss"] = evaluate()
            if is_main:
                history.append(rec)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if a.checkpoint_every and step % a.checkpoint_every == 0:
                current = a.output / f"checkpoint_{step}.pt"
                save_checkpoint(current, step, rec["elapsed_seconds"])
                if is_main and a.keep_checkpoints == "last":
                    for old in a.output.glob("checkpoint_*.pt"):
                        if old != current:
                            old.unlink()
                if is_main:
                    print(f"[ckpt] wrote {current.name}", flush=True)
                if ddp:
                    dist.barrier()
            if is_main and (
                step == 1 or step % a.log_every == 0 or step == a.steps
            ):
                print(json.dumps(rec), flush=True)

    wall = time.perf_counter() - start
    total_wall = prior_elapsed + wall
    report = {
        "scope": "pretraining pilot on the packed Long-Data-Collections corpus",
        "mixer": a.mixer,
        "config": asdict(cfg),
        "parameters": sum(q.numel() for q in raw_model.parameters()),
        "corpus": {
            "shards": len(corpus.files),
            "sequences": len(corpus),
            "seq_len": a.seq_len,
            "tokens_available": len(corpus) * a.seq_len,
        },
        "split": split_manifest(split, corpus, panel, a.seq_len),
        "resume": resume_info,
        "training": {
            "steps": len(history),
            "batch_size": a.batch_size,
            "tokens_per_step": a.batch_size * a.seq_len,
            "segment_wall_seconds": wall,
            "total_wall_seconds": total_wall,
            "seconds_per_step": total_wall / max(len(history), 1),
        },
        "final": history[-1] if history else None,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "threads": a.threads,
        },
    }
    if is_main:
        (a.output / "report.json").write_text(
            json.dumps(report, indent=2, default=str)
        )
    save_checkpoint(
        a.output / "checkpoint.pt",
        len(history),
        history[-1]["elapsed_seconds"] if history else total_wall,
    )
    if ddp:
        dist.barrier()
    if a.keep_checkpoints == "last":
        for old in a.output.glob("checkpoint_*.pt"):
            old.unlink()
    print(
        json.dumps(
            {
                "steps": len(history),
                "final": history[-1] if history else None,
                "seconds_per_step": report["training"]["seconds_per_step"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
