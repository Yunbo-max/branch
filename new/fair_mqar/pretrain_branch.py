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
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

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

    def __init__(self, root: Path, seq_len: int) -> None:
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
        self.lengths = np.array([m.shape[0] for m in self.maps], dtype=np.int64)
        self.cum = np.cumsum(self.lengths)

    def __len__(self) -> int:
        return int(self.cum[-1])

    def gather(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        shard = np.searchsorted(self.cum, idx, side="right")
        within = idx - np.where(shard > 0, self.cum[shard - 1], 0)
        rows = np.stack(
            [self.maps[s][w][: self.seq_len] for s, w in zip(shard, within)]
        )
        return rows.astype(np.int64)

    def sample(self, n: int, gen: torch.Generator) -> np.ndarray:
        idx = torch.randint(len(self), (n,), generator=gen).numpy()
        return self.gather(idx)

    def sample_split(self, n: int, gen: torch.Generator, split: RowSplit) -> np.ndarray:
        idx = split.train_indices(n, gen, len(self))
        return self.gather(idx)


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
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=10000,
        help="write checkpoint_<step>.pt every N steps (0 disables)",
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
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--max-states", type=int, default=8)
    p.add_argument("--chunk-len", type=int, default=64)
    p.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="recompute each chunk in backward; cuts activation memory",
    )
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
    a = p.parse_args()

    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)

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
            anchor_interval=a.anchor_interval,
            max_anchors=8,
            read_topk=a.read_topk,
            d_route=a.d_route,
            max_states=a.max_states,
            protect=a.protect,
            fork_cost=a.fork_cost,
            covariance=a.covariance,
            differentiate_covariance=a.differentiate_covariance,
            solver_refresh=a.solver_refresh,
            chunk_len=chunk_len,
            grad_checkpoint=a.grad_checkpoint,
        )
        return FairLM(cfg).to(device), cfg

    gen = torch.Generator().manual_seed(a.seed + 1)
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
    if mixer_cfg is not None:
        print(
            "[mixer-config] " + json.dumps(asdict(mixer_cfg), default=str), flush=True
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=a.lr, weight_decay=a.weight_decay
    )

    a.output.mkdir(parents=True, exist_ok=True)

    panel_ids = panel.tensor(device)
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

    start = time.perf_counter()
    history = []
    with (a.output / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for step in range(1, a.steps + 1):
            if (
                a.time_budget_seconds
                and time.perf_counter() - start > a.time_budget_seconds
            ):
                print("time budget reached", flush=True)
                break
            model.train()
            ids = torch.from_numpy(corpus.sample_split(a.batch_size, gen, split)).to(
                device
            )
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                model.parameters(), a.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            rec = {
                "step": step,
                "train_loss": float(loss.detach()),
                "grad_norm": float(gn),
                "elapsed_seconds": time.perf_counter() - start,
            }
            if step == 1 or step % a.eval_every == 0 or step == a.steps:
                rec["eval_loss"] = evaluate()
            history.append(rec)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if a.checkpoint_every and step % a.checkpoint_every == 0:
                torch.save(
                    {
                        "config": asdict(cfg),
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                    },
                    a.output / f"checkpoint_{step}.pt",
                )
                print(f"[ckpt] wrote checkpoint_{step}.pt", flush=True)
            if step == 1 or step % a.log_every == 0 or step == a.steps:
                print(json.dumps(rec), flush=True)

    wall = time.perf_counter() - start
    report = {
        "scope": "pretraining pilot on the packed Long-Data-Collections corpus",
        "mixer": a.mixer,
        "config": asdict(cfg),
        "parameters": sum(q.numel() for q in model.parameters()),
        "corpus": {
            "shards": len(corpus.files),
            "sequences": len(corpus),
            "seq_len": a.seq_len,
            "tokens_available": len(corpus) * a.seq_len,
        },
        "split": split_manifest(split, corpus, panel, a.seq_len),
        "training": {
            "steps": len(history),
            "batch_size": a.batch_size,
            "tokens_per_step": a.batch_size * a.seq_len,
            "wall_seconds": wall,
            "seconds_per_step": wall / max(len(history), 1),
        },
        "final": history[-1] if history else None,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "threads": a.threads,
        },
    }
    (a.output / "report.json").write_text(json.dumps(report, indent=2, default=str))
    torch.save(
        {"config": asdict(cfg), "model": model.state_dict(), "step": len(history)},
        a.output / "checkpoint.pt",
    )
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
