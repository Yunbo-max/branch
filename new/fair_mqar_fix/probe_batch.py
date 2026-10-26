#!/usr/bin/env python3
"""Measure peak GPU memory and step time for several batch sizes at 16K.

Builds the exact pretraining model, runs two steps per batch size, and prints
`torch.cuda.max_memory_allocated` so the batch-size decision is measured, not guessed.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from fair_models import FairLM, FairLMConfig  # noqa: E402
from pretrain_branch import PackedCorpus  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--packed-dir", type=Path, required=True)
    p.add_argument("--seq-len", type=int, default=16384)
    p.add_argument("--batches", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--mixer", default="chunked")
    p.add_argument("--chunk-len", type=int, default=256)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d-key", type=int, default=64)
    p.add_argument("--d-value", type=int, default=64)
    p.add_argument("--d-route", type=int, default=32)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--max-states", type=int, default=8)
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="match the training path; cuts activation memory")
    p.add_argument("--seed", type=int, default=17)
    a = p.parse_args()

    device = torch.device("cuda")
    torch.set_num_threads(8)
    corpus = PackedCorpus(a.packed_dir, a.seq_len)
    gen = torch.Generator().manual_seed(a.seed)
    total = torch.cuda.get_device_properties(0).total_memory
    rows = []

    for bs in a.batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cfg = FairLMConfig(mixer=a.mixer, vocab_size=32000, d_model=a.d_model,
                           n_layers=a.layers, n_heads=a.heads, d_key=a.d_key,
                           d_value=a.d_value, ff_mult=a.ff_mult, chunk_len=a.chunk_len,
                           max_states=a.max_states, grad_checkpoint=a.grad_checkpoint)
        try:
            model = FairLM(cfg).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
            ids = torch.from_numpy(corpus.sample(bs, gen)).to(device)
            for _ in range(2):                      # warmup + measured step
                t0 = time.perf_counter()
                logits, _ = model(ids)
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                       ids[:, 1:].reshape(-1))
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                step_s = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated()
            rec = {"batch": bs, "peak_allocated_gib": round(peak / 2**30, 2),
                   "total_gib": round(total / 2**30, 2),
                   "utilization_pct": round(peak / total * 100, 1),
                   "seconds_per_step": round(step_s, 3),
                   "tokens_per_second": round(bs * a.seq_len / step_s, 0)}
            print(json.dumps(rec), flush=True)
            rows.append(rec)
            del model, opt, ids, logits, loss
            gc.collect(); torch.cuda.empty_cache()
        except Exception as exc:
            print(json.dumps({"batch": bs, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}),
                  flush=True)
            rows.append({"batch": bs, "error": type(exc).__name__})
            gc.collect(); torch.cuda.empty_cache()

    Path("/tmp/probe_batch.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
