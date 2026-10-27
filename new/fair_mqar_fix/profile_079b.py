"""One-GPU, model-scale preflight for 0.79B GeoDelta/Delta proxy variants.

This measures the actual PyTorch implementation before committing 8xH200 time.
It does not assert any training-quality result. For operator costs, use --profile.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fair_models import FairLM, FairLMConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("single_full", "isotropic", "rayleigh", "delta_proxy", "march_proxy"),
                   default="single_full")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--chunk-len", type=int, default=256)
    p.add_argument("--delta-tile-len", type=int, default=0)
    p.add_argument("--d-model", type=int, default=1536)
    p.add_argument("--layers", type=int, default=21)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--d-key", type=int, default=256)
    p.add_argument("--d-value", type=int, default=256)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--amp", choices=("bf16", "none"), default="bf16")
    p.add_argument("--protection-decay", type=float, default=0.95)
    p.add_argument("--differentiate-covariance", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--trace-dir", type=Path, default=None,
                   help="optional one-step PyTorch operator profiler trace directory")
    p.add_argument("--output", type=Path, default=Path("results/scale_0p79b/profile.json"))
    a = p.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("No CUDA device found; H200 profiling must run on GPU")
    if a.seq_len < 1 or a.chunk_len < 1 or a.repeats < 1 or a.warmup < 0:
        raise SystemExit("sequence/chunk/repeats must be positive")
    if a.delta_tile_len < 0 or (a.delta_tile_len and a.delta_tile_len > a.chunk_len):
        raise SystemExit("invalid delta_tile_len")
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = (a.device == "cuda")
    kw = {
        "vocab_size": 32000 if a.d_model > 64 else 128,
        "d_model": a.d_model, "n_layers": a.layers, "n_heads": a.heads,
        "d_key": a.d_key, "d_value": a.d_value, "d_route": 64 if a.d_model > 64 else 8,
        "ff_mult": 4, "ff_extra": 1280 if a.d_model > 64 else 0,
        "chunk_len": a.chunk_len, "delta_tile_len": a.delta_tile_len,
        "grad_checkpoint": True, "max_states": 1,
        "protection_decay": a.protection_decay,
        "differentiate_covariance": a.differentiate_covariance,
    }
    if a.arm == "march_proxy":
        kw.update(mixer="march_chunked", max_states=64, ff_extra=1088 if a.d_model > 64 else 0,
                  read_topk=4)
    else:
        kw.update(mixer="single_geo", geometry_mode="isotropic" if a.arm == "delta_proxy" else
                  a.arm.replace("single_", ""), read_mode="current_only"
                  if a.arm == "delta_proxy" else "chunk_start")
    cfg = FairLMConfig(**kw)
    device = torch.device(a.device)
    model = FairLM(cfg).to(device).train()
    param_count = sum(x.numel() for x in model.parameters())
    ids = torch.randint(cfg.vocab_size, (1, a.seq_len), dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=(a.device == "cuda"))
    amp_enabled = a.amp == "bf16" and a.device == "cuda"

    def synchronized_time():
        if a.device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    def one_step():
        optimizer.zero_grad(set_to_none=True)
        before = synchronized_time()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            logits, _ = model(ids)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                   ids[:, 1:].reshape(-1))
        forward_done = synchronized_time()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss")
        loss.backward()
        backward_done = synchronized_time()
        optimizer.step()
        step_done = synchronized_time()
        return dict(forward_sec=forward_done-before, backward_sec=backward_done-forward_done,
                    optimizer_sec=step_done-backward_done, total_sec=step_done-before,
                    loss=float(loss.detach()))

    rows = []
    for it in range(a.warmup + a.repeats):
        if a.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        row = one_step()
        if it >= a.warmup:
            if a.device == "cuda":
                row["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
            rows.append(row)
            print(f"[{a.arm}] measured={it-a.warmup+1}/{a.repeats}: {row}", flush=True)

    if a.trace_dir is not None:
        a.trace_dir.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if a.device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities, record_shapes=True, profile_memory=True,
            with_stack=False,
        ) as prof:
            one_step()
        print(prof.key_averages().table(sort_by="self_cuda_time_total"
              if a.device == "cuda" else "self_cpu_time_total", row_limit=35))
        prof.export_chrome_trace(str(a.trace_dir / f"{a.arm}_trace.json"))

    result = {
        "config": {**vars(a), "output": str(a.output), "trace_dir":
                   None if a.trace_dir is None else str(a.trace_dir)},
        "parameters": param_count, "runs": rows,
        "median_total_sec": statistics.median(z["total_sec"] for z in rows),
        "median_forward_sec": statistics.median(z["forward_sec"] for z in rows),
        "median_backward_sec": statistics.median(z["backward_sec"] for z in rows),
        "median_optimizer_sec": statistics.median(z["optimizer_sec"] for z in rows),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: result[k] for k in ("parameters", "median_total_sec",
                                            "median_forward_sec", "median_backward_sec")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
