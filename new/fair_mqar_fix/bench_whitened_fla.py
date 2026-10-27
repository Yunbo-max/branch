"""H200 comparison of original PyTorch Full-C Delta and FLA-whitened Delta.

Benchmarks ONLY one frozen geometry chunk (including original Cholesky and
the basis transforms), not whole-language-model training. Requires CUDA+FLA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F

from chunked_delta import chunked_delta_from_directions
from experimental_whitened_fla import geo_via_fla_delta


def original_torch(state, k, v, q, H):
    L = torch.linalg.cholesky(H)
    u = torch.cholesky_solve(k.permute(0, 2, 3, 1), L).permute(0, 3, 1, 2)
    a = u / (1 + (k * u).sum(-1, keepdim=True))
    return chunked_delta_from_directions(
        state, k, v, q, a, return_states=True
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", choices=["cuda"])
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--d-key", type=int, default=256)
    p.add_argument("--d-value", type=int, default=256)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--repeats", type=int, default=12)
    p.add_argument("--include-backward", action="store_true")
    p.add_argument("--output", type=Path,
                   default=Path("results/bench_whitened_fla.json"))
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("requires CUDA GPU with installed FLA")
    import fla.ops.delta_rule  # noqa: F401
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    torch.manual_seed(37)
    d, dv, T, h = a.d_key, a.d_value, a.chunk, a.heads
    S = torch.randn(1, h, d, dv, device="cuda") * .01
    K = F.normalize(torch.randn(1, T, h, d, device="cuda"), dim=-1)
    Q = F.normalize(torch.randn(1, T, h, d, device="cuda"), dim=-1)
    V = torch.randn(1, T, h, dv, device="cuda").tanh()
    U = torch.randn(1, h, d, 48, device="cuda")
    H = torch.eye(d, device="cuda") + .005 * (U @ U.transpose(-1, -2))
    for x in (S, K, Q, V, U):
        x.requires_grad_(a.include_backward)

    def fla_run():
        return geo_via_fla_delta(S, K, V, Q, H, chunk_size=64)

    def native_gdn_run():
        # A speed control, NOT an equivalent mathematical model.
        # Scalar log-decay g<0 and the same q/k/v/state/head layout.
        return chunk_gated_delta_rule(
            q=Q.to(torch.bfloat16), k=K.to(torch.bfloat16),
            v=V.to(torch.bfloat16),
            g=torch.full((1, T, h), -0.01, device="cuda", dtype=torch.bfloat16),
            beta=torch.full((1, T, h), 0.5, device="cuda", dtype=torch.bfloat16),
            scale=1.0, initial_state=S,
            output_final_state=True, chunk_size=64,
            use_qk_l2norm_in_kernel=False,
        )

    methods = [("original_pytorch", lambda: original_torch(S, K, V, Q, H)),
               ("whitened_fla", fla_run),
               ("native_fla_gdn", native_gdn_run)]
    records = []
    for name, fn in methods:
        for _ in range(a.warmup):
            out, final = fn()
            if a.include_backward:
                (out.float().square().mean() + final.square().mean()).backward()
        torch.cuda.synchronize()
        times = []
        for _ in range(a.repeats):
            if a.include_backward:
                for x in (S, K, Q, V, U):
                    x.grad = None
            start = time.perf_counter()
            out, final = fn()
            if a.include_backward:
                (out.float().square().mean() + final.square().mean()).backward()
            torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1e3)
        records.append({"method": name, "median_ms": statistics.median(times),
                        "min_ms": min(times)})
        print(records[-1], flush=True)

    with torch.no_grad():
        ref_o, ref_s = methods[0][1]()
        fla_o, fla_s = methods[1][1]()
        accuracy = {
            "mean_output_abs": float((ref_o.float()-fla_o.float()).abs().mean()),
            "max_output_abs": float((ref_o.float()-fla_o.float()).abs().max()),
            "mean_state_abs": float((ref_s.float()-fla_s.float()).abs().mean()),
            "max_state_abs": float((ref_s.float()-fla_s.float()).abs().max()),
        }
    result = {
        "hardware": torch.cuda.get_device_name(), "settings": {
            "heads": h, "d_key": d, "d_value": dv, "chunk": T,
            "backward": a.include_backward,
        },
        "records": records, "errors": accuracy,
        "note": "isolated frozen-C block; native FLA GDN is runtime control, not an equivalent objective or full 0.79B LM",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    print("wrote", a.output)


if __name__ == "__main__":
    main()
