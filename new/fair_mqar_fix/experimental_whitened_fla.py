"""EXPERIMENTAL bridge from exact Full-C GeoDelta to stock FLA DeltaNet kernel.

The math is exact in real arithmetic: transform state by L^T and q/k by
L^-1, with L=Cholesky(ridge I+protect C); run *ordinary* Delta Rule with
beta = 1/(tau+||L^-1 k||^2), then transform the final state back.

This adapter quantizes q/k/v into BF16 because FLA's fused chunk Delta
kernel does not accept FP32 inputs. Thus actual execution is NOT
bit-exact relative to FP32 Full-C, and speed has not been tested on H200.
"""
from __future__ import annotations

import torch
from torch import Tensor

from whitened_delta_reference import geo_reference


def geo_via_fla_delta(
    state: Tensor, k: Tensor, v: Tensor, q: Tensor, H: Tensor, *,
    tau: float = 1.0, chunk_size: int = 64,
    log_gate: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if not k.is_cuda:
        raise RuntimeError("FLA fused Delta kernel requires a CUDA GPU")
    if chunk_size not in (16, 32, 64):
        raise ValueError("FLA chunk_size must be 16, 32, or 64")
    try:
        from fla.ops.delta_rule import chunk_delta_rule
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except ImportError as exc:
        raise ImportError("Install fla and matching Triton for GPU testing") from exc

    with torch.autocast(device_type="cuda", enabled=False):
        Hf = H.float()
        sf = state.float()
        kf, qf = k.float(), q.float()
        L = torch.linalg.cholesky(Hf)
        z_k = torch.linalg.solve_triangular(
            L, kf.permute(0, 2, 3, 1), upper=False
        ).permute(0, 3, 1, 2)
        z_q = torch.linalg.solve_triangular(
            L, qf.permute(0, 2, 3, 1), upper=False
        ).permute(0, 3, 1, 2)
        W0 = L.transpose(-1, -2) @ sf
        beta = 1. / (tau + z_k.square().sum(-1))

    # FLA kernel needs BF16 q/k/v, but keeps recurrent state FP32 where
    # supported. Do not enable implicit K normalization or 1/sqrt(d) scaling.
    common = dict(
        q=z_q.to(torch.bfloat16),
        k=z_k.to(torch.bfloat16),
        v=v.to(torch.bfloat16),
        beta=beta.to(torch.bfloat16),
        scale=1.0,
        initial_state=W0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        chunk_size=chunk_size,
    )
    if log_gate is None:
        out, W_final = chunk_delta_rule(**common)
    else:
        if log_gate.shape != k.shape[:3]:
            raise ValueError("log_gate must be [batch, tokens, heads]")
        out, W_final = chunk_gated_delta_rule(
            g=log_gate.to(torch.bfloat16), **common
        )
    if W_final is None:
        raise RuntimeError("FLA did not return a final recurrent state")
    with torch.autocast(device_type="cuda", enabled=False):
        S_final = torch.linalg.solve_triangular(
            L.transpose(-1, -2), W_final.float(), upper=True
        )
    return out, S_final
