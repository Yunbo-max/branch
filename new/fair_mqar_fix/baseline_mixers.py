"""Baseline mixers for the fair MQAR comparison.

Both expose the same interface as the other arms — ``forward(x, cache) -> (out, cache, stats)``
with a cache exposing ``detach()`` and ``allocated_tensor_bytes()`` — so the shared LM,
data path and evaluation code are unchanged.

* ``DeltaNetMixer``: one evolving delta-rule state, read ``Sᵀq``. No anchor bank, no
  branching, no history read. This is the paper's Gated DeltaNet baseline at our scale.
* ``TransformerMixer``: standard causal multi-head attention with RoPE and an incremental
  KV cache.

Fairness note: the short depthwise convolution is shared by every arm (including the
Transformer). The paper's Transformer baseline has no such convolution; keeping it here
means the arms differ only in the memory mechanism, which is the point of this harness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class BaselineConfig:
    d_model: int = 128
    n_heads: int = 2
    d_key: int = 32
    d_value: int = 32
    conv_kernel: int = 3
    tau: float = 1.0
    ridge: float = 1.0
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.conv_kernel < 1:
            raise ValueError("conv_kernel must be positive")


def _causal_conv(x: Tensor, conv: nn.Conv1d, cache, kernel: int):
    conv_input = torch.cat([cache, x.transpose(1, 2)], -1)
    local = x + F.silu(conv(conv_input).transpose(1, 2))
    nxt = conv_input[..., -(kernel - 1):] if kernel > 1 else conv_input[..., :0]
    return local, nxt


class _BaseCache:
    def __init__(self, conv_context: Tensor, step: int = 0) -> None:
        self.conv_context = conv_context
        self.step = step

    def detach(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def allocated_tensor_bytes(self) -> int:
        return self.conv_context.numel() * self.conv_context.element_size()


class DeltaCache(_BaseCache):
    def __init__(self, conv_context: Tensor, state: Tensor | None = None,
                 step: int = 0, max_len: int = 0) -> None:
        super().__init__(conv_context, step)
        self.state = state
        self.max_len = max_len

    def detach(self) -> "DeltaCache":
        return DeltaCache(self.conv_context.detach(), self.state.detach(), self.step, self.max_len)

    def allocated_tensor_bytes(self) -> int:
        return (super().allocated_tensor_bytes()
                + self.state.numel() * self.state.element_size())


class AttentionCache(_BaseCache):
    def __init__(self, conv_context: Tensor, k: Tensor | None = None,
                 v: Tensor | None = None, step: int = 0, length: int = 0) -> None:
        super().__init__(conv_context, step)
        self.k, self.v, self.length = k, v, length

    def detach(self) -> "AttentionCache":
        return AttentionCache(self.conv_context.detach(), self.k.detach(), self.v.detach(),
                              self.step, self.length)

    def allocated_tensor_bytes(self) -> int:
        return (super().allocated_tensor_bytes()
                + self.k.numel() * self.k.element_size()
                + self.v.numel() * self.v.element_size())


class DeltaNetMixer(nn.Module):
    """Single-state gated delta rule: S ← S + β(v − Sk)kᵀ, read Sᵀq."""

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        self.conv = nn.Conv1d(c.d_model, c.d_model, c.conv_kernel, groups=c.d_model, bias=False)
        self.q_proj = nn.Linear(c.d_model, c.n_heads * c.d_key, bias=False)
        self.k_proj = nn.Linear(c.d_model, c.n_heads * c.d_key, bias=False)
        self.v_proj = nn.Linear(c.d_model, c.n_heads * c.d_value, bias=False)
        self.out_proj = nn.Linear(c.n_heads * c.d_value, c.d_model, bias=False)

    def initial_cache(self, x: Tensor) -> DeltaCache:
        c = self.config
        b = x.shape[0]
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        state = torch.zeros(b, c.n_heads, c.d_key, c.d_value, device=x.device, dtype=dtype)
        return DeltaCache(x.new_zeros(b, c.d_model, c.conv_kernel - 1), state, 0, 0)

    def forward(self, x: Tensor, cache: DeltaCache | None = None):
        c = self.config
        if x.ndim != 3 or x.shape[-1] != c.d_model:
            raise ValueError("x must be [B,T,d_model]")
        b, t, _ = x.shape
        h = c.n_heads
        cache = cache or self.initial_cache(x)
        dtype = cache.state.dtype

        local, next_context = _causal_conv(x, self.conv, cache.conv_context, c.conv_kernel)
        q = F.normalize(self.q_proj(local).view(b, t, h, c.d_key).to(dtype), dim=-1, eps=1e-6)
        k = F.normalize(self.k_proj(local).view(b, t, h, c.d_key).to(dtype), dim=-1, eps=1e-6)
        v = torch.tanh(self.v_proj(local).view(b, t, h, c.d_value).to(dtype))
        state = cache.state
        beta = 1.0 / (1.0 + c.ridge * c.tau)

        outs = []
        for i in range(t):
            kt, vt, qt = k[:, i], v[:, i], q[:, i]
            residual = vt - torch.einsum("bhkv,bhk->bhv", state, kt)
            state = state + beta * residual.unsqueeze(-1) * kt.unsqueeze(-2)
            outs.append(torch.einsum("bhkv,bhk->bhv", state, qt))

        out = self.out_proj(torch.stack(outs, 1).reshape(b, t, h * c.d_value).to(local.dtype))
        new_cache = DeltaCache(next_context, state, cache.step + t, 0)
        return out, new_cache, _empty_stats(x, b, t, h, dtype, active=1)


class GatedDeltaNetMixer(nn.Module):
    """Gated DeltaNet (GDN) with a learnable input-dependent forget gate.

    S_t = alpha_t S_{t-1} + beta_t k_t (v_t - (alpha_t S_{t-1})^T k_t)^T,
    alpha_t = sigmoid(W_g x_t + b_g) per head, beta_t = 1 / (tau*ridge + ||k_t||^2).

    beta matches single_geo ``isotropic`` so the only difference vs that arm is
    the scalar forget gate. Init biases alpha_t to ~0.88 so the model starts
    near the ungated delta rule and can learn to forget.
    """

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        self.conv = nn.Conv1d(c.d_model, c.d_model, c.conv_kernel, groups=c.d_model, bias=False)
        self.q_proj = nn.Linear(c.d_model, c.n_heads * c.d_key, bias=False)
        self.k_proj = nn.Linear(c.d_model, c.n_heads * c.d_key, bias=False)
        self.v_proj = nn.Linear(c.d_model, c.n_heads * c.d_value, bias=False)
        self.out_proj = nn.Linear(c.n_heads * c.d_value, c.d_model, bias=False)
        self.gate_proj = nn.Linear(c.d_model, c.n_heads, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 2.0)  # sigmoid(2.0) ~= 0.88

    def initial_cache(self, x: Tensor) -> DeltaCache:
        c = self.config
        b = x.shape[0]
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        state = torch.zeros(b, c.n_heads, c.d_key, c.d_value, device=x.device, dtype=dtype)
        return DeltaCache(x.new_zeros(b, c.d_model, c.conv_kernel - 1), state, 0, 0)

    def forward(self, x: Tensor, cache: DeltaCache | None = None):
        c = self.config
        if x.ndim != 3 or x.shape[-1] != c.d_model:
            raise ValueError("x must be [B,T,d_model]")
        b, t, _ = x.shape
        h = c.n_heads
        cache = cache or self.initial_cache(x)
        dtype = cache.state.dtype

        local, next_context = _causal_conv(x, self.conv, cache.conv_context, c.conv_kernel)
        q = F.normalize(self.q_proj(local).view(b, t, h, c.d_key).to(dtype), dim=-1, eps=1e-6)
        k = F.normalize(self.k_proj(local).view(b, t, h, c.d_key).to(dtype), dim=-1, eps=1e-6)
        v = torch.tanh(self.v_proj(local).view(b, t, h, c.d_value).to(dtype))
        gate = torch.sigmoid(self.gate_proj(local).to(dtype))  # [B, T, H]
        state = cache.state

        outs = []
        for i in range(t):
            kt, vt, qt = k[:, i], v[:, i], q[:, i]
            alpha = gate[:, i].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
            state = alpha * state  # forget on the old state
            residual = vt - torch.einsum("bhkv,bhk->bhv", state, kt)  # [B,H,V]
            beta = 1.0 / (c.tau * c.ridge + (kt * kt).sum(-1))  # [B,H]
            state = state + beta.unsqueeze(-1).unsqueeze(-1) * kt.unsqueeze(-1) * residual.unsqueeze(-2)
            outs.append(torch.einsum("bhkv,bhk->bhv", state, qt))

        out = self.out_proj(torch.stack(outs, 1).reshape(b, t, h * c.d_value).to(local.dtype))
        new_cache = DeltaCache(next_context, state, cache.step + t, 0)
        return out, new_cache, _empty_stats(x, b, t, h, dtype, active=1)


class TransformerMixer(nn.Module):
    """Causal multi-head attention with RoPE and an incremental KV cache."""

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        if c.d_model % c.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.head_dim = c.d_model // c.n_heads
        self.conv = nn.Conv1d(c.d_model, c.d_model, c.conv_kernel, groups=c.d_model, bias=False)
        self.q_proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.k_proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.v_proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.out_proj = nn.Linear(c.d_model, c.d_model, bias=False)
        inv = 1.0 / (c.rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("rope_inv", inv, persistent=False)

    def _rope(self, x: Tensor, offset: int) -> Tensor:
        # x: B H T D
        t = x.shape[2]
        pos = torch.arange(offset, offset + t, device=x.device, dtype=torch.float32)
        freq = torch.einsum("t,d->td", pos, self.rope_inv.to(x.device))
        cos, sin = freq.cos().to(x.dtype), freq.sin().to(x.dtype)
        cos = cos.view(1, 1, t, -1)
        sin = sin.view(1, 1, t, -1)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

    def initial_cache(self, x: Tensor) -> AttentionCache:
        c = self.config
        b = x.shape[0]
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        h, d = c.n_heads, self.head_dim
        return AttentionCache(
            x.new_zeros(b, c.d_model, c.conv_kernel - 1),
            torch.zeros(b, h, 0, d, device=x.device, dtype=dtype),
            torch.zeros(b, h, 0, d, device=x.device, dtype=dtype),
            0, 0,
        )

    def forward(self, x: Tensor, cache: AttentionCache | None = None):
        c = self.config
        if x.ndim != 3 or x.shape[-1] != c.d_model:
            raise ValueError("x must be [B,T,d_model]")
        b, t, _ = x.shape
        h, d = c.n_heads, self.head_dim
        cache = cache or self.initial_cache(x)
        dtype = cache.k.dtype

        local, next_context = _causal_conv(x, self.conv, cache.conv_context, c.conv_kernel)
        q = self.q_proj(local).view(b, t, h, d).transpose(1, 2).to(dtype)
        k = self.k_proj(local).view(b, t, h, d).transpose(1, 2).to(dtype)
        v = self.v_proj(local).view(b, t, h, d).transpose(1, 2).to(dtype)

        offset = cache.length
        q, k = self._rope(q, offset), self._rope(k, offset)
        k_all = torch.cat([cache.k, k], dim=2)
        v_all = torch.cat([cache.v, v], dim=2)

        total = k_all.shape[2]
        # Use the fused SDPA/FlashAttention path. Training (no cache) uses is_causal;
        # cached decode uses an explicit additive mask with the correct offset.
        if offset == 0:
            ctx = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=True)
        else:
            causal = torch.ones(
                t, total, device=x.device, dtype=torch.bool
            ).tril(diagonal=offset)
            ctx = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=causal)
        out = self.out_proj(ctx.transpose(1, 2).reshape(b, t, c.d_model).to(local.dtype))
        new_cache = AttentionCache(next_context, k_all, v_all, cache.step + t, offset + t)
        return out, new_cache, _empty_stats(x, b, t, h, dtype, active=offset + t)


def _empty_stats(x: Tensor, b: int, t: int, h: int, dtype, active: int) -> dict:
    """Same stat keys as the memory arms, so shared code does not branch."""
    return {
        "active_states": torch.full((b, t), float(active), device=x.device, dtype=dtype),
        "fork": torch.zeros(b, t, h, dtype=torch.bool, device=x.device),
        "parent": torch.zeros(b, t, h, dtype=torch.long, device=x.device),
        "destination": torch.zeros(b, t, h, dtype=torch.long, device=x.device),
        "historical_candidates": torch.full((b, t), float(active), device=x.device, dtype=dtype),
        "write_cost": torch.zeros(b, t, h, dtype=dtype, device=x.device),
    }
