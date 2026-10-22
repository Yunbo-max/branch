"""Chunked MARCH: the same anchor mechanism at chunk granularity.

The per-token ``march_mixer`` is correct but costs one Python iteration per token, which
is infeasible at 16K context. This variant keeps the official read semantics from
``march_mixer`` (state layout ``[K,V]``, ``S^T q`` read, Top-K, null option absorbing
softmax mass, inclusive boundary visibility) and replaces the base recurrence with the
verified chunked delta update.

Configuration: one anchor per chunk (``anchor_interval == chunk_len``), so anchor ``m``
summarises the cumulative state after chunk ``m`` and is visible to tokens at or after
its boundary. MARCH has no protection matrix, so the chunk geometry is ``ridge * I`` and
the solve reduces to division by ``ridge``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_BRANCH_DIR = Path(__file__).resolve().parents[1] / "branch_memory"
if str(_BRANCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BRANCH_DIR))

from branch_memory.mixer import BranchMemoryMixer  # noqa: E402
from chunked_delta import chunked_delta_update  # noqa: E402


@dataclass
class ChunkedMARCHConfig:
    d_model: int = 128
    n_heads: int = 2
    d_key: int = 32
    d_value: int = 32
    d_route: int = 32
    n_probes: int = 2
    max_states: int = 8          # anchor capacity
    conv_kernel: int = 3
    tau: float = 1.0
    ridge: float = 1.0
    read_topk: int = 4
    route_logit_scale: float = 1.0
    null_read: bool = True
    chunk_len: int = 256

    def __post_init__(self) -> None:
        for name in ("d_model", "n_heads", "d_key", "d_value", "d_route",
                     "n_probes", "max_states", "conv_kernel", "chunk_len"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.tau <= 0 or self.ridge <= 0:
            raise ValueError("tau and ridge must be positive")


@dataclass
class MARCHChunkCache:
    state: Tensor            # B H K V
    anchors: Tensor          # B M H K V (empty slots are zero)
    boundaries: Tensor       # B M  (absolute token index where the anchor was taken)
    anchor_k: Tensor         # B M H R
    counts: Tensor           # B
    conv_context: Tensor
    step: int = 0
    max_states: int = 0

    def detach(self) -> "MARCHChunkCache":
        return MARCHChunkCache(self.state.detach(), self.anchors.detach(),
                               self.boundaries.detach(), self.anchor_k.detach(),
                               self.counts.detach(), self.conv_context.detach(),
                               self.step, self.max_states)

    def allocated_tensor_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in
                   (self.state, self.anchors, self.boundaries, self.anchor_k,
                    self.conv_context))


class ChunkedMARCHMixer(BranchMemoryMixer):
    """Anchor memory with chunk-level recurrence and a frozen anchor read."""

    def __init__(self, config: ChunkedMARCHConfig) -> None:
        base = type("_Base", (), dict(
            d_model=config.d_model, n_heads=config.n_heads, d_key=config.d_key,
            d_value=config.d_value, d_route=config.d_route, n_probes=config.n_probes,
            max_states=config.max_states, mode="branch", snapshot_interval=4,
            conv_kernel=config.conv_kernel, tau=config.tau, ridge=config.ridge,
            protect=0.0, fork_cost=0.0, protection_decay=0.95, covariance="full",
            differentiate_covariance=False,
        ))()
        super().__init__(base)
        self.config = config
        self.anchor_embed = nn.Parameter(torch.randn(config.d_model) / (config.d_model ** 0.5))
        self.anchor_k_proj = nn.Linear(config.d_model, config.n_heads * config.d_route, bias=False)
        self.null_gate_proj = (nn.Linear(config.d_model, config.n_heads, bias=True)
                               if config.null_read else None)

    def initial_cache(self, x: Tensor) -> MARCHChunkCache:
        c = self.config
        b = x.shape[0]
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        z = lambda *s: torch.zeros(*s, device=x.device, dtype=dtype)
        return MARCHChunkCache(
            z(b, c.n_heads, c.d_key, c.d_value),
            z(b, c.max_states, c.n_heads, c.d_key, c.d_value),
            torch.full((b, c.max_states), -1, dtype=torch.long, device=x.device),
            z(b, c.max_states, c.n_heads, c.d_route),
            torch.zeros(b, dtype=torch.long, device=x.device),
            x.new_zeros(b, c.d_model, c.conv_kernel - 1),
            0, c.max_states,
        )

    def _read(self, q: Tensor, rq: Tensor, anchors: Tensor, anchor_k: Tensor,
              boundaries: Tensor, counts: Tensor, start: int,
              null_logits: Tensor | None) -> Tensor:
        """Official MARCH read: Top-K over visible anchors, null option, S^T q."""
        c = self.config
        B, L, H, _ = q.shape
        M = anchors.shape[1]
        if M == 0:
            return q.new_zeros(B, L, H, c.d_value)
        tok = torch.arange(start, start + L, device=q.device, dtype=torch.long)
        valid = boundaries.view(B, 1, M) <= tok.view(1, L, 1)      # (B,L,M), inclusive
        valid = valid.view(B, 1, L, M).expand(B, H, L, M)          # (B,H,L,M)
        live = torch.arange(M, device=q.device).view(1, 1, M) < counts.view(B, 1, 1)
        valid = valid & live.view(B, 1, 1, M)

        scale = (c.d_key ** -0.5) * c.route_logit_scale
        logits = torch.einsum("blhr,bmhr->bhlm", rq.float(), anchor_k.float()) * scale
        if 0 < c.read_topk < M:
            _, idx = logits.masked_fill(~valid, float("-inf")).topk(c.read_topk, dim=-1)
            keep = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, idx, True)
            valid = valid & keep
        if null_logits is not None:
            logits = torch.cat([logits, null_logits.float()], dim=-1)
            valid = torch.cat([valid, torch.ones(B, H, L, 1, dtype=torch.bool, device=q.device)], -1)
        weights = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)[..., :M]

        reads = torch.einsum("blhk,bmhkv->bmlhv", q, anchors) * (c.d_key ** -0.5)
        reads = reads.masked_fill(~valid[..., :M].permute(0, 3, 2, 1).unsqueeze(-1), 0)
        return torch.einsum("bhlm,bmlhv->blhv", weights.to(reads.dtype), reads)

    def forward(self, x: Tensor, cache: MARCHChunkCache | None = None):
        c = self.config
        if x.ndim != 3 or x.shape[-1] != c.d_model:
            raise ValueError("x must be [B,T,d_model]")
        B, T, _ = x.shape
        H = c.n_heads
        if cache is None:
            cache = self.initial_cache(x)
        dtype = cache.state.dtype

        conv_input = torch.cat([cache.conv_context, x.transpose(1, 2)], -1)
        local = x + F.silu(self.conv(conv_input).transpose(1, 2))
        next_context = conv_input[..., -(c.conv_kernel - 1):] if c.conv_kernel > 1 else conv_input[..., :0]

        q = F.normalize(self.q_proj(local).view(B, T, H, c.d_key).to(dtype), dim=-1, eps=1e-6)
        k = F.normalize(self.k_proj(local).view(B, T, H, c.d_key).to(dtype), dim=-1, eps=1e-6)
        v = torch.tanh(self.v_proj(local).view(B, T, H, c.d_value).to(dtype))
        rq = self.route_q(local).view(B, T, H, c.d_route).to(dtype)
        null_logits = None
        if self.null_gate_proj is not None:
            null_logits = self.null_gate_proj(local).to(dtype)

        state = cache.state
        anchors, boundaries, anchor_k, counts = (cache.anchors, cache.boundaries,
                                                 cache.anchor_k, cache.counts)
        factor = torch.cholesky(c.ridge * torch.eye(c.d_key, device=x.device, dtype=dtype))
        factor = factor.expand(B, H, c.d_key, c.d_key).contiguous()
        outs = []

        for start in range(0, T, c.chunk_len):
            end = min(T, start + c.chunk_len)
            L = end - start
            cur, new_state = chunked_delta_update(
                state, k[:, start:end], v[:, start:end], q[:, start:end], factor, c.tau,
                return_states=True,
            )
            nl = (None if null_logits is None
                  else null_logits[:, start:end].unsqueeze(-1).transpose(1, 2))  # (B,H,L,1)
            hist = self._read(q[:, start:end], rq[:, start:end], anchors, anchor_k,
                              boundaries, counts, cache.step + start, nl)
            outs.append(cur + hist)
            state = new_state

            # anchor at the chunk boundary: cumulative state after this chunk
            slot = (cache.step + end) // c.chunk_len - 1
            if 0 <= slot < c.max_states:
                mask = torch.arange(c.max_states, device=x.device) == slot
                anchors = torch.where(mask.view(1, c.max_states, 1, 1, 1),
                                      state.unsqueeze(1), anchors)
                boundaries = torch.where(mask.view(1, c.max_states),
                                         torch.full_like(boundaries, cache.step + end - 1),
                                         boundaries)
                a_hidden = self.anchor_embed.to(dtype).view(1, 1, c.d_model).expand(B, 1, c.d_model)
                ak = self.anchor_k_proj(a_hidden).view(B, 1, H, c.d_route).to(dtype)
                anchor_k = torch.where(mask.view(1, c.max_states, 1, 1),
                                       ak.expand(B, 1, H, c.d_route), anchor_k)
                counts = torch.clamp(counts, min=slot + 1)

        out = self.out_proj(torch.cat(outs, 1).reshape(B, T, H * c.d_value).to(local.dtype))
        new_cache = MARCHChunkCache(state, anchors, boundaries, anchor_k, counts,
                                    next_context, cache.step + T, c.max_states)
        stats = {
            "active_states": counts.view(B, 1).expand(B, T).float(),
            "fork": torch.zeros(B, T, H, dtype=torch.bool, device=x.device),
            "parent": torch.zeros(B, T, H, dtype=torch.long, device=x.device),
            "destination": torch.zeros(B, T, H, dtype=torch.long, device=x.device),
            "historical_candidates": counts.view(B, 1).expand(B, T).float(),
            "write_cost": torch.zeros(B, T, H, dtype=dtype, device=x.device),
        }
        return out, new_cache, stats
