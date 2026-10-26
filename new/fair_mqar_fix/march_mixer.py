"""MARCH mixer — faithful to the released implementation.

Semantics taken from the official code archived at
``new/MARCH/code/march_official/`` (HuggingFace ``AI45Research/ReSI-Safin-1-35B-A3B``):

* state orientation is ``[K, V]`` per head; the read is ``S^T q``
  (``einsum("bthk,bmhkv->bthmv", read_q, states)``);
* anchor m is visible at token t iff ``boundary_m <= t`` (inclusive);
* route logits are scaled by ``head_k_dim**-0.5 * route_logit_scale``;
* Top-K (released default K=4) is applied over the valid anchors;
* the null option is an extra candidate that takes softmax mass but is dropped
  from the output, so anchor weights sum to less than one;
* routing uses the official masked-softmax-and-renormalise helper
  (``ops.route_softmax_weights``), imported from the archived source;
* fusion is additive: ``o = S^T q + history_scale * sum_m w_m S_m^T q``.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_OFFICIAL_DIR = Path(__file__).resolve().parents[1] / "MARCH" / "code" / "march_official"


def _official_route_softmax():
    """Import the official helper so routing numerics are not re-derived."""
    if not (_OFFICIAL_DIR / "ops.py").exists():
        raise FileNotFoundError(
            f"official ops.py missing at {_OFFICIAL_DIR}; see march_official/PROVENANCE.md"
        )
    if str(_OFFICIAL_DIR) not in sys.path:
        sys.path.insert(0, str(_OFFICIAL_DIR))
    from ops import route_softmax_weights

    return route_softmax_weights


route_softmax_weights = _official_route_softmax()


@dataclass
class MARCHMemoryConfig:
    d_model: int = 128
    n_heads: int = 2
    d_key: int = 32
    d_value: int = 32
    d_route: int = 32          # route head dim (official hsd_route_head_dim)
    anchor_interval: int = 16  # C text tokens between anchors
    max_anchors: int = 16
    read_topk: int = 4         # official hsd_read_topk
    use_route_scale: bool = True
    route_logit_scale: float = 1.0
    null_read: bool = True
    history_scale: float = 1.0
    conv_kernel: int = 3
    tau: float = 1.0
    ridge: float = 1.0

    def __post_init__(self) -> None:
        for name in ("d_model", "n_heads", "d_key", "d_value", "d_route",
                     "anchor_interval", "max_anchors", "conv_kernel"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.read_topk < 0:
            raise ValueError("read_topk must be >= 0 (0 disables the Top-K mask)")
        if self.tau <= 0 or self.ridge <= 0:
            raise ValueError("tau and ridge must be positive")
        if self.history_scale < 0:
            raise ValueError("history_scale must be non-negative")


@dataclass
class MARCHCache:
    states: Tensor        # B M H K V  (official layout: anchors before heads)
    current: Tensor       # B H K V
    boundaries: Tensor    # B M
    anchor_k: Tensor      # B M H R
    counts: Tensor        # B  (number of live anchors)
    conv_context: Tensor
    step: int = 0
    max_anchors: int = 0

    def detach(self) -> "MARCHCache":
        return MARCHCache(
            self.states.detach(), self.current.detach(), self.boundaries.detach(),
            self.anchor_k.detach(), self.counts.detach(), self.conv_context.detach(),
            self.step, self.max_anchors,
        )

    def allocated_tensor_bytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for t in (self.states, self.current, self.boundaries,
                             self.anchor_k, self.conv_context))


class MARCHMixer(nn.Module):
    """Anchor memory + content-routed history read over a shared delta recurrence."""

    def __init__(self, config: MARCHMemoryConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        self.conv = nn.Conv1d(c.d_model, c.d_model, c.conv_kernel, groups=c.d_model, bias=False)
        self.q_proj = nn.Linear(c.d_model, c.n_heads * c.d_key, bias=False)
        self.k_proj = nn.Linear(c.d_model, c.n_heads * c.d_key, bias=False)
        self.v_proj = nn.Linear(c.d_model, c.n_heads * c.d_value, bias=False)
        self.route_q = nn.Linear(c.d_model, c.n_heads * c.d_route, bias=False)
        # Anchor position representation: one shared learned anchor token per layer.
        self.anchor_embed = nn.Parameter(torch.randn(c.d_model) / math.sqrt(c.d_model))
        self.anchor_k_proj = nn.Linear(c.d_model, c.n_heads * c.d_route, bias=False)
        # Null gate: official has a dedicated projection plus a learned bias.
        self.null_gate_proj = nn.Linear(c.d_model, c.n_heads, bias=True) if c.null_read else None
        self.out_proj = nn.Linear(c.n_heads * c.d_value, c.d_model, bias=False)

    # ---------- helpers ----------

    def _route_scale(self) -> float:
        c = self.config
        scale = c.d_key ** -0.5 if c.use_route_scale else 1.0
        return scale * c.route_logit_scale

    def initial_cache(self, x: Tensor) -> MARCHCache:
        c = self.config
        b = x.shape[0]
        h = c.n_heads
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        z = lambda *s: torch.zeros(*s, device=x.device, dtype=dtype)
        return MARCHCache(
            z(b, c.max_anchors, h, c.d_key, c.d_value),
            z(b, h, c.d_key, c.d_value),
            torch.zeros(b, c.max_anchors, device=x.device, dtype=torch.long),
            z(b, c.max_anchors, h, c.d_route),
            torch.zeros(b, device=x.device, dtype=torch.long),
            x.new_zeros(b, c.d_model, c.conv_kernel - 1),
            step=0,
            max_anchors=c.max_anchors,
        )

    # ---------- read path (official semantics) ----------

    def history_read(
        self,
        read_q: Tensor,      # B T H K
        router_q: Tensor,    # B T H R
        states: Tensor,      # B M H K V
        anchor_k: Tensor,    # B M H R
        boundaries: Tensor,  # B M
        null_logits: Tensor | None,  # B T H 1
        start_index: int = 0,
    ) -> tuple[Tensor, Tensor]:
        c = self.config
        b, t, h, _ = read_q.shape
        m = states.shape[1]
        v = states.shape[-1]
        if m == 0:
            return read_q.new_zeros(b, t, h, v), read_q.new_zeros(b, t, h, 0)

        token_index = torch.arange(start_index, start_index + t,
                                   device=read_q.device, dtype=torch.long)
        valid = (boundaries.view(b, 1, m) <= token_index.view(1, t, 1))  # inclusive
        valid = valid.view(b, t, 1, m).expand(b, t, h, m)

        logits = torch.einsum("bthr,bmhr->bthm", router_q.float(), anchor_k.float())
        logits = logits * self._route_scale()

        topk = c.read_topk
        if 0 < topk < m:
            _, idx = logits.masked_fill(~valid, float("-inf")).topk(k=topk, dim=-1)
            keep = torch.zeros_like(logits, dtype=torch.bool)
            keep.scatter_(-1, idx, True)
            valid = valid & keep

        if null_logits is not None:
            null_valid = torch.ones(b, t, h, 1, dtype=torch.bool, device=valid.device)
            logits_with_null = torch.cat([logits, null_logits.float()], dim=-1)
            valid_with_null = torch.cat([valid, null_valid], dim=-1)
            weights = route_softmax_weights(logits_with_null, valid_with_null, dim=-1)[..., :m]
        else:
            weights = route_softmax_weights(logits, valid, dim=-1)

        reads = torch.einsum("bthk,bmhkv->bthmv", read_q.to(states.dtype), states)
        reads = reads * (c.d_key ** -0.5)
        reads = reads.masked_fill(~valid.unsqueeze(-1), 0)
        out = torch.einsum("bthm,bthmv->bthv", weights.to(reads.dtype), reads)
        return out, weights

    # ---------- forward ----------

    def forward(self, x: Tensor, cache: MARCHCache | None = None):
        c = self.config
        if x.ndim != 3 or x.shape[-1] != c.d_model:
            raise ValueError("x must be [B,T,d_model]")
        b, t, _ = x.shape
        h = c.n_heads
        if cache is None:
            cache = self.initial_cache(x)
        dtype = cache.states.dtype

        conv_input = torch.cat([cache.conv_context, x.transpose(1, 2)], -1)
        local = x + F.silu(self.conv(conv_input).transpose(1, 2))
        next_context = conv_input[..., -(c.conv_kernel - 1):] if c.conv_kernel > 1 else conv_input[..., :0]

        q = F.normalize(self.q_proj(local).view(b, t, h, c.d_key).to(dtype), dim=-1, eps=1e-6)
        k = F.normalize(self.k_proj(local).view(b, t, h, c.d_key).to(dtype), dim=-1, eps=1e-6)
        v = torch.tanh(self.v_proj(local).view(b, t, h, c.d_value).to(dtype))
        rq = self.route_q(local).view(b, t, h, c.d_route).to(dtype)
        null_logits = None
        if self.null_gate_proj is not None:
            null_logits = self.null_gate_proj(local).to(dtype).unsqueeze(-1)  # B T H 1

        states, current = cache.states, cache.current
        boundaries, anchor_k, counts = cache.boundaries, cache.anchor_k, cache.counts
        beta = 1.0 / (1.0 + c.ridge * c.tau)

        outputs = []
        forkless = []
        for i in range(t):
            kt, vt, qt = k[:, i], v[:, i], q[:, i]
            # Shared delta recurrence, identical to Branch Memory's `delta` control.
            residual = vt - torch.einsum("bhkv,bhk->bhv", current, kt)
            current = current + beta * residual.unsqueeze(-1) * kt.unsqueeze(-2)

            absolute = cache.step + i
            # Anchor boundary: snapshot the cumulative state, then read.
            if (absolute + 1) % c.anchor_interval == 0:
                slot = (absolute + 1) // c.anchor_interval - 1
                if slot < c.max_anchors:
                    slot_mask = (torch.arange(c.max_anchors, device=x.device) == slot)
                    states = torch.where(slot_mask[None, :, None, None, None],
                                         current.unsqueeze(1), states)
                    boundaries = torch.where(slot_mask[None, :],
                                             torch.full_like(boundaries, absolute),
                                             boundaries)
                    anchor_hidden = self.anchor_embed.to(dtype).view(1, 1, c.d_model).expand(b, 1, c.d_model)
                    ak = self.anchor_k_proj(anchor_hidden).view(b, 1, h, c.d_route).to(dtype)
                    anchor_k = torch.where(slot_mask[None, :, None, None],
                                           ak.expand(b, 1, h, c.d_route), anchor_k)
                    counts = torch.clamp(counts, min=slot + 1)

            hist, _ = self.history_read(qt.unsqueeze(1), rq[:, i:i + 1], states, anchor_k,
                                        boundaries, None if null_logits is None else null_logits[:, i:i + 1],
                                        start_index=absolute)
            current_out = torch.einsum("bhkv,bhk->bhv", current, qt)
            outputs.append(current_out + c.history_scale * hist[:, 0])
            forkless.append(states.new_zeros(b, h))

        out = torch.stack(outputs, 1).reshape(b, t, h * c.d_value)
        out = self.out_proj(out.to(local.dtype))
        new_cache = MARCHCache(states, current, boundaries, anchor_k, counts,
                               next_context, cache.step + t, c.max_anchors)
        stats = {
            "active_states": counts.view(b, 1).expand(b, t).clone(),
            "fork": torch.zeros(b, t, h, dtype=torch.bool, device=x.device),
            "parent": torch.zeros(b, t, h, dtype=torch.long, device=x.device),
            "destination": torch.zeros(b, t, h, dtype=torch.long, device=x.device),
            "historical_candidates": counts.view(b, 1).expand(b, t).clone(),
            "write_cost": torch.zeros(b, t, h, dtype=dtype, device=x.device),
        }
        return out, new_cache, stats
