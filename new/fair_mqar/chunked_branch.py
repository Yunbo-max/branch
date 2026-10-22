"""Chunked Branch Memory: frozen-routing iteration with chunk-level linear algebra.

Relative to ``branch_memory``:

* one SPD factorization per state per **chunk** instead of per token;
* routing for all tokens in a chunk is decided from the **chunk-start** states and
  geometry (deliberate approximation: a state created inside the chunk cannot be a parent
  until the next boundary);
* the per-token write becomes the exact chunked triangular solve in ``chunked_delta``;
* ``C`` is refreshed at the boundary with the exact EMA
  ``C_{b+1} = gamma^L C_b + (1-gamma) sum_s gamma^{L-1-s} w_s q_s q_s^T``;
* content descriptors are refreshed at the boundary.

``chunk_len == 1`` reproduces a per-token reference (``tests/test_chunked_branch.py``).

Not implemented here, explicitly: fork / copy-on-write version management. Every token
replaces its selected parent, so this is chunked replace-only Branch Memory.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

_BRANCH_DIR = Path(__file__).resolve().parents[1] / "branch_memory"
if str(_BRANCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BRANCH_DIR))

from branch_memory.mixer import BranchMemoryMixer  # noqa: E402
from chunked_delta import _solve_keys, chunked_delta_update  # noqa: E402


@dataclass
class ChunkedBranchConfig:
    d_model: int = 128
    n_heads: int = 2
    d_key: int = 32
    d_value: int = 32
    d_route: int = 32
    n_probes: int = 2
    max_states: int = 8
    conv_kernel: int = 3
    tau: float = 1.0
    ridge: float = 1.0
    protect: float = 8.0
    protection_decay: float = 0.95
    fork_cost: float = 0.02
    chunk_len: int = 64
    grad_checkpoint: bool = False

    def __post_init__(self) -> None:
        for name in ("d_model", "n_heads", "d_key", "d_value", "d_route",
                     "n_probes", "max_states", "conv_kernel", "chunk_len"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.tau <= 0 or self.ridge <= 0:
            raise ValueError("tau and ridge must be positive")
        if not 0 <= self.protection_decay < 1:
            raise ValueError("protection_decay must be in [0,1)")


@dataclass
class ChunkedCache:
    states: Tensor          # B H M K V
    protection: Tensor      # B H M K K
    active: Tensor          # B H M
    addresses: Tensor       # B H M R
    conv_context: Tensor
    step: int = 0
    max_states: int = 0

    def detach(self) -> "ChunkedCache":
        return ChunkedCache(self.states.detach(), self.protection.detach(),
                            self.active.detach(), self.addresses.detach(),
                            self.conv_context.detach(), self.step, self.max_states)

    def allocated_tensor_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in
                   (self.states, self.protection, self.active, self.addresses,
                    self.conv_context))


class ChunkedBranchMixer(BranchMemoryMixer):
    """Branch Memory with chunk-level factorization, routing and write."""

    def __init__(self, config: ChunkedBranchConfig) -> None:
        base = type("_Base", (), dict(
            d_model=config.d_model, n_heads=config.n_heads, d_key=config.d_key,
            d_value=config.d_value, d_route=config.d_route, n_probes=config.n_probes,
            max_states=config.max_states, mode="branch", snapshot_interval=4,
            conv_kernel=config.conv_kernel, tau=config.tau, ridge=config.ridge,
            protect=config.protect, fork_cost=config.fork_cost,
            protection_decay=config.protection_decay, covariance="full",
            differentiate_covariance=True,
        ))()
        super().__init__(base)
        self.config = config

    # ---- cache -------------------------------------------------------------
    def initial_cache(self, x: Tensor) -> ChunkedCache:
        c = self.config
        b = x.shape[0]
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        z = lambda *s: torch.zeros(*s, device=x.device, dtype=dtype)
        active = torch.zeros(b, c.n_heads, c.max_states, device=x.device, dtype=torch.bool)
        active[..., 0] = True
        return ChunkedCache(z(b, c.n_heads, c.max_states, c.d_key, c.d_value),
                            z(b, c.n_heads, c.max_states, c.d_key, c.d_key),
                            active, z(b, c.n_heads, c.max_states, c.d_route),
                            x.new_zeros(b, c.d_model, c.conv_kernel - 1),
                            0, c.max_states)

    # ---- routing -----------------------------------------------------------
    def _route(self, states: Tensor, factor: Tensor, active: Tensor,
               key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        """Frozen-routing argmin over {replace, fork} x active parents.

        Returns (parent (B,T,H), is_fork (B,T,H)). The fork arm uses the isotropic
        geometry ``lambda I`` and a per-fork cost, matching the reference objective.
        Forking is disabled when the pool is full.
        """
        B, T, H, K = key.shape
        M = states.shape[2]
        if factor.shape != (B, H, M, K, K):
            raise ValueError("factor must be (B,H,M,K,K)")
        pred = torch.einsum("bhmkv,bthk->btmhv", states, key)   # (B,T,M,H,V)
        r2 = (value.unsqueeze(2) - pred).square().sum(-1)        # (B,T,M,H)

        kk = key.unsqueeze(2).expand(B, T, M, H, K)
        fb = factor.permute(0, 2, 1, 3, 4).reshape(B * M, H, K, K)
        kb = kk.permute(0, 2, 3, 1, 4).reshape(B * M, T, H, K)
        u = _solve_keys(fb, kb).reshape(B, M, T, H, K).permute(0, 2, 1, 3, 4)
        h = (kk * u).sum(-1)                                    # (B,T,M,H)

        tau = self.config.tau
        cost_replace = r2 / (2 * (tau + h))                     # (B,T,M,H)
        # isotropic fork arm: h0 = ||k||^2 / lambda, and k is L2-normalised
        h0 = 1.0 / self.config.ridge
        cost_fork = r2 / (2 * (tau + h0)) + self.config.fork_cost

        vis = active.permute(0, 2, 1).view(B, 1, M, H).expand(B, T, M, H)
        has_free = (~active).any(-1).view(B, 1, 1, H)            # (B,1,1,H)
        cost_replace = cost_replace.masked_fill(~vis, float("inf"))
        cost_fork = cost_fork.masked_fill(~(vis & has_free), float("inf"))

        costs = torch.cat([cost_replace, cost_fork], dim=2)     # (B,T,2M,H)
        choice = costs.permute(0, 1, 3, 2).argmin(-1)           # (B,T,H)
        return choice % M, choice >= M

    # ---- forward -----------------------------------------------------------
    def forward(self, x: Tensor, cache: ChunkedCache | None = None):
        c = self.config
        if x.ndim != 3 or x.shape[-1] != c.d_model:
            raise ValueError("x must be [B,T,d_model]")
        B, T, _ = x.shape
        H, M = c.n_heads, c.max_states
        if cache is None:
            cache = self.initial_cache(x)
        dtype = cache.states.dtype

        conv_input = torch.cat([cache.conv_context, x.transpose(1, 2)], -1)
        local = x + F.silu(self.conv(conv_input).transpose(1, 2))
        next_context = conv_input[..., -(c.conv_kernel - 1):] if c.conv_kernel > 1 else conv_input[..., :0]

        q = F.normalize(self.q_proj(local).view(B, T, H, c.d_key).to(dtype), dim=-1, eps=1e-6)
        k = F.normalize(self.k_proj(local).view(B, T, H, c.d_key).to(dtype), dim=-1, eps=1e-6)
        v = torch.tanh(self.v_proj(local).view(B, T, H, c.d_value).to(dtype))
        rq = self.route_q(local).view(B, T, H, c.d_route).to(dtype)

        states, C, active, addresses = cache.states, cache.protection, cache.active, cache.addresses
        outs = []

        for start in range(0, T, c.chunk_len):
            end = min(T, start + c.chunk_len)
            kk, vv, qq, rr = (k[:, start:end], v[:, start:end],
                              q[:, start:end], rq[:, start:end])
            nl = self.null_route(local[:, start:end]).to(dtype).unsqueeze(-1)  # (B,L,H,1)
            if c.grad_checkpoint and self.training:
                out, states, C, active = checkpoint(
                    self._chunk_step, states, C, active, kk, vv, qq, rr, nl,
                    addresses, end, use_reentrant=False)
            else:
                out, states, C, active = self._chunk_step(
                    states, C, active, kk, vv, qq, rr, nl, addresses, end)
            outs.append(out)
            addresses = self.descriptor(
                states.permute(0, 2, 1, 4, 3).reshape(B * M, H, c.d_value, c.d_key)
            ).reshape(B, M, H, c.d_route).permute(0, 2, 1, 3)

        out = self.out_proj(torch.cat(outs, 1).reshape(B, T, H * c.d_value).to(local.dtype))
        new_cache = ChunkedCache(states, C, active, addresses, next_context,
                                 cache.step + T, c.max_states)
        stats = {
            "active_states": active.sum(dim=(-1, -2)).view(B, 1).expand(B, T).float(),
            "fork": torch.zeros(B, T, H, dtype=torch.bool, device=x.device),
            "parent": torch.zeros(B, T, H, dtype=torch.long, device=x.device),
            "destination": torch.zeros(B, T, H, dtype=torch.long, device=x.device),
            "historical_candidates": active.sum(dim=(-1, -2)).view(B, 1).expand(B, T).float(),
            "write_cost": torch.zeros(B, T, H, dtype=dtype, device=x.device),
        }
        return out, new_cache, stats

    def _chunk_step(self, bank, Cbank, active, kk, vv, qq, rr, nl, addresses, absolute_end):
        """One chunk: factorise, route, copy-on-write, write, read, update C."""
        c = self.config
        B, L, H = kk.shape[0], kk.shape[1], kk.shape[2]
        M = c.max_states
        dtype = bank.dtype
        eye = torch.eye(c.d_key, device=kk.device, dtype=dtype)
        Hm = c.ridge * eye + c.protect * Cbank
        fb = torch.linalg.cholesky(
            Hm.permute(0, 2, 1, 3, 4).reshape(B * M, H, c.d_key, c.d_key)
        )
        parent, is_fork = self._route(
            bank, fb.reshape(B, M, H, c.d_key, c.d_key).permute(0, 2, 1, 3, 4),
            active, kk, vv)
        snapshot = bank.clone()
        fork_count = torch.zeros(B, H, M, dtype=torch.float32, device=kk.device)
        fork_count.scatter_add_(2, parent.permute(0, 2, 1), is_fork.permute(0, 2, 1).float())
        onehot = F.one_hot(parent, M).permute(0, 3, 1, 2).to(dtype)
        Kb = kk.unsqueeze(2).expand(B, L, M, H, c.d_key)
        Vb = vv.unsqueeze(2).expand(B, L, M, H, c.d_value)
        Qb = qq.unsqueeze(2).expand(B, L, M, H, c.d_key)
        Sb = bank.permute(0, 2, 1, 3, 4)
        out_m, state_m = chunked_delta_update(
            Sb.reshape(B * M, H, c.d_key, c.d_value),
            Kb.permute(0, 2, 1, 3, 4).reshape(B * M, L, H, c.d_key),
            Vb.permute(0, 2, 1, 3, 4).reshape(B * M, L, H, c.d_value),
            Qb.permute(0, 2, 1, 3, 4).reshape(B * M, L, H, c.d_key),
            fb, c.tau, write_mask=onehot.reshape(B * M, L, H), return_states=True)
        out_m = out_m.reshape(B, M, L, H, c.d_value)
        new_bank = state_m.reshape(B, M, H, c.d_key, c.d_value).permute(0, 2, 1, 3, 4)
        idx = parent.unsqueeze(1).unsqueeze(-1).expand(B, 1, L, H, c.d_value)
        current_out = torch.gather(out_m, 1, idx).squeeze(1)

        logits = torch.einsum("bthr,bhmr->bthm", rr, addresses) * (c.d_route ** -0.5)
        vis = active.view(B, 1, H, M).expand(B, L, H, M)
        logits = logits.masked_fill(~vis, float("-inf"))
        weights = F.softmax(torch.cat([logits, nl], -1), -1)[..., :-1]
        hist = torch.einsum("bthm,bhmkv,bthk->bthv", weights, bank, qq)
        out = current_out + hist

        use = weights + F.one_hot(parent, M).to(dtype)
        gamma = c.protection_decay
        powers = gamma ** torch.arange(L - 1, -1, -1, device=kk.device, dtype=dtype)
        outer = torch.einsum("blhk,blhm->blhkm", qq, qq)
        wq = use.permute(0, 3, 2, 1) * powers.view(1, 1, 1, L)
        contrib = torch.einsum("bahl,blhkm->bahkm", wq, outer)
        new_C = gamma ** L * Cbank + (1 - gamma) * contrib.permute(0, 2, 1, 3, 4)

        free = ~active
        any_fork = fork_count > 0
        rank_free = torch.cumsum(free.to(torch.long), dim=-1) - 1
        rank_fork = torch.cumsum(any_fork.to(torch.long), dim=-1) - 1
        take = (rank_free.unsqueeze(-1) == rank_fork.unsqueeze(-2)) & free.unsqueeze(-2) & any_fork.unsqueeze(-1)
        slots = torch.arange(M, device=kk.device, dtype=torch.long)
        child = torch.where(take.any(-1), (take.to(torch.long) * slots).sum(-1),
                            torch.full_like(rank_fork, -1))
        has_child = child >= 0
        if bool(has_child.any()):
            src = snapshot.permute(0, 2, 1, 3, 4)
            dst = new_bank.permute(0, 2, 1, 3, 4)
            g5 = child.clamp(min=0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            dst.scatter_(1, g5.expand(B, 1, H, c.d_key, c.d_value),
                         torch.gather(src, 1, g5.expand(B, 1, H, c.d_key, c.d_value)))
            new_bank = dst.permute(0, 2, 1, 3, 4)
            Cb = new_C.permute(0, 2, 1, 3, 4)
            Cb.scatter_(1, g5.expand(B, 1, H, c.d_key, c.d_key),
                        torch.zeros(B, 1, H, c.d_key, c.d_key, dtype=dtype, device=kk.device))
            new_C = Cb.permute(0, 2, 1, 3, 4)
            act = active.permute(0, 2, 1)
            act.scatter_(1, child.clamp(min=0).unsqueeze(-1).expand(B, 1, H),
                         has_child.unsqueeze(-1).expand(B, 1, H))
            active = act.permute(0, 2, 1)
        return out, new_bank, new_C, active
