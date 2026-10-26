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

Fork semantics (explicit definition):

* a fork child copies the selected parent's state at the chunk start;
* the child starts with zero protection, so both its routing cost and its update use the
  isotropic geometry ``ridge * I``; the parent keeps its protected geometry;
* all tokens that chose fork write to the child, all replace tokens write to the parent,
  and the parent keeps its chunk-start value when it has no replace tokens;
* one child per forking parent, assigned to distinct free slots; if there are more
  forking parents than free slots, the unassigned forks fall back to replace.

The child is not visible to reads inside the chunk in which it is created (frozen
routing); it becomes active at the next chunk boundary.
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
    covariance: str = "full"
    differentiate_covariance: bool = True
    chunk_len: int = 64
    grad_checkpoint: bool = False

    def __post_init__(self) -> None:
        for name in (
            "d_model",
            "n_heads",
            "d_key",
            "d_value",
            "d_route",
            "n_probes",
            "max_states",
            "conv_kernel",
            "chunk_len",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.tau <= 0 or self.ridge <= 0:
            raise ValueError("tau and ridge must be positive")
        if not 0 <= self.protection_decay < 1:
            raise ValueError("protection_decay must be in [0,1)")
        if self.covariance not in ("full", "diagonal"):
            raise ValueError("covariance must be 'full' or 'diagonal'")
        if self.protect < 0 or self.fork_cost < 0:
            raise ValueError("protect and fork_cost must be non-negative")


@dataclass
class ChunkedCache:
    states: Tensor  # B H M K V
    protection: Tensor  # B H M K K (frozen at chunk start, used for directions)
    active: Tensor  # B H M
    addresses: Tensor  # B H M R
    conv_context: Tensor
    chunk_start_states: Tensor  # B H M K V (S0 for the chunk-start read)
    chunk_protection: Tensor  # B H M K K (running C, committed at chunk boundary)
    chunk_factor: Tensor  # B H K K (frozen Cholesky factor, single_geo full mode)
    chunk_pos: int = 0  # tokens consumed in the current chunk [0, chunk_len)
    step: int = 0
    max_states: int = 0

    def detach(self) -> "ChunkedCache":
        return ChunkedCache(
            self.states.detach(),
            self.protection.detach(),
            self.active.detach(),
            self.addresses.detach(),
            self.conv_context.detach(),
            self.chunk_start_states.detach(),
            self.chunk_protection.detach(),
            self.chunk_factor.detach(),
            self.chunk_pos,
            self.step,
            self.max_states,
        )

    def allocated_tensor_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.states,
                self.protection,
                self.active,
                self.addresses,
                self.conv_context,
                self.chunk_start_states,
                self.chunk_protection,
                self.chunk_factor,
            )
        )


class ChunkedBranchMixer(BranchMemoryMixer):
    """Branch Memory with chunk-level factorization, routing and write."""

    def __init__(self, config: ChunkedBranchConfig) -> None:
        base = type(
            "_Base",
            (),
            dict(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_key=config.d_key,
                d_value=config.d_value,
                d_route=config.d_route,
                n_probes=config.n_probes,
                max_states=config.max_states,
                mode="branch",
                snapshot_interval=4,
                conv_kernel=config.conv_kernel,
                tau=config.tau,
                ridge=config.ridge,
                protect=config.protect,
                fork_cost=config.fork_cost,
                protection_decay=config.protection_decay,
                covariance=config.covariance,
                differentiate_covariance=config.differentiate_covariance,
            ),
        )()
        super().__init__(base)
        self.config = config

    # ---- cache -------------------------------------------------------------
    def initial_cache(self, x: Tensor) -> ChunkedCache:
        c = self.config
        b = x.shape[0]
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        z = lambda *s: torch.zeros(*s, device=x.device, dtype=dtype)
        active = torch.zeros(
            b, c.n_heads, c.max_states, device=x.device, dtype=torch.bool
        )
        active[..., 0] = True
        return ChunkedCache(
            z(b, c.n_heads, c.max_states, c.d_key, c.d_value),
            z(b, c.n_heads, c.max_states, c.d_key, c.d_key),
            active,
            z(b, c.n_heads, c.max_states, c.d_route),
            x.new_zeros(b, c.d_model, c.conv_kernel - 1),
            z(b, c.n_heads, c.max_states, c.d_key, c.d_value),
            z(b, c.n_heads, c.max_states, c.d_key, c.d_key),
            z(b, c.n_heads, c.d_key, c.d_key),
            0,
            0,
            c.max_states,
        )

    # ---- routing -----------------------------------------------------------
    def _route(
        self, states: Tensor, factor: Tensor, active: Tensor, key: Tensor, value: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Frozen-routing argmin over {replace, fork} x active parents.

        Returns (parent (B,T,H), is_fork (B,T,H), chosen_cost (B,T,H)). The fork arm
        uses the isotropic geometry ``lambda I`` and a per-fork cost, which matches the
        defined child semantics (a child starts with zero protection). Forking is
        disabled when the pool is full.
        """
        B, T, H, K = key.shape
        M = states.shape[2]
        if factor.shape != (B, H, M, K, K):
            raise ValueError("factor must be (B,H,M,K,K)")
        pred = torch.einsum("bhmkv,bthk->btmhv", states, key)  # (B,T,M,H,V)
        r2 = (value.unsqueeze(2) - pred).square().sum(-1)  # (B,T,M,H)

        kk = key.unsqueeze(2).expand(B, T, M, H, K)
        fb = factor.permute(0, 2, 1, 3, 4).reshape(B * M, H, K, K)
        kb = kk.permute(0, 2, 1, 3, 4).reshape(B * M, T, H, K)
        u = _solve_keys(fb, kb).reshape(B, M, T, H, K).permute(0, 2, 1, 3, 4)
        h = (kk * u).sum(-1)  # (B,T,M,H)

        tau = self.config.tau
        cost_replace = r2 / (2 * (tau + h))  # (B,T,M,H)
        # isotropic fork arm: h0 = ||k||^2 / lambda, and k is L2-normalised
        h0 = 1.0 / self.config.ridge
        cost_fork = r2 / (2 * (tau + h0)) + self.config.fork_cost

        vis = active.permute(0, 2, 1).view(B, 1, M, H).expand(B, T, M, H)
        has_free = (~active).any(-1).view(B, 1, 1, H)  # (B,1,1,H)
        cost_replace = cost_replace.masked_fill(~vis, float("inf"))
        cost_fork = cost_fork.masked_fill(~(vis & has_free), float("inf"))

        costs = torch.cat([cost_replace, cost_fork], dim=2)  # (B,T,2M,H)
        costs = costs.permute(0, 1, 3, 2)  # (B,T,H,2M)
        choice = costs.argmin(-1)  # (B,T,H)
        return choice % M, choice >= M, costs

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
        next_context = (
            conv_input[..., -(c.conv_kernel - 1) :]
            if c.conv_kernel > 1
            else conv_input[..., :0]
        )

        q = F.normalize(
            self.q_proj(local).view(B, T, H, c.d_key).to(dtype), dim=-1, eps=1e-6
        )
        k = F.normalize(
            self.k_proj(local).view(B, T, H, c.d_key).to(dtype), dim=-1, eps=1e-6
        )
        v = torch.tanh(self.v_proj(local).view(B, T, H, c.d_value).to(dtype))
        rq = self.route_q(local).view(B, T, H, c.d_route).to(dtype)

        states, C, active, addresses = (
            cache.states,
            cache.protection,
            cache.active,
            cache.addresses,
        )
        outs = []
        chunk_stats = []

        for start in range(0, T, c.chunk_len):
            end = min(T, start + c.chunk_len)
            kk, vv, qq, rr = (
                k[:, start:end],
                v[:, start:end],
                q[:, start:end],
                rq[:, start:end],
            )
            nl = (
                self.null_route(local[:, start:end]).to(dtype).unsqueeze(-1)
            )  # (B,L,H,1)
            if c.grad_checkpoint and self.training:
                out, states, C, active, telemetry = checkpoint(
                    self._chunk_step,
                    states,
                    C,
                    active,
                    kk,
                    vv,
                    qq,
                    rr,
                    nl,
                    addresses,
                    end,
                    use_reentrant=False,
                )
            else:
                out, states, C, active, telemetry = self._chunk_step(
                    states, C, active, kk, vv, qq, rr, nl, addresses, end
                )
            outs.append(out)
            chunk_stats.append(telemetry)
            addresses = (
                self.descriptor(
                    states.permute(0, 2, 1, 4, 3).reshape(B * M, H, c.d_value, c.d_key)
                )
                .reshape(B, M, H, c.d_route)
                .permute(0, 2, 1, 3)
            )

        out = self.out_proj(
            torch.cat(outs, 1).reshape(B, T, H * c.d_value).to(local.dtype)
        )
        new_cache = ChunkedCache(
            states,
            C,
            active,
            addresses,
            next_context,
            states,
            C,
            torch.zeros(B, H, c.d_key, c.d_key, device=x.device, dtype=dtype),
            0,
            cache.step + T,
            c.max_states,
        )
        fork, parent, destination, cost, active_count = [
            torch.cat(parts, 1) for parts in zip(*chunk_stats)
        ]
        stats = {
            "active_states": active_count,
            "fork": fork,
            "parent": parent,
            "destination": destination,
            "historical_candidates": active_count,
            "write_cost": cost,
        }
        return out, new_cache, stats

    @staticmethod
    def _allocate_children(
        active: Tensor, any_fork: Tensor, first_request: Tensor | None = None
    ) -> Tensor:
        """Assign free slots to forking parents by their *first request time*.

        An earlier fork must not lose a slot when a later token requests a
        lower-numbered parent. Ties are broken by parent slot index. The
        allocation can be computed for a whole chunk because, for each
        prefix, its allocations are identical to online first-come-first-served
        processing. Children become readable only at the next chunk boundary.

        first_request=None retains parent-index ordering for direct legacy
        callers; the actual chunk writer always passes the first-event times.
        """
        B, H, M = active.shape
        if any_fork.shape != active.shape:
            raise ValueError("any_fork must match active")
        slots = torch.arange(M, device=active.device, dtype=torch.long).view(1, 1, M)
        free = ~active
        rank_free = torch.where(
            free,
            torch.cumsum(free.to(torch.long), -1) - 1,
            torch.full_like(active, -1, dtype=torch.long),
        )
        if first_request is None:
            rank_fork = torch.where(
                any_fork,
                torch.cumsum(any_fork.to(torch.long), -1) - 1,
                torch.full_like(active, -1, dtype=torch.long),
            )
        else:
            if first_request.shape != active.shape:
                raise ValueError("first_request must match active")
            # Deterministic lexicographic rank: (first_token, parent_slot).
            priority = first_request.to(torch.long) * M + slots
            order = priority.argsort(dim=-1)
            rank_by_parent = torch.empty_like(order).scatter_(
                -1, order, slots.expand(B, H, M)
            )
            rank_fork = torch.where(
                any_fork,
                rank_by_parent,
                torch.full_like(active, -1, dtype=torch.long),
            )

        match = (
            (rank_fork.unsqueeze(-1) == rank_free.unsqueeze(-2))
            & any_fork.unsqueeze(-1)
            & free.unsqueeze(-2)
        )
        return torch.where(
            match.any(-1),
            (match.to(torch.long) * slots).sum(-1),
            torch.full_like(rank_fork, -1),
        )

    def _geometry_factor(self, Cbank: Tensor) -> Tensor:
        """Cholesky factor of ``ridge I + protect C`` per slot (B,M,H,K,K)."""
        c = self.config
        B, H, M, K, _ = Cbank.shape
        eye = torch.eye(K, device=Cbank.device, dtype=Cbank.dtype)
        if c.covariance == "diagonal":
            C = torch.diag_embed(Cbank.diagonal(dim1=-2, dim2=-1))
        else:
            C = Cbank
        Hm = c.ridge * eye + c.protect * C
        return torch.linalg.cholesky(
            Hm.permute(0, 2, 1, 3, 4).reshape(B * M, H, K, K)
        ).reshape(B, M, H, K, K)

    def _chunk_step(
        self, bank, Cbank, active, kk, vv, qq, rr, nl, addresses, absolute_end
    ):
        """One chunk: factorise, route, copy-on-write, write, read, update C."""
        c = self.config
        B, L, H = kk.shape[0], kk.shape[1], kk.shape[2]
        M = c.max_states
        dtype = bank.dtype
        # Keep the geometry and recurrent state in FP32 even under BF16 autocast.
        kk = kk.to(dtype)
        vv = vv.to(dtype)
        qq = qq.to(dtype)
        rr = rr.to(dtype)
        nl = nl.to(dtype)
        addresses = addresses.to(dtype)
        Cbank = Cbank if c.differentiate_covariance else Cbank.detach()
        fb = self._geometry_factor(Cbank)  # (B,M,H,K,K)
        parent, is_fork, costs = self._route(
            bank, fb.permute(0, 2, 1, 3, 4), active, kk, vv
        )
        snapshot = bank.clone()
        # First-come-first-served fork allocation is causal even if the whole
        # chunk is processed in parallel. A later parent's first request cannot
        # preempt an earlier request merely because its slot index is smaller.
        fork_event = (
            F.one_hot(parent.permute(0, 2, 1), M).bool()
            & is_fork.permute(0, 2, 1).unsqueeze(-1)
        )  # (B,H,L,M)
        first_request = torch.where(
            fork_event,
            torch.arange(L, device=kk.device).view(1, 1, L, 1),
            L,
        ).amin(dim=2)  # (B,H,M), L means no request
        child = self._allocate_children(
            active, first_request < L, first_request=first_request
        )  # (B,H,M)
        has_child = child >= 0
        child_tok = torch.gather(child, 2, parent.permute(0, 2, 1)).permute(0, 2, 1)
        is_fork = is_fork & (child_tok >= 0)
        destination = torch.where(is_fork, child_tok, parent)
        replace_cost = costs[..., :M].gather(-1, parent.unsqueeze(-1)).squeeze(-1)
        fork_cost = costs[..., M:].gather(-1, parent.unsqueeze(-1)).squeeze(-1)
        chosen_cost = torch.where(is_fork, fork_cost, replace_cost)

        # copy-on-write: child <- parent state at chunk start; protection starts at 0
        src = snapshot.permute(0, 2, 1, 3, 4)  # (B,M,H,K,V)
        dst = bank.permute(0, 2, 1, 3, 4).clone()
        C0 = Cbank.permute(0, 2, 1, 3, 4).clone()
        onehot_child = F.one_hot(child.clamp(min=0), M).to(dtype) * has_child.unsqueeze(
            -1
        ).to(dtype)
        dest_src = (
            (onehot_child * torch.arange(M, device=kk.device).view(1, 1, M, 1))
            .sum(2)
            .to(torch.long)
        )  # (B,H,Md)
        dest_valid = onehot_child.sum(2) > 0  # (B,H,Md)
        mask_v = (
            dest_valid.permute(0, 2, 1)
            .view(B, M, H, 1, 1)
            .expand(B, M, H, c.d_key, c.d_value)
        )
        idx_v = (
            dest_src.permute(0, 2, 1)
            .clamp(min=0)
            .view(B, M, H, 1, 1)
            .expand(B, M, H, c.d_key, c.d_value)
        )
        dst = torch.where(mask_v, torch.gather(src, 1, idx_v), dst)
        mask_c = (
            dest_valid.permute(0, 2, 1)
            .view(B, M, H, 1, 1)
            .expand(B, M, H, c.d_key, c.d_key)
        )
        C0 = C0.masked_fill(mask_c, 0.0)
        bank0 = dst.permute(0, 2, 1, 3, 4)
        C0 = C0.permute(0, 2, 1, 3, 4)

        iso = torch.linalg.cholesky(
            c.ridge * torch.eye(c.d_key, device=kk.device, dtype=dtype)
        )
        fb_used = torch.where(
            mask_c,
            iso.view(1, 1, 1, c.d_key, c.d_key).expand(B, M, H, c.d_key, c.d_key),
            fb,
        )

        dest_onehot = F.one_hot(destination, M).permute(0, 3, 1, 2).to(dtype)
        Kb = kk.unsqueeze(2).expand(B, L, M, H, c.d_key)
        Vb = vv.unsqueeze(2).expand(B, L, M, H, c.d_value)
        Qb = qq.unsqueeze(2).expand(B, L, M, H, c.d_key)
        Sb = bank0.permute(0, 2, 1, 3, 4)
        out_m, state_m = chunked_delta_update(
            Sb.reshape(B * M, H, c.d_key, c.d_value),
            Kb.permute(0, 2, 1, 3, 4).reshape(B * M, L, H, c.d_key),
            Vb.permute(0, 2, 1, 3, 4).reshape(B * M, L, H, c.d_value),
            Qb.permute(0, 2, 1, 3, 4).reshape(B * M, L, H, c.d_key),
            fb_used.reshape(B * M, H, c.d_key, c.d_key),
            c.tau,
            write_mask=dest_onehot.reshape(B * M, L, H),
            return_states=True,
        )
        out_m = out_m.reshape(B, M, L, H, c.d_value)
        new_bank = state_m.reshape(B, M, H, c.d_key, c.d_value).permute(0, 2, 1, 3, 4)
        idx = destination.unsqueeze(1).unsqueeze(-1).expand(B, 1, L, H, c.d_value)
        current_out = torch.gather(out_m, 1, idx).squeeze(1)

        logits = torch.einsum("bthr,bhmr->bthm", rr, addresses) * (c.d_route**-0.5)
        vis = active.view(B, 1, H, M).expand(B, L, H, M)
        logits = logits.masked_fill(~vis, float("-inf"))
        weights = F.softmax(torch.cat([logits, nl], -1), -1)[..., :-1]
        hist = torch.einsum("bthm,bhmkv,bthk->bthv", weights, bank, qq)
        out = current_out + hist

        use = weights + F.one_hot(destination, M).to(dtype)
        gamma = c.protection_decay
        powers = gamma ** torch.arange(L - 1, -1, -1, device=kk.device, dtype=dtype)
        outer = torch.einsum("blhk,blhm->blhkm", qq, qq)
        wq = use.permute(0, 3, 2, 1) * powers.view(1, 1, 1, L)
        contrib = torch.einsum("bahl,blhkm->bahkm", wq, outer)
        new_C = gamma**L * C0 + (1 - gamma) * contrib.permute(0, 2, 1, 3, 4)
        if c.covariance == "diagonal":
            new_C = torch.diag_embed(new_C.diagonal(dim1=-2, dim2=-1))

        active_new = (active.permute(0, 2, 1) | dest_valid.permute(0, 2, 1)).permute(
            0, 2, 1
        )
        active_count = active.sum(dim=(-1, -2)).view(B, 1).expand(B, L).float()
        telemetry = (is_fork, parent, destination, chosen_cost, active_count)
        return out, new_bank, new_C, active_new, telemetry
