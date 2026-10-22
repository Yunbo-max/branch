"""Branch Memory with a cached SPD factorization — an execution-only variant.

Motivation (measured; see README): the reference mixer calls ``torch.linalg.cholesky`` on
``H_j = ridge*I + protect*C_j`` for **every state, every token, every layer**. At K=64 that
is ``K^3/3`` flops, against ``2K^2`` for the solve that consumes it, so the factorization
dominates step time.

What this variant changes is *when the factorization happens*, not what is computed from it:

* ``C`` keeps its per-token rank-one update (exact, ``O(MK^2)``, no factorization);
* routing stays exact per token against the live state;
* addresses stay fresh;
* only the factorization of ``H`` is reused for ``solver_refresh`` consecutive tokens.

``solver_refresh == 1`` reproduces the reference bit-for-bit (asserted in
``tests/test_cached_solver.py``). ``solver_refresh > 1`` reuses a stale factor: that is an
approximation whose quality cost must be measured separately. It is **not** the
chunked-triangular rewrite and it does **not** remove the per-token sequential dependency.

The reference implementation under ``branch_memory/`` is left untouched.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

_BRANCH_DIR = Path(__file__).resolve().parents[1] / "branch_memory"
if str(_BRANCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BRANCH_DIR))

from branch_memory.cache import MemoryCache  # noqa: E402
from branch_memory.geometry import update_protection  # noqa: E402
from branch_memory.mixer import BranchMemoryMixer, MemoryConfig  # noqa: E402


def solve_with_factor(state: Tensor, key: Tensor, value: Tensor, factor: Tensor,
                      tau: float) -> tuple[Tensor, Tensor]:
    """Same arithmetic as ``branch_memory.math.correction_full``, factor supplied."""
    direction = torch.cholesky_solve(key.unsqueeze(-1), factor).squeeze(-1)
    denominator = tau + (key * direction).sum(-1)
    residual = value - torch.einsum("...vk,...k->...v", state, key)
    delta = residual.unsqueeze(-1) * direction.unsqueeze(-2) / denominator[..., None, None]
    cost = residual.square().sum(-1) / (2 * denominator)
    return delta, cost


def isotropic_solution(state: Tensor, key: Tensor, value: Tensor, ridge: float,
                       tau: float) -> tuple[Tensor, Tensor]:
    """Fork arm: ``H = ridge*I`` has the closed form used by the reference."""
    residual = value - torch.einsum("...vk,...k->...v", state, key)
    direction = key / ridge
    denominator = tau + (key * direction).sum(-1)
    delta = residual.unsqueeze(-1) * direction.unsqueeze(-2) / denominator[..., None, None]
    cost = residual.square().sum(-1) / (2 * denominator)
    return delta, cost


def choose_write_cached(states: Tensor, protection: Tensor, active: Tensor, key: Tensor,
                        value: Tensor, factor: Tensor, *, tau: float, ridge: float,
                        protect: float, fork_cost: float,
                        detach_protection: bool = True) -> dict:
    """``branch_memory.math.choose_write`` with a caller-supplied SPD factor."""
    if states.ndim != 5 or active.shape != states.shape[:3]:
        raise ValueError("states must be [B,H,M,V,K] and active [B,H,M]")
    if not active.any(-1).all():
        raise ValueError("at least one active state per batch/head is required")
    M, V, K = states.shape[-3:]
    C = protection.detach() if detach_protection else protection
    k, v = key.unsqueeze(2), value.unsqueeze(2)

    delta_r, cr = solve_with_factor(states, k, v, factor, tau)
    delta_f, cf = isotropic_solution(states, k, v, ridge, tau)
    cf = cf + fork_cost

    cr = cr.masked_fill(~active, float("inf"))
    has_free = (~active).any(-1)
    cf = cf.masked_fill(~(active & has_free.unsqueeze(-1)), float("inf"))
    costs = torch.cat([cr, cf], -1)
    choice = costs.detach().argmin(-1)
    fork = choice >= M
    parent = choice % M

    parent_index = parent[..., None, None, None].expand(*parent.shape, 1, V, K)
    base = states.gather(2, parent_index).squeeze(2)
    dr = delta_r.gather(2, parent_index).squeeze(2)
    df = delta_f.gather(2, parent_index).squeeze(2)
    current = base + torch.where(fork[..., None, None], df, dr)

    free_slot = (~active).to(torch.int64).argmax(-1)
    destination = torch.where(fork, free_slot, parent)
    mask = torch.arange(M, device=states.device) == destination[..., None]
    updated = torch.where(mask[..., None, None], current.unsqueeze(2), states)

    ix = parent[..., None, None, None].expand(*parent.shape, 1, K, K)
    inherited = protection.gather(2, ix).squeeze(2)
    next_c = torch.where(fork[..., None, None], torch.zeros_like(inherited), inherited)
    new_protection = torch.where(mask[..., None, None], next_c.unsqueeze(2), protection)

    new_active = active | mask
    cost = costs.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
    return dict(states=updated, protection=new_protection, active=new_active, current=current,
                parent=parent, destination=destination, fork=fork, cost=cost,
                replacement_cost=cr, branching_cost=cf)


@dataclass
class CachedSolverConfig(MemoryConfig):
    solver_refresh: int = 1     # 1 = reference behaviour (fresh Cholesky every token)

    def __post_init__(self):
        super().__post_init__()
        if self.solver_refresh < 1:
            raise ValueError("solver_refresh must be >= 1")
        if self.covariance != "full":
            raise ValueError("the cached-solver variant supports covariance='full' only")
        if self.mode != "branch":
            raise ValueError("the cached-solver variant supports mode='branch' only")


class CachedBranchMixer(BranchMemoryMixer):
    """Branch Memory whose SPD factorization is refreshed every ``solver_refresh`` steps."""

    def __init__(self, config: CachedSolverConfig) -> None:
        super().__init__(config)
        self.config: CachedSolverConfig = config
        self._factor: Tensor | None = None
        self._factor_key: tuple | None = None

    def forward(self, x: Tensor, cache: MemoryCache | None = None):
        c = self.config
        if x.ndim != 3 or x.shape[-1] != c.d_model or x.shape[1] < 1:
            raise ValueError("x must be [B,T,d_model], with T>=1")
        if not torch.isfinite(x).all():
            raise ValueError("non-finite input")
        B, T, _ = x.shape
        H = c.n_heads
        if cache is None:
            cache = self.initial_cache(x)
        M = c.max_states
        if cache.covariance != c.covariance or cache.protection.shape != (B, H, M, c.d_key, c.d_key):
            raise ValueError("cache covariance/protection mismatch")
        if cache.states.shape != (B, H, M, c.d_value, c.d_key):
            raise ValueError("cache state dimensions mismatch")

        conv_input = torch.cat([cache.conv_context, x.transpose(1, 2)], -1)
        local = x + F.silu(self.conv(conv_input).transpose(1, 2))
        next_context = conv_input[..., -(c.conv_kernel - 1):] if c.conv_kernel > 1 else conv_input[..., :0]
        dtype = cache.states.dtype

        q = F.normalize(self.q_proj(local).view(B, T, H, c.d_key).to(dtype), dim=-1, eps=1e-6)
        k = F.normalize(self.k_proj(local).view(B, T, H, c.d_key).to(dtype), dim=-1, eps=1e-6)
        v = torch.tanh(self.v_proj(local).view(B, T, H, c.d_value).to(dtype))
        rq = self.route_q(local).view(B, T, H, c.d_route).to(dtype)

        states, C, active = cache.states, cache.protection, cache.active
        addresses, current = cache.addresses, cache.current
        eye = torch.eye(c.d_key, device=x.device, dtype=dtype)

        ys = []
        trace = {z: [] for z in ["parent", "destination", "fork", "active_states",
                                 "historical_candidates", "write_cost"]}
        for t in range(T):
            qt, kt, vt = q[:, t], k[:, t], v[:, t]
            absolute = cache.step + t

            key_tag = (B, H, M, c.d_key)
            refresh = (self._factor is None
                       or self._factor_key != key_tag
                       or absolute % c.solver_refresh == 0)
            if refresh:
                self._factor = torch.linalg.cholesky(c.ridge * eye + c.protect * C)
                self._factor_key = key_tag

            w = choose_write_cached(
                states, C, active, kt, vt, self._factor,
                tau=c.tau, ridge=c.ridge, protect=c.protect, fork_cost=c.fork_cost,
                detach_protection=not c.differentiate_covariance,
            )
            states, C, active, current = w["states"], w["protection"], w["active"], w["current"]
            parent, destination, fork, cost = w["parent"], w["destination"], w["fork"], w["cost"]
            dest_mask = torch.arange(M, device=x.device) == destination[..., None]
            addresses = torch.where(dest_mask[..., None],
                                    self.descriptor(current).unsqueeze(2), addresses)
            visible = active & ~dest_mask

            current_out = torch.einsum("bhvk,bhk->bhv", current, qt)
            hist, weights = self.historical_read(qt, rq[:, t], local[:, t], states,
                                                 addresses, visible)
            ys.append(current_out + hist)
            trace["historical_candidates"].append(visible.sum(-1).detach())

            use = weights + dest_mask.to(dtype)
            C = update_protection(C, qt, use, active, decay=c.protection_decay,
                                  covariance=c.covariance,
                                  detach=not c.differentiate_covariance)
            trace["parent"].append(parent.detach())
            trace["destination"].append(destination.detach())
            trace["fork"].append(fork.detach())
            trace["active_states"].append(active.sum(-1).detach())
            trace["write_cost"].append(cost.detach())

        output = self.out_proj(torch.stack(ys, 1).reshape(B, T, H * c.d_value).to(local.dtype))
        new_cache = MemoryCache(states, C, active, addresses, current, next_context,
                                cache.step + T, c.covariance)
        stats = {name: torch.stack(values, 1) for name, values in trace.items()}
        return output, new_cache, stats
