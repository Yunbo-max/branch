"""Single-state SG-Delta geometry mixer (M=1).

This is the controlled experiment arm: the same projections, convolution, chunk-start
read, and covariance EMA as the corrected single-state Branch mixer, but the write
direction is chosen among:

* ``isotropic``: a = k / (tau*lambda + ||k||^2), C is not used or updated
* ``rayleigh``:  a = k / (tau*(lambda + mu*c) + ||k||^2), c = k^T C k / ||k||^2
* ``full``:      a = H^{-1}k / (tau + k^T H^{-1}k), H = lambda I + mu C
* ``certified``: group-level choice between rayleigh and full, using the certificate U

The state update is the exact triangular recurrence with externally supplied directions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from chunked_branch import ChunkedBranchConfig, ChunkedBranchMixer, ChunkedCache
from chunked_delta import _solve_keys, chunked_delta_from_directions
from geometry_certificate import rayleigh_certificate


@dataclass
class SingleGeometryConfig(ChunkedBranchConfig):
    geometry_mode: str = "full"
    geometry_threshold: float = 0.1
    read_mode: str = "chunk_start"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.geometry_mode not in ("isotropic", "rayleigh", "full", "certified"):
            raise ValueError("unknown geometry_mode")
        if self.geometry_threshold < 0:
            raise ValueError("geometry_threshold must be non-negative")
        if self.read_mode not in ("chunk_start", "current_only"):
            raise ValueError("read_mode must be 'chunk_start' or 'current_only'")
        if self.max_states != 1:
            raise ValueError("single_geometry requires max_states=1")


class SingleGeometryMixer(ChunkedBranchMixer):
    def __init__(self, config: SingleGeometryConfig) -> None:
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
                max_states=1,
                mode="branch",
                snapshot_interval=4,
                conv_kernel=config.conv_kernel,
                tau=config.tau,
                ridge=config.ridge,
                protect=config.protect,
                fork_cost=0.0,
                protection_decay=config.protection_decay,
                covariance="full",
                differentiate_covariance=config.differentiate_covariance,
            ),
        )()
        super().__init__(base)
        self.config = config

    def _directions(
        self, kk: Tensor, Cbank: Tensor, factor: Tensor | None = None
    ) -> tuple[Tensor, dict]:
        c = self.config
        B, L, H, _ = kk.shape
        C = Cbank[:, :, 0]  # (B,H,K,K)
        if not c.differentiate_covariance:
            C = C.detach()

        if c.geometry_mode == "isotropic":
            k2 = kk.square().sum(-1, keepdim=True)
            a = kk / (c.tau * c.ridge + k2)
            return a, {"mode": "isotropic"}

        if c.geometry_mode == "rayleigh":
            a_ray, upper, _, _ = rayleigh_certificate(
                kk, C.unsqueeze(1), lam=c.ridge, mu=c.protect, tau=c.tau
            )
            return a_ray, {"mode": "rayleigh", "upper": upper}

        if factor is None:
            eye = torch.eye(c.d_key, device=kk.device, dtype=kk.dtype)
            H = c.ridge * eye + c.protect * C
            factor = torch.linalg.cholesky(H)
        u = _solve_keys(factor, kk)
        a_full = u / (c.tau + (kk * u).sum(-1, keepdim=True))

        if c.geometry_mode == "full":
            return a_full, {"mode": "full"}

        a_ray, upper, _, _ = rayleigh_certificate(
            kk, C.unsqueeze(1), lam=c.ridge, mu=c.protect, tau=c.tau
        )
        easy = upper.max(dim=1).values <= c.geometry_threshold  # (B,H)
        a = torch.where(easy[:, None, :, None], a_ray, a_full)
        return a, {
            "mode": "certified",
            "groupId": easy,
            "upper": upper,
        }

    def _make_factor(self, C_frozen: Tensor) -> Tensor:
        """Frozen Cholesky factor of H = ridge I + protect C at a chunk boundary (full only)."""
        c = self.config
        C = C_frozen[:, :, 0]  # (B,H,K,K)
        if c.geometry_mode != "full":
            return torch.zeros_like(C)
        eye = torch.eye(c.d_key, device=C.device, dtype=C.dtype)
        H = c.ridge * eye + c.protect * C
        return torch.linalg.cholesky(H)  # (B,H,K,K)

    def _refresh_addresses(self, states: Tensor) -> Tensor:
        c = self.config
        B, M = states.shape[0], states.shape[2]
        return (
            self.descriptor(
                states.permute(0, 2, 1, 4, 3).reshape(
                    B * M, c.n_heads, c.d_value, c.d_key
                )
            )
            .reshape(B, M, c.n_heads, c.d_route)
            .permute(0, 2, 1, 3)
        )

    def _token_step(
        self,
        states,
        S0,
        C_frozen,
        C_run,
        active,
        factor,
        kk,
        vv,
        qq,
        rr,
        loc,
        addresses,
    ):
        """One streaming token inside the current chunk: frozen geometry, deferred C commit.

        The write direction uses the chunk-start geometry ``C_frozen`` (identical to the
        batched ``_chunk_step``), the chunk-start read uses ``S0``, and ``C_run`` is
        accumulated so the protection matrix is only committed at the chunk boundary.
        """
        c = self.config
        B, L, H = kk.shape[0], kk.shape[1], kk.shape[2]
        dtype = states.dtype
        kk = kk.to(dtype)
        vv = vv.to(dtype)
        qq = qq.to(dtype)
        rr = rr.to(dtype)
        addresses = addresses.to(dtype)
        C_frozen_used = C_frozen if c.differentiate_covariance else C_frozen.detach()

        a, _ = self._directions(kk, C_frozen_used, factor)
        S = states[:, :, 0]  # (B,H,K,V)
        out_m, state_m = chunked_delta_from_directions(
            S, kk, vv, qq, a, write_mask=None, return_states=True
        )
        new_states = state_m.unsqueeze(2)

        need_weights = (c.read_mode == "chunk_start") or (
            c.geometry_mode != "isotropic"
        )
        weights = None
        if need_weights:
            logits = torch.einsum("bthr,bhmr->bthm", rr, addresses) * (c.d_route**-0.5)
            vis = active.view(B, 1, H, 1).expand(B, L, H, 1)
            logits = logits.masked_fill(~vis, float("-inf"))
            nl = self.null_route(loc).to(dtype).unsqueeze(-1)
            weights = F.softmax(torch.cat([logits, nl], -1), -1)[..., :-1]

        if c.read_mode == "chunk_start":
            hist = torch.einsum("bthm,bhmkv,bthk->bthv", weights, S0, qq)
            out = out_m + hist
        else:
            out = out_m

        if c.geometry_mode != "isotropic":
            use = weights + torch.ones_like(weights)
            gamma = c.protection_decay
            # L=1 token: match the batched EMA exactly (powers == 1 for a single step).
            w = use[..., 0]  # (B,L,H)
            qw = qq * w.unsqueeze(-1)  # (B,L,H,K)
            contrib = torch.einsum("blhk,blhj->bhkj", qw, qq)  # (B,H,K,J)
            C_run = gamma * C_run + (1 - gamma) * contrib.unsqueeze(2)

        return out, new_states, C_run

    def forward(self, x: Tensor, cache: ChunkedCache | None = None):
        c = self.config
        if x.ndim != 3 or x.shape[-1] != c.d_model:
            raise ValueError("x must be [B,T,d_model]")
        if c.geometry_mode == "certified":
            raise ValueError(
                "geometry_mode='certified' is a chunk-level (non-causal) decision and is "
                "not supported for autoregressive language modeling. It remains available "
                "only through _directions() for offline numerical analysis."
            )
        B, T, _ = x.shape
        H = c.n_heads
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

        states = cache.states
        C_frozen = cache.protection
        C_run = cache.chunk_protection
        active = cache.active
        addresses = cache.addresses
        S0 = cache.chunk_start_states
        factor = cache.chunk_factor
        pos = cache.chunk_pos
        if pos == 0:
            # At a chunk boundary (or a fresh cache) the frozen factor must match the
            # committed protection matrix; initial_cache() only zero-fills chunk_factor.
            factor = self._make_factor(C_frozen)

        outs = []
        chunk_stats = []
        t = 0
        while t < T:
            room = c.chunk_len - pos
            take = min(room, T - t)
            if take == c.chunk_len:
                kk = k[:, t : t + take]
                vv = v[:, t : t + take]
                qq = q[:, t : t + take]
                rr = rq[:, t : t + take]
                nl = self.null_route(local[:, t : t + take]).to(dtype).unsqueeze(-1)
                if c.grad_checkpoint and self.training:
                    out, states, newC, active, telemetry = checkpoint(
                        self._chunk_step,
                        states,
                        C_frozen,
                        active,
                        kk,
                        vv,
                        qq,
                        rr,
                        nl,
                        addresses,
                        t + take,
                        use_reentrant=False,
                    )
                else:
                    out, states, newC, active, telemetry = self._chunk_step(
                        states,
                        C_frozen,
                        active,
                        kk,
                        vv,
                        qq,
                        rr,
                        nl,
                        addresses,
                        t + take,
                    )
                outs.append(out)
                chunk_stats.append(telemetry)
                C_frozen = newC
                C_run = newC
                S0 = states
                addresses = self._refresh_addresses(states)
                factor = self._make_factor(C_frozen)
                pos = 0
                t += take
            else:
                for _ in range(take):
                    out, states, C_run = self._token_step(
                        states,
                        S0,
                        C_frozen,
                        C_run,
                        active,
                        factor,
                        k[:, t : t + 1],
                        v[:, t : t + 1],
                        q[:, t : t + 1],
                        rq[:, t : t + 1],
                        local[:, t : t + 1],
                        addresses,
                    )
                    outs.append(out)
                    zeros_b = torch.zeros(B, 1, H, dtype=torch.bool, device=x.device)
                    zeros_l = torch.zeros(B, 1, H, dtype=torch.long, device=x.device)
                    zeros_c = torch.zeros(B, 1, H, dtype=dtype, device=x.device)
                    act = active.sum(dim=(-1, -2)).view(B, 1).expand(B, 1).float()
                    chunk_stats.append((zeros_b, zeros_l, zeros_l, zeros_c, act))
                    t += 1
                    pos += 1
                    if pos == c.chunk_len:
                        C_frozen = C_run
                        S0 = states
                        addresses = self._refresh_addresses(states)
                        factor = self._make_factor(C_frozen)
                        pos = 0

        out = self.out_proj(
            torch.cat(outs, 1).reshape(B, T, H * c.d_value).to(local.dtype)
        )
        new_cache = ChunkedCache(
            states,
            C_frozen,
            active,
            addresses,
            next_context,
            S0,
            C_run,
            factor,
            pos,
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

    def _chunk_step(
        self, bank, Cbank, active, kk, vv, qq, rr, nl, addresses, absolute_end
    ):
        c = self.config
        B, L, H = kk.shape[0], kk.shape[1], kk.shape[2]
        dtype = bank.dtype
        # Keep geometry/state in FP32 under BF16 autocast.
        kk = kk.to(dtype)
        vv = vv.to(dtype)
        qq = qq.to(dtype)
        rr = rr.to(dtype)
        nl = nl.to(dtype)
        addresses = addresses.to(dtype)
        if c.max_states != 1:
            raise ValueError("single_geometry is M=1 only")

        Cbank_used = Cbank if c.differentiate_covariance else Cbank.detach()
        a, telemetry = self._directions(kk, Cbank_used)
        S0 = bank[:, :, 0]
        out_m, state_m = chunked_delta_from_directions(
            S0, kk, vv, qq, a, write_mask=None, return_states=True
        )
        new_bank = state_m.unsqueeze(2)

        need_weights = (c.read_mode == "chunk_start") or (
            c.geometry_mode != "isotropic"
        )
        weights = None
        if need_weights:
            logits = torch.einsum("bthr,bhmr->bthm", rr, addresses) * (c.d_route**-0.5)
            vis = active.view(B, 1, H, 1).expand(B, L, H, 1)
            logits = logits.masked_fill(~vis, float("-inf"))
            weights = F.softmax(torch.cat([logits, nl], -1), -1)[..., :-1]
        if c.read_mode == "chunk_start":
            hist = torch.einsum("bthm,bhmkv,bthk->bthv", weights, bank, qq)
            out = out_m + hist
        else:
            out = out_m

        if c.geometry_mode == "isotropic":
            new_C = Cbank_used
        else:
            use = weights + torch.ones(B, L, H, 1, dtype=dtype, device=kk.device)
            gamma = c.protection_decay
            powers = gamma ** torch.arange(L - 1, -1, -1, device=kk.device, dtype=dtype)
            # M=1: sum_t w_t q_t q_t^T without materialising [L,K,K].
            w = use[..., 0] * powers.view(1, L, 1)
            qw = qq * w.unsqueeze(-1)
            contrib = torch.einsum("blhk,blhj->bhkj", qw, qq)
            new_C = gamma**L * Cbank_used + (1 - gamma) * contrib.unsqueeze(2)

        zeros_bool = torch.zeros(B, L, H, dtype=torch.bool, device=kk.device)
        zeros_long = torch.zeros(B, L, H, dtype=torch.long, device=kk.device)
        zeros_cost = torch.zeros(B, L, H, dtype=dtype, device=kk.device)
        active_count = active.sum(dim=(-1, -2)).view(B, 1).expand(B, L).float()
        telemetry = (zeros_bool, zeros_long, zeros_long, zeros_cost, active_count)
        return out, new_bank, new_C, active, telemetry
