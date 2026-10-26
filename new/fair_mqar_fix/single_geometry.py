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

    def _directions(self, kk: Tensor, Cbank: Tensor) -> tuple[Tensor, dict]:
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
