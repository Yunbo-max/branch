"""Conditional closed-form writes, state orientation [..., value, key].

Full geometry uses every entry of H through a checked Cholesky solve. The legacy
`correction` diagonal solver is retained only as an explicit ablation/reference.
Hard action choice has no derivative across switches; selected writes do.
"""
from dataclasses import dataclass
import math
import torch
from torch import Tensor


def _working_dtype(*tensors: Tensor) -> torch.dtype:
    if any(not x.is_floating_point() for x in tensors):
        raise ValueError('real floating-point tensors are required')
    return torch.float64 if any(x.dtype == torch.float64 for x in tensors) else torch.float32


def _validate_tau(tau: float) -> None:
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError('tau must be finite and positive')


def correction(state: Tensor, key: Tensor, value: Tensor, hdiag: Tensor,
               tau: float) -> tuple[Tensor, Tensor]:
    """Diagonal ablation: minimize ||Delta k-e||²/(2*tau)+tr(Delta H Delta.T)/2.

    Leading dimensions broadcast. Cost sums over value dimensions. FP16/BF16 are
    promoted to FP32; FP64 is preserved. This is not a downstream LM loss.
    """
    _validate_tau(tau)
    dtype = _working_dtype(state, key, value, hdiag)
    with torch.autocast(device_type=state.device.type, enabled=False):
        state, key, value, hdiag = [x.to(dtype) for x in (state, key, value, hdiag)]
        if not torch.isfinite(hdiag).all() or torch.any(hdiag <= 0):
            raise ValueError('hdiag must be finite and positive')
        residual = value - torch.einsum('...vk,...k->...v', state, key)
        direction = key / hdiag
        denominator = tau + (key * direction).sum(-1)
        delta = residual.unsqueeze(-1) * direction.unsqueeze(-2) / denominator[..., None, None]
        cost = residual.square().sum(-1) / (2 * denominator)
        return delta, cost


def correction_full(state: Tensor, key: Tensor, value: Tensor, geometry: Tensor,
                    tau: float) -> tuple[Tensor, Tensor]:
    """Exact full-SPD conditional minimizer (up to floating-point arithmetic).

    state [...,V,K], key [...,K], value [...,V], geometry [...,K,K].
    Solves H*u=k without forming H^{-1}; all leading dimensions broadcast.
    The original H is used: no diagonal/low-rank fallback or adaptive jitter.
    Invalid/non-SPD geometry raises instead of silently changing the objective.
    Differentiable in S,k,v,H on the SPD domain (e.g. H=R R.T+ridge I).
    """
    _validate_tau(tau)
    if state.ndim < 2 or geometry.ndim < 2:
        raise ValueError('state and geometry must be matrices')
    K = state.shape[-1]
    if geometry.shape[-2:] != (K, K) or key.shape[-1] != K or value.shape[-1] != state.shape[-2]:
        raise ValueError('state/key/value/geometry dimensions disagree')
    tensors = (state, key, value, geometry)
    if any(x.device != state.device for x in tensors):
        raise ValueError('all solver tensors must use one device')
    dtype = _working_dtype(*tensors)
    with torch.autocast(device_type=state.device.type, enabled=False):
        state, key, value, H = [x.to(dtype) for x in tensors]
        if any(not torch.isfinite(x).all() for x in (state, key, value, H)):
            raise ValueError('solver tensors must be finite')
        tol = 1e-10 if dtype == torch.float64 else 1e-5
        if not torch.allclose(H, H.mT, atol=tol, rtol=tol):
            raise ValueError('geometry must be symmetric')
        # C is explicitly symmetrized when updated. This is not a symmetrizing
        # or jitter fallback for arbitrary external geometry.
        factor = torch.linalg.cholesky(H)
        direction = torch.cholesky_solve(key.unsqueeze(-1), factor).squeeze(-1)
        denominator = tau + (key * direction).sum(-1)
        residual = value - torch.einsum('...vk,...k->...v', state, key)
        delta = residual.unsqueeze(-1) * direction.unsqueeze(-2) / denominator[..., None, None]
        cost = residual.square().sum(-1) / (2 * denominator)
        return delta, cost


@dataclass
class WriteResult:
    states: Tensor
    protection: Tensor
    active: Tensor
    current: Tensor
    parent: Tensor
    destination: Tensor
    fork: Tensor
    cost: Tensor
    replacement_cost: Tensor
    branching_cost: Tensor


def choose_write(states: Tensor, protection: Tensor, active: Tensor,
                 key: Tensor, value: Tensor, *, tau: float, ridge: float,
                 protect: float, fork_cost: float,
                 detach_protection: bool = True) -> WriteResult:
    """Hard minimum across every (parent, replace/fork), functional cache update.

    states [B,H,M,V,K]; protection is FULL [B,H,M,K,K] or diagonal [B,H,M,K].
    All active parents are considered. No ANN/top-k candidate truncation.
    A fork leaves its parent's matrix unchanged and resets the new child's
    usage statistics; a later replacement can retire a bounded-pool version.
    At capacity forking is disabled. This is not an unbounded persistent DAG.
    """
    if any(not math.isfinite(x) for x in (ridge, protect, fork_cost)):
        raise ValueError('cost parameters must be finite')
    if ridge <= 0 or protect < 0 or fork_cost < 0:
        raise ValueError('ridge>0, protect>=0 and fork_cost>=0 are required')
    if states.ndim != 5 or active.shape != states.shape[:3]:
        raise ValueError('states must be [B,H,M,V,K] and active [B,H,M]')
    if not active.any(-1).all():
        raise ValueError('at least one active state per batch/head is required')
    M, V, K = states.shape[-3:]
    full = protection.shape == (*states.shape[:3], K, K)
    diagonal = protection.shape == (*states.shape[:3], K)
    if not full and not diagonal:
        raise ValueError('protection shape must be full [B,H,M,K,K] or diagonal [B,H,M,K]')
    C = protection.detach() if detach_protection else protection
    k, v = key.unsqueeze(2), value.unsqueeze(2)
    if full:
        H = ridge * torch.eye(K, device=states.device, dtype=states.dtype) + protect * C
        delta_r, cr = correction_full(states, k, v, H, tau)
    else:
        delta_r, cr = correction(states, k, v, ridge + protect * C, tau)
    # Fork has no old-version drift penalty because the parent remains readable.
    # Isotropic solve is exact here; no second Cholesky is needed.
    h0 = torch.full_like(states[..., 0, :], ridge)
    delta_f, cf = correction(states, k, v, h0, tau)
    cf = cf + fork_cost
    cr = cr.masked_fill(~active, float('inf'))
    has_free = (~active).any(-1)
    cf = cf.masked_fill(~(active & has_free.unsqueeze(-1)), float('inf'))
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
    if full:
        ix = parent[..., None, None, None].expand(*parent.shape, 1, K, K)
        inherited = protection.gather(2, ix).squeeze(2)
        next_c = torch.where(fork[..., None, None], torch.zeros_like(inherited), inherited)
        new_protection = torch.where(mask[..., None, None], next_c.unsqueeze(2), protection)
    else:
        ix = parent[..., None, None].expand(*parent.shape, 1, K)
        inherited = protection.gather(2, ix).squeeze(2)
        next_c = torch.where(fork[..., None], torch.zeros_like(inherited), inherited)
        new_protection = torch.where(mask[..., None], next_c.unsqueeze(2), protection)
    new_active = active | mask
    cost = costs.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
    return WriteResult(updated, new_protection, new_active, current,
                       parent, destination, fork, cost, cr, cf)
