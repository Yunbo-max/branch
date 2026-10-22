"""Causal, usage-weighted query second moments (not centred covariance)."""
import math
import torch
from torch import Tensor


def update_protection(protection: Tensor, query: Tensor, use: Tensor, active: Tensor,
                      *, decay: float, covariance: str = 'full',
                      detach: bool = True) -> Tensor:
    """Call AFTER current write/read, so the next write sees only past queries.

    C_new = decay*C + (1-decay)*use*q*q^T, full geometry.
    The diagonal ablation keeps only q^2. `use` is the original prototype's
    historical route probability plus the current-state indicator, NOT a
    squared mixture coefficient. It defines a per-state surrogate, not exact
    change of the entire routed-network loss.

    Normalized q and nonnegative use, with PSD initial C, preserve PSD in exact
    arithmetic. No eigenvalue clipping/rank truncation is performed.
    """
    if not math.isfinite(decay) or not 0 <= decay < 1:
        raise ValueError('decay must be finite and in [0,1)')
    if covariance not in {'full', 'diagonal'}:
        raise ValueError('covariance must be full or diagonal')
    if use.shape != active.shape or query.shape[:2] != use.shape[:2]:
        raise ValueError('query/use/active dimensions disagree')
    if torch.any(use < 0) or not torch.isfinite(use).all():
        raise ValueError('use must be finite and nonnegative')
    if detach:
        protection, query, use = protection.detach(), query.detach(), use.detach()
    dtype = torch.float64 if protection.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=query.device.type, enabled=False):
        C, q, weight = protection.to(dtype), query.to(dtype), use.to(dtype)
        if covariance == 'full':
            outer = q.unsqueeze(-1) * q.unsqueeze(-2)
            updated = decay*C + (1-decay)*weight[..., None, None]*outer.unsqueeze(2)
            updated = (updated + updated.mT)*.5
            return updated * active[..., None, None]
        return (decay*C + (1-decay)*weight[..., None]*q.square().unsqueeze(2))*active[..., None]
