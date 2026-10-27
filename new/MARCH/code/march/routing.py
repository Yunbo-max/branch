"""Content-routed historical reading (paper Eq. 12-14).

Visible anchors for a text token at original position ``t`` are those whose boundary
``b_m`` satisfies ``b_m <= t`` (the anchor was emitted after finishing token ``b_m - 1``).
A learned null option with a zero payload lets the model bypass history entirely.
"""

from __future__ import annotations

import torch
from torch import Tensor


def routing_weights(
    route_query: Tensor,      # (B, T, d_r)  rho_t = W_R x_t
    anchor_keys: Tensor,      # (B, M, d_r)  kappa_m
    null_logit: Tensor,       # (B, T)       n_t = w_null^T x_t + b_null
    visible: Tensor,          # (B, T, M)    bool, b_m <= t
    top_k: int = -1,
) -> Tensor:
    """Return pi over ``[anchor_0 .. anchor_{M-1}, null]`` -> (B, T, M + 1)."""
    scores = torch.einsum("btd,bmd->btm", route_query, anchor_keys)  # a_{t,m}
    scores = scores.masked_fill(~visible, float("-inf"))

    if top_k > 0 and top_k < scores.shape[-1]:
        # Keep the K best visible anchors; everything else is routed to null.
        order = scores.topk(top_k, dim=-1).indices
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep.scatter_(-1, order, True)
        scores = scores.masked_fill(~keep, float("-inf"))

    # A row is [-inf ... -inf, null]: softmax yields null = 1, which is exactly the
    # paper's "if no anchor is visible, the null option receives all probability mass".
    logits = torch.cat([scores, null_logit.unsqueeze(-1)], dim=-1)
    logits = torch.nan_to_num(logits, nan=float("-inf"))
    pi = torch.softmax(logits, dim=-1)
    return pi


def historical_readout(pi: Tensor, anchors: Tensor, read_query: Tensor) -> Tensor:
    """Weighted aggregation of anchor states read by the *state-read* query.

    Args:
        pi:          (B, T, M + 1) routing probabilities (last slot is null).
        anchors:     (B, M, H, dv, dk) state anchors for this layer.
        read_query:  (B, T, H, dk) q_t.
    Returns:
        (B, T, H, dv) sum_j pi_{t,j} A_j q_t, with the null anchor contributing zero.
    """
    # (B, H, M, dv, dk) @ (B, H, T, dk) -> (B, H, M, T, dv)
    per_anchor = torch.einsum("bmhvk,bthk->bhmtv", anchors, read_query)
    weights = pi[:, :, :-1].permute(0, 2, 1)  # (B, M, T)
    return torch.einsum("bmt,bhmtv->bthv", weights, per_anchor)
