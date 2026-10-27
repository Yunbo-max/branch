"""Eager reference implementation of the Gated DeltaNet recurrence (paper Eq. 6).

This is deliberately a per-token Python loop: it exposes the full state trajectory
``S_0 .. S_T``, which MARCH needs in order to snapshot anchors. It is a correctness
reference, not a fast path. For real training, use FLA's chunkwise GDN kernels
(``code/flash-linear-attention``) for the text positions and only materialise states at
anchor boundaries.

    S_t = a_t * S_{t-1} + b_t * (v_t - a_t * S_{t-1} k_t) k_t^T
    o_t = S_t q_t

Per head the state is ``S in R^{d_v x d_k}``. Following FLA's GatedDeltaNet convention,
``value_dim = num_heads * head_dim * expand_v`` with ``expand_v = 1``, so ``d_v == d_k``
unless ``expand_v`` is changed.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GatedDeltaRuleReference(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim                          # d_k
        self.value_dim = num_heads * head_dim             # d_v (expand_v = 1)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        # Input-dependent retention / delta strength (Eq. 6).
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=True)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """Run the recurrence over a batch of text tokens.

        Args:
            h: (B, T, D) hidden states for text tokens, in order.
        Returns:
            outputs: (B, T, H, dv) current-state readouts ``S_t q_t``.
            states:  (B, T, H, dv, dk) cumulative state *after* each token.
        """
        b, t, _ = h.shape
        dk, nh = self.head_dim, self.num_heads
        dv = dk  # expand_v = 1: per-head value dim equals head_dim

        k = self.k_proj(h).view(b, t, nh, dk)
        q = self.q_proj(h).view(b, t, nh, dk)
        v = self.v_proj(h).view(b, t, nh, dv)
        a = torch.sigmoid(self.a_proj(h)).view(b, t, nh, 1)      # (0, 1)
        beta = torch.sigmoid(self.b_proj(h)).view(b, t, nh, 1)   # [0, 1]

        state = h.new_zeros(b, nh, dv, dk)  # S_0 = 0
        outs: list[Tensor] = []
        states: list[Tensor] = []
        for i in range(t):
            k_i = k[:, i]           # (B, H, dk)
            v_i = v[:, i]           # (B, H, dv)
            q_i = q[:, i]           # (B, H, dk)
            a_i = a[:, i]           # (B, H, 1)
            b_i = beta[:, i]        # (B, H, 1)

            sk = torch.einsum("bhvk,bhk->bhv", state, k_i)        # S_{t-1} k_t
            state = a_i.unsqueeze(-1) * state + b_i.unsqueeze(-1) * (
                v_i - a_i * sk
            ).unsqueeze(-1) * k_i.unsqueeze(-2)                    # (B,H,dv,dk)
            outs.append(torch.einsum("bhvk,bhk->bhv", state, q_i))
            states.append(state)

        if t == 0:
            return h.new_zeros(b, 0, nh, dv), h.new_zeros(b, 0, nh, dv, dk)
        return torch.stack(outs, dim=1), torch.stack(states, dim=1)
