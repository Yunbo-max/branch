"""Small, differentiable PyTorch reference of the ORIGINAL gated Delta update.

Per head S has shape [value_dim, key_dim]:
  S' = alpha*S + beta*(v - (alpha*S)@k) outer k
  read = S'@q
This is a correctness and training bootstrap, NOT the Flash Linear Attention
fused/chunk kernel, and not an inference-speed claim.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GatedDeltaCell(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int):
        super().__init__()
        if min(dim, heads, head_dim) <= 0 or heads * head_dim != dim:
            raise ValueError("v0 requires hidden_dim = heads * head_dim")
        self.heads, self.head_dim = heads, head_dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.a_proj = nn.Linear(dim, heads)
        self.b_proj = nn.Linear(dim, heads)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        nn.init.constant_(self.a_proj.bias, 4.0)  # initial retention ~0.982
        nn.init.constant_(self.b_proj.bias, -1.0)  # initial write strength ~0.269

    def initial_state(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.shape[0], self.heads, self.head_dim, self.head_dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        if x.ndim != 2:
            raise ValueError("one token [batch,hidden] is required")
        batch = x.shape[0]
        h, d = self.heads, self.head_dim
        if state is None:
            state = self.initial_state(x)
        if state.shape != (batch, h, d, d):
            raise ValueError("invalid recurrent state shape")
        if state.device != x.device or state.dtype != x.dtype:
            raise ValueError("state device/dtype must match hidden token")
        q = F.normalize(self.q_proj(x).view(batch, h, d), dim=-1, eps=1e-6)
        k = F.normalize(self.k_proj(x).view(batch, h, d), dim=-1, eps=1e-6)
        v = self.v_proj(x).view(batch, h, d)
        alpha = torch.sigmoid(self.a_proj(x)).view(batch, h, 1, 1)
        beta = torch.sigmoid(self.b_proj(x)).view(batch, h, 1, 1)
        retained = alpha * state
        residual = v - torch.einsum("bhvk,bhk->bhv", retained, k)
        state = retained + beta * residual.unsqueeze(-1) * k.unsqueeze(-2)
        out = torch.einsum("bhvk,bhk->bhv", state, q)
        return self.o_proj(out.reshape(batch, h * d)), state


class GatedDeltaBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cell = GatedDeltaCell(dim, heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.SiLU(),
                                 nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        delta, state = self.cell(self.norm1(x), state)
        x = x + delta
        return x + self.ffn(self.norm2(x)), state
