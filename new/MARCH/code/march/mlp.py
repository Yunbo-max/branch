"""SwiGLU feed-forward, matching FLA's GatedMLP dimension convention.

FLA picks ``intermediate = int(hidden * ratio * 2 / 3)`` rounded up to a multiple of 256,
so ``hidden_ratio=4`` on hidden 1536 gives 4096. Mirrored here so a MARCH layer and a
Gated DeltaNet layer of the same config differ only in the memory mechanism.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GatedMLP(nn.Module):
    def __init__(self, hidden_size: int, hidden_ratio: int = 4,
                 intermediate_size: int | None = None) -> None:
        super().__init__()
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 255) // 256)
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
