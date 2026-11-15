"""Train-only lightweight readout adapter and supervised true-state-jump head."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .features import FixedStateSketch


class NativeReadoutAdapter(nn.Module):
    """Same adapter topology for C01/C02/C03/C04; zero-init equals native MTP.

    It changes the MTP *prediction readout*, not the 32 frozen backbone layers.
    The adapter receives a real frozen GDN or GQA mixer feature sampled at
    the verified context boundary; it never sees target future readouts.
    """
    def __init__(self, hidden: int = 2560, rank: int = 128):
        super().__init__()
        if hidden <= 0 or rank <= 0:
            raise ValueError("positive adapter dimensions required")
        self.norm = nn.LayerNorm(hidden)
        self.down = nn.Linear(hidden, rank, bias=False)
        self.up = nn.Linear(rank, hidden, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, native_mtp_hidden: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        if native_mtp_hidden.shape != source.shape or native_mtp_hidden.ndim != 2:
            raise ValueError("adapter expects aligned [B,hidden] tensors")
        correction = self.up(F.silu(self.down(self.norm(source.float()))))
        return native_mtp_hidden.float() + correction


class StateJumpHead(nn.Module):
    """Auxiliary: predict S_end-S_start in a fixed bilinear state sketch.

    The MTP hidden after the proposed tokens is already available to the draft
    procedure. The true future state is computed by the frozen backbone and
    used ONLY as a training target, never fed into the student.
    """
    def __init__(self, hidden: int = 2560, sketch_rank: int = 8, width: int = 256):
        super().__init__()
        self.hidden = hidden
        self.sketch = FixedStateSketch(rank=sketch_rank)
        d = self.sketch.output_dim
        self.net = nn.Sequential(
            nn.LayerNorm(hidden + d),
            nn.Linear(hidden + d, width),
            nn.GELU(),
            nn.Linear(width, d),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def predict_delta(self, final_draft_hidden: torch.Tensor,
                      start_state: torch.Tensor) -> torch.Tensor:
        if final_draft_hidden.ndim != 2 or final_draft_hidden.shape[-1] != self.hidden:
            raise ValueError("final draft hidden mismatch")
        start = self.sketch(start_state)
        return self.net(torch.cat((final_draft_hidden.float(), start), dim=-1))

    def loss(self, final_draft_hidden: torch.Tensor,
             start_state: torch.Tensor, end_state: torch.Tensor):
        actual_delta = self.sketch(end_state) - self.sketch(start_state)
        pred = self.predict_delta(final_draft_hidden, start_state)
        # State deltas can be tiny: scale relative to the true delta magnitude.
        denominator = actual_delta.pow(2).mean().detach().clamp_min(1e-4)
        loss = F.mse_loss(pred, actual_delta) / denominator
        return loss, {"sketch_delta_rms": float(actual_delta.pow(2).mean().sqrt().item()),
                      "predicted_delta_rms": float(pred.detach().pow(2).mean().sqrt().item())}
