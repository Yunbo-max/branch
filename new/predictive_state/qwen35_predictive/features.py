"""Causally aligned native Qwen3.5 GDN/GQA readouts and TRUE GDN state sketches.

Readouts come from native layer mixer outputs (not synthetic attention).
The state sketch comes from Qwen's actual FP32 cache matrix S [B,H,K,V].
No model weights or checkpoint files are altered.
"""
from __future__ import annotations

from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F

GDN_LAYER = 26  # third GDN in the seventh 3-GDN/1-GQA group
GQA_LAYER = 27  # following full-attention layer


def assert_native_layout(base) -> None:
    cfg = base.config
    layer_types = list(cfg.layer_types)
    if len(layer_types) != 32 or layer_types[GDN_LAYER] != "linear_attention" \
            or layer_types[GQA_LAYER] != "full_attention":
        raise RuntimeError("Pinned Qwen3.5-4B 24 GDN + 8 GQA layout is required")
    if int(cfg.hidden_size) != 2560 or int(cfg.linear_num_value_heads) != 32 \
            or int(cfg.linear_key_head_dim) != 128 or int(cfg.linear_value_head_dim) != 128:
        raise RuntimeError("unexpected Qwen GDN dimensions")
    layers = base.model.layers
    if not hasattr(layers[GDN_LAYER], "linear_attn") or not hasattr(layers[GQA_LAYER], "self_attn"):
        raise RuntimeError("native mixer modules not found; do not fall back to hidden-state differences")


@contextmanager
def collect_mixer_readouts(base):
    """Register only for one frozen full-prefix backbone forward, then remove."""
    assert_native_layout(base)
    record = {}
    def hook(label):
        def save(_module, _args, output):
            if label in record:
                raise RuntimeError("mixer invoked more than once for a single capture")
            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[-1] != 2560:
                raise RuntimeError("native mixer output shape changed")
            record[label] = value[:, -1, :].detach().clone()
        return save
    handles = [
        base.model.layers[GDN_LAYER].linear_attn.register_forward_hook(hook("gdn")),
        base.model.layers[GQA_LAYER].self_attn.register_forward_hook(hook("gqa")),
    ]
    try:
        yield record
    finally:
        for handle in handles:
            handle.remove()


def native_gdn_matrix(cache, layer: int = GDN_LAYER) -> torch.Tensor:
    """Obtain the actual final GDN recurrent matrix, not the hidden readout.

    Qwen's official recurrent rule stores [batch,value-heads,key-dim,value-dim],
    i.e. [B,32,128,128] for Qwen3.5-4B.
    """
    try:
        value = cache.layers[layer].recurrent_states[0]
    except (IndexError, AttributeError, TypeError) as exc:
        raise RuntimeError("Qwen native GDN cache missing; use_cache=True required") from exc
    if value is None or value.ndim != 4 or tuple(value.shape[1:]) != (32, 128, 128):
        raise RuntimeError(f"Unexpected native GDN state shape: {getattr(value, 'shape', None)}")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("nonfinite native GDN recurrent state")
    return value.detach().clone()


class FixedStateSketch(nn.Module):
    """Deterministic random bilinear measurements of 4 selected GDN state heads.

    P^T S Q is a sketch, not the full original matrix; no capacity claim is made.
    Registered buffers are saved with C04 checkpoints to make the target invariant.
    """
    def __init__(self, *, rank: int = 8, seed: int = 149):
        super().__init__()
        if rank < 1 or rank > 128:
            raise ValueError("sketch rank must be between 1 and 128")
        rng = torch.Generator(device="cpu").manual_seed(seed)
        self.register_buffer("left", torch.randn(128, rank, generator=rng) / (128**0.5))
        self.register_buffer("right", torch.randn(128, rank, generator=rng) / (128**0.5))
        self.register_buffer("heads", torch.tensor([0, 8, 16, 24], dtype=torch.long))
        self.output_dim = 4 * rank * rank

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 4 or tuple(state.shape[1:]) != (32, 128, 128):
            raise ValueError("state sketch expects true Qwen GDN matrix [B,32,128,128]")
        s = state.float().index_select(1, self.heads.to(state.device))
        left, right = self.left.float(), self.right.float()
        # [B,4,128,128] -> [B,4,r,r]
        latent = torch.einsum("bhkv,kr,vs->bhrs", s, left, right)
        return latent.reshape(state.shape[0], -1).detach()


def select_source(case: str, gdn: torch.Tensor, gqa: torch.Tensor) -> torch.Tensor:
    if case == "c01_gdn":
        return F.layer_norm(gdn.float(), (gdn.shape[-1],))
    if case == "c02_gqa":
        return F.layer_norm(gqa.float(), (gqa.shape[-1],))
    if case in {"c03_fusion", "c04_state"}:
        x = F.layer_norm(gdn.float(), (gdn.shape[-1],))
        y = F.layer_norm(gqa.float(), (gqa.shape[-1],))
        return (x + y) * 0.5
    if case == "c00_native":
        return torch.zeros_like(gdn.float())
    raise ValueError(f"unknown case: {case}")
