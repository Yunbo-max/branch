"""Shared LM skeleton for the fair MARCH vs Branch Memory comparison.

Both models use this identical backbone: embedding -> N x (pre-LN mixer, pre-LN MLP)
-> final norm -> LM head. Only the mixer differs. The mixer interface is
``forward(x, cache) -> (output, cache, stats)`` with a cache exposing ``detach()``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

_BRANCH_DIR = Path(__file__).resolve().parents[1] / "branch_memory"
if str(_BRANCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BRANCH_DIR))

from march_mixer import MARCHCache, MARCHMemoryConfig, MARCHMixer  # noqa: E402
from baseline_mixers import BaselineConfig, DeltaNetMixer, TransformerMixer  # noqa: E402

MIXERS = (
    "march",
    "march_chunked",
    "branch",
    "branch_cached",
    "chunked",
    "single_geo",
    "delta",
    "transformer",
)


@dataclass
class FairLMConfig:
    mixer: str = "march"
    vocab_size: int = 8192
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 2
    d_key: int = 32
    d_value: int = 32
    ff_mult: int = 2
    ff_extra: int = 0  # extra FFN hidden units, used to match total parameters
    conv_kernel: int = 3
    # MARCH
    anchor_interval: int = 16
    max_anchors: int = 16
    read_topk: int = 4
    # Branch Memory
    d_route: int = 32
    max_states: int = 8
    tau: float = 1.0
    ridge: float = 1.0
    protect: float = 8.0
    fork_cost: float = 0.02
    protection_decay: float = 0.95
    covariance: str = "full"
    differentiate_covariance: bool = True
    solver_refresh: int = 1  # branch_cached only
    chunk_len: int = 64  # chunked only
    geometry_mode: str = "full"  # single_geo only
    geometry_threshold: float = 0.1  # single_geo certified only
    read_mode: str = "chunk_start"  # single_geo only
    grad_checkpoint: bool = False

    def __post_init__(self) -> None:
        if self.mixer not in MIXERS:
            raise ValueError(f"mixer must be one of {MIXERS}")


def build_mixer(cfg: FairLMConfig) -> nn.Module:
    if cfg.mixer == "march":
        return MARCHMixer(
            MARCHMemoryConfig(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                d_key=cfg.d_key,
                d_value=cfg.d_value,
                d_route=cfg.d_route,
                anchor_interval=cfg.anchor_interval,
                max_anchors=cfg.max_anchors,
                read_topk=cfg.read_topk,
                conv_kernel=cfg.conv_kernel,
                tau=cfg.tau,
                ridge=cfg.ridge,
            )
        )
    if cfg.mixer == "delta":
        return DeltaNetMixer(
            BaselineConfig(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                d_key=cfg.d_key,
                d_value=cfg.d_value,
                conv_kernel=cfg.conv_kernel,
                tau=cfg.tau,
                ridge=cfg.ridge,
            )
        )
    if cfg.mixer == "transformer":
        return TransformerMixer(
            BaselineConfig(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                d_key=cfg.d_key,
                d_value=cfg.d_value,
                conv_kernel=cfg.conv_kernel,
            )
        )
    if cfg.mixer == "march_chunked":
        from chunked_march import ChunkedMARCHConfig, ChunkedMARCHMixer

        return ChunkedMARCHMixer(
            ChunkedMARCHConfig(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                d_key=cfg.d_key,
                d_value=cfg.d_value,
                d_route=cfg.d_route,
                max_states=cfg.max_states,
                conv_kernel=cfg.conv_kernel,
                tau=cfg.tau,
                ridge=cfg.ridge,
                read_topk=cfg.read_topk,
                chunk_len=cfg.chunk_len,
            )
        )
    if cfg.mixer == "chunked":
        from chunked_branch import ChunkedBranchConfig, ChunkedBranchMixer

        return ChunkedBranchMixer(
            ChunkedBranchConfig(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                d_key=cfg.d_key,
                d_value=cfg.d_value,
                d_route=cfg.d_route,
                max_states=cfg.max_states,
                conv_kernel=cfg.conv_kernel,
                tau=cfg.tau,
                ridge=cfg.ridge,
                protect=cfg.protect,
                fork_cost=cfg.fork_cost,
                protection_decay=cfg.protection_decay,
                covariance=cfg.covariance,
                differentiate_covariance=cfg.differentiate_covariance,
                chunk_len=cfg.chunk_len,
                grad_checkpoint=cfg.grad_checkpoint,
            )
        )
    if cfg.mixer == "single_geo":
        from single_geometry import SingleGeometryConfig, SingleGeometryMixer

        return SingleGeometryMixer(SingleGeometryConfig(
            d_model=cfg.d_model, n_heads=cfg.n_heads, d_key=cfg.d_key,
            d_value=cfg.d_value, d_route=cfg.d_route, max_states=1,
            conv_kernel=cfg.conv_kernel, tau=cfg.tau, ridge=cfg.ridge,
            protect=cfg.protect, protection_decay=cfg.protection_decay,
            covariance="full", differentiate_covariance=cfg.differentiate_covariance,
            chunk_len=cfg.chunk_len, grad_checkpoint=cfg.grad_checkpoint,
            geometry_mode=cfg.geometry_mode,
            geometry_threshold=cfg.geometry_threshold,
            read_mode=cfg.read_mode,
        ))
    if cfg.mixer == "branch_cached":
        from branch_cached import CachedBranchMixer, CachedSolverConfig

        return CachedBranchMixer(
            CachedSolverConfig(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                d_key=cfg.d_key,
                d_value=cfg.d_value,
                d_route=cfg.d_route,
                max_states=cfg.max_states,
                mode="branch",
                conv_kernel=cfg.conv_kernel,
                tau=cfg.tau,
                ridge=cfg.ridge,
                protect=cfg.protect,
                fork_cost=cfg.fork_cost,
                protection_decay=cfg.protection_decay,
                covariance="full",
                differentiate_covariance=cfg.differentiate_covariance,
                solver_refresh=cfg.solver_refresh,
            )
        )
    from branch_memory.mixer import BranchMemoryMixer, MemoryConfig

    return BranchMemoryMixer(
        MemoryConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_key=cfg.d_key,
            d_value=cfg.d_value,
            d_route=cfg.d_route,
            max_states=cfg.max_states,
            mode="branch",
            conv_kernel=cfg.conv_kernel,
            tau=cfg.tau,
            ridge=cfg.ridge,
            protect=cfg.protect,
            fork_cost=cfg.fork_cost,
            protection_decay=cfg.protection_decay,
            covariance=cfg.covariance,
            differentiate_covariance=cfg.differentiate_covariance,
        )
    )


class FairBlock(nn.Module):
    def __init__(self, cfg: FairLMConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.mixer = build_mixer(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        hidden = cfg.ff_mult * cfg.d_model + cfg.ff_extra
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.d_model),
        )

    def forward(self, x: Tensor, cache=None):
        z, cache, stats = self.mixer(self.norm1(x), cache)
        x = x + z
        return x + self.ff(self.norm2(x)), cache, stats


class FairLM(nn.Module):
    def __init__(self, cfg: FairLMConfig) -> None:
        super().__init__()
        self.config = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([FairBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, input_ids: Tensor, caches=None, return_cache: bool = False):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch,time]")
        if caches is not None and len(caches) != len(self.blocks):
            raise ValueError("wrong number of caches")
        x = self.embedding(input_ids)
        out_caches, stats = [], []
        for i, block in enumerate(self.blocks):
            x, cache, s = block(x, None if caches is None else caches[i])
            out_caches.append(cache)
            stats.append(s)
        logits = self.head(self.norm(x))
        return (logits, stats, out_caches) if return_cache else (logits, stats)


def count_mixer_parameters(model: FairLM) -> dict[str, int]:
    mixer = sum(p.numel() for b in model.blocks for p in b.mixer.parameters())
    other = sum(p.numel() for p in model.parameters()) - mixer
    return {"mixer": mixer, "backbone_and_head": other, "total": mixer + other}
