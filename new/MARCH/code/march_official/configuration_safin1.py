"""Configuration wrappers for the Safin-1 dense model."""

from typing import ClassVar

from huggingface_hub.dataclasses import strict
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)


def _reject_legacy_anchor_fields(config) -> None:
    legacy = sorted(
        name for name in vars(config)
        if name.startswith("summary_") or name.startswith("hsd_summary_")
    )
    if legacy:
        raise ValueError(
            f"Unsupported pre-anchor Safin-1 config fields: {legacy}. "
            "Run scripts/march/migrate_checkpoint.py on a checkpoint copy."
        )


@strict
class Safin1TextConfig(Qwen3_5TextConfig):
    """Dense Qwen3.5 text config plus historical linear-attention controls."""

    model_type = "safin1_4b_text"

    hist_chunk_size: int = 512
    hist_read_block_size: int = 32768
    hist_read_chunk_block_size: int = 16
    hist_use_history_read: bool = True
    hist_disable_cache: bool = False
    hist_replace_linear_attn: bool = True

    anchor_chunk_size: int = 512
    anchor_token_num: int = 1
    anchor_placeholder_token_id: int = 0
    anchor_bos_token_id: int | None = 248045
    anchor_position_ids_type: str = "zeros"
    anchor_embedding_type: str = "learned"
    anchor_embedding_init_from: str = "bos"
    anchor_embedding_init_scale: float | None = 0.5

    hsd_read_block_size: int = 32768
    hsd_read_chunk_block_size: int = 16
    hsd_use_global_read: bool = True
    hsd_global_read_layers: str | list[int] | tuple[int, ...] | None = "10-30"
    hsd_history_read_q_mode: str = "shared_local"
    hsd_history_read_q_lora_rank: int = 32
    hsd_full_attention_anchor_tokens: bool = False
    hsd_use_route_scale: bool = True
    hsd_route_logit_scale: float = 1.0
    hsd_route_head_dim: int | None = None
    hsd_route_query_heads: int | None = None
    hsd_route_key_heads: int | None = None
    hsd_route_qk_l2norm: bool = False
    hsd_route_position_encoding: str = "none"
    hsd_route_position_mode: str = "chunk_end"
    hsd_read_topk: int = 4
    hsd_null_read_gate: bool = True
    hsd_null_read_gate_init_bias: float = 0.0
    # One virtual anchor at document boundary zero whose recurrent state is
    # replaced by a per-linear-layer FP32 parameter.  This is opt-in so older
    # checkpoints keep their exact layout and state dict by default.
    hsd_persistent_memory_enabled: bool = False
    # Optional per-layer key used only by the boundary-zero persistent slot.
    # Dynamic anchors always keep using anchor_k_proj(anchor_hidden).
    hsd_persistent_memory_route_key_enabled: bool = False
    hsd_persistent_memory_route_key_mode: str = "legacy"
    hsd_history_scale: float = 1.0
    hsd_history_out_rmsnorm: bool = False
    hsd_history_out_rmsnorm_eps: float = 1e-6
    hsd_learnable_history_scale: bool | None = None
    hsd_learnable_history_scale_init: float | None = None
    hsd_learnable_history_scale_max: float | None = None
    hsd_history_gate_schedule_start: float | None = None
    hsd_history_gate_schedule_end: float | None = None
    hsd_history_gate_schedule_steps: int = 0

    hsd_history_out_grad_scale: float | None = None
    hsd_history_out_grad_scale_start: float | None = None
    hsd_history_out_grad_scale_end: float | None = None
    hsd_history_out_grad_scale_steps: int = 0
    hsd_anchor_content_grad_scale: float | None = None
    hsd_anchor_content_grad_scale_start: float | None = None
    hsd_anchor_content_grad_scale_end: float | None = None
    hsd_anchor_content_grad_scale_steps: int = 0
    hsd_anchor_embedding_grad_scale: float | None = None
    hsd_anchor_embedding_grad_scale_start: float | None = None
    hsd_anchor_embedding_grad_scale_end: float | None = None
    hsd_anchor_embedding_grad_scale_steps: int = 0
    hsd_anchor_embedding_lr_mult: float | None = None

    hsd_local_state_mode: str = "full"
    hsd_text_only_mlp: bool = False
    hsd_text_only_in_proj: bool = False
    hsd_anchor_isolated_in_proj: bool = False
    hsd_text_single_pass_delta: bool = False
    hsd_anchor_mlp: bool = False
    hsd_anchor_mlp_hidden_size: int | None = None

    HSD_TRAINING_ONLY_FIELDS: ClassVar[tuple[str, ...]] = (
        "hsd_history_out_grad_scale",
        "hsd_history_out_grad_scale_start",
        "hsd_history_out_grad_scale_end",
        "hsd_history_out_grad_scale_steps",
        "hsd_anchor_content_grad_scale",
        "hsd_anchor_content_grad_scale_start",
        "hsd_anchor_content_grad_scale_end",
        "hsd_anchor_content_grad_scale_steps",
        "hsd_anchor_embedding_grad_scale",
        "hsd_anchor_embedding_grad_scale_start",
        "hsd_anchor_embedding_grad_scale_end",
        "hsd_anchor_embedding_grad_scale_steps",
        "hsd_anchor_embedding_lr_mult",
    )

    def validate_anchor_schema(self) -> None:
        _reject_legacy_anchor_fields(self)


@strict
class Safin1Config(Qwen3_5Config):
    """Top-level dense historical config used by `trust_remote_code` checkpoints."""

    model_type = "safin1_4b"
    sub_configs = {
        "vision_config": Qwen3_5VisionConfig,
        "text_config": Safin1TextConfig,
    }

    def validate_anchor_schema(self) -> None:
        _reject_legacy_anchor_fields(self)


__all__ = ["Safin1Config", "Safin1TextConfig"]
