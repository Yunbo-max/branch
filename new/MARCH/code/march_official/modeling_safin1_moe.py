"""Model wrappers that replace Qwen3.5-MoE linear attention with historical GDN."""

from __future__ import annotations

import inspect
from types import MethodType

import torch
from torch import nn
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen35_moe_modeling
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeModel,
    Qwen3_5MoeModelOutputWithPast,
    Qwen3_5MoePreTrainedModel,
    Qwen3_5MoeTextModel,
    Qwen3_5MoeVisionModel,
)

# Transformers local remote-code caching copies direct relative imports.
# Declare transitive dependencies here so a fresh cache is self-contained.
from .fused_history_read import fused_prefill_top4_history_read as _remote_history_read  # noqa: F401
from .fused_march_segments import FLA_CHUNK_SIZE as _remote_fla_chunk_size  # noqa: F401
from .configuration_safin1_moe import Safin1MoeConfig, Safin1MoeTextConfig
from .march_gdn import (
    Safin1MoeGatedDeltaNet,
    collect_persistent_memory_metrics,
    load_persistent_memory_artifact,
    save_persistent_memory_artifact,
    trace_hsd_internal,
)
from .ops import route_softmax_weights as _route_softmax_weights  # noqa: F401


def _anchor_enabled(config: Safin1MoeTextConfig) -> bool:
    return (
        bool(getattr(config, "hist_replace_linear_attn", True))
        and int(getattr(config, "anchor_token_num", 0) or 0) > 0
        and (
            int(getattr(config, "anchor_chunk_size", 0) or 0) > 0
            or bool(getattr(config, "hsd_persistent_memory_enabled", False))
        )
    )


def _persistent_memory_enabled(config) -> bool:
    return bool(getattr(config, "hsd_persistent_memory_enabled", False))


def _anchor_mlp_enabled(config: Safin1MoeTextConfig) -> bool:
    return bool(getattr(config, "hsd_anchor_mlp", False))


def _text_only_mlp_enabled(config: Safin1MoeTextConfig) -> bool:
    return _anchor_mlp_enabled(config) or bool(getattr(config, "hsd_text_only_mlp", False))


def _config_return_dict(config) -> bool:
    value = getattr(config, "return_dict", None)
    if value is not None:
        return bool(value)
    return bool(getattr(config, "use_return_dict", True))


def _new_qwen35_cache(config):
    cache_cls = getattr(qwen35_moe_modeling, "Qwen3_5MoeDynamicCache", DynamicCache)
    try:
        cache = cache_cls(config=config)
    except TypeError:
        cache = cache_cls()
    return _patch_hsd_cache_get_seq_length(cache)


def _patch_hsd_cache_get_seq_length(cache):
    if cache is None or getattr(cache, "_historical_hsd_get_seq_length_patched", False):
        return cache
    original_get_seq_length = getattr(cache, "get_seq_length", None)
    if not callable(original_get_seq_length):
        return cache

    def get_seq_length(self, *args, **kwargs):
        try:
            return original_get_seq_length(*args, **kwargs)
        except ValueError as exc:
            message = str(exc)
            if "get_seq_length" not in message or "LinearAttention" not in message:
                raise
            value = getattr(self, "hsd_seen_expanded_tokens", None)
            if value is not None:
                return _length_value_to_int(value)
            value = getattr(self, "hsd_seen_text_tokens", None)
            if value is not None:
                return _length_value_to_int(value)
            return 0

    cache.get_seq_length = MethodType(get_seq_length, cache)
    cache._historical_hsd_get_seq_length_patched = True
    return cache


def _cache_has_previous_state(past_key_values) -> bool:
    if past_key_values is None:
        return False
    if _length_value_to_int(getattr(past_key_values, "hsd_seen_expanded_tokens", 0)) > 0:
        return True
    if _length_value_to_int(getattr(past_key_values, "hsd_seen_text_tokens", 0)) > 0:
        return True
    if _cache_text_length(past_key_values) > 0:
        return True
    has_previous_state = getattr(past_key_values, "has_previous_state", None)
    if callable(has_previous_state):
        try:
            return bool(has_previous_state())
        except TypeError:
            return bool(has_previous_state(0))
    if isinstance(has_previous_state, bool):
        return has_previous_state
    try:
        return int(past_key_values.get_seq_length()) > 0
    except Exception:
        return False


def _cache_text_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    value = getattr(past_key_values, "hsd_seen_text_tokens", None)
    if value is not None:
        return _length_value_to_int(value)
    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        for layer in layers:
            value = getattr(layer, "hsd_seen_text_tokens", None)
            if value is not None:
                return _length_value_to_int(value)
    layers = getattr(past_key_values, "_hsd_layer_states", None)
    if layers is not None:
        for layer in layers:
            value = getattr(layer, "hsd_seen_text_tokens", None)
            if value is not None:
                return _length_value_to_int(value)
    try:
        return int(past_key_values.get_seq_length())
    except Exception:
        return 0


def _cache_expanded_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    value = getattr(past_key_values, "hsd_seen_expanded_tokens", None)
    if value is not None:
        return _length_value_to_int(value)
    try:
        value = int(past_key_values.get_seq_length())
        if value > 0:
            return value
    except Exception:
        pass
    text_len = _cache_text_length(past_key_values)
    return text_len


def _length_value_to_int(value) -> int:
    if value is None:
        return 0
    if torch.is_tensor(value):
        if value.numel() == 0:
            return 0
        return int(value.detach().max().item())
    return int(value or 0)


def _cache_lengths(
    past_key_values,
    *,
    base_attr: str,
    per_row_attr: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if past_key_values is None:
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    for source in (past_key_values,):
        value = getattr(source, per_row_attr, None)
        if value is not None:
            tensor = torch.as_tensor(value, dtype=torch.long, device=device).reshape(-1)
            if tensor.numel() == batch_size:
                return tensor
            if tensor.numel() == 1:
                return tensor.expand(batch_size)
    for attr_name in ("layers", "_hsd_layer_states"):
        layers = getattr(past_key_values, attr_name, None)
        if layers is None:
            continue
        for layer in layers:
            value = getattr(layer, per_row_attr, None)
            if value is not None:
                tensor = torch.as_tensor(value, dtype=torch.long, device=device).reshape(-1)
                if tensor.numel() == batch_size:
                    return tensor
                if tensor.numel() == 1:
                    return tensor.expand(batch_size)
            value = getattr(layer, base_attr, None)
            if value is not None:
                scalar = _length_value_to_int(value)
                return torch.full((batch_size,), scalar, dtype=torch.long, device=device)
    value = getattr(past_key_values, base_attr, None)
    if value is not None:
        scalar = _length_value_to_int(value)
        return torch.full((batch_size,), scalar, dtype=torch.long, device=device)
    scalar = _cache_text_length(past_key_values) if base_attr == "hsd_seen_text_tokens" else _cache_expanded_length(past_key_values)
    return torch.full((batch_size,), scalar, dtype=torch.long, device=device)


def _cache_text_lengths(past_key_values, batch_size: int, device: torch.device) -> torch.Tensor:
    return _cache_lengths(
        past_key_values,
        base_attr="hsd_seen_text_tokens",
        per_row_attr="hsd_seen_text_tokens_per_row",
        batch_size=batch_size,
        device=device,
    )


def _cache_expanded_lengths(past_key_values, batch_size: int, device: torch.device) -> torch.Tensor:
    return _cache_lengths(
        past_key_values,
        base_attr="hsd_seen_expanded_tokens",
        per_row_attr="hsd_seen_expanded_tokens_per_row",
        batch_size=batch_size,
        device=device,
    )


def _set_root_cache_lengths(
    past_key_values,
    *,
    text_lengths: torch.Tensor,
    expanded_lengths: torch.Tensor,
) -> None:
    if past_key_values is None:
        return
    text_cpu = text_lengths.detach().to(dtype=torch.long)
    expanded_cpu = expanded_lengths.detach().to(dtype=torch.long)
    past_key_values.hsd_seen_text_tokens_per_row = text_cpu
    past_key_values.hsd_seen_expanded_tokens_per_row = expanded_cpu
    past_key_values.hsd_seen_text_tokens = _length_value_to_int(text_cpu)
    past_key_values.hsd_seen_expanded_tokens = _length_value_to_int(expanded_cpu)


def _build_anchor_layout(
    batch_size: int,
    text_len: int,
    config: Safin1MoeTextConfig,
    device: torch.device,
) -> torch.Tensor:
    anchor_mask, _ = _build_anchor_layout_masks(batch_size, text_len, config, device)
    return anchor_mask


def _build_anchor_layout_masks(
    batch_size: int,
    text_len: int,
    config: Safin1MoeTextConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunk_size = int(getattr(config, "anchor_chunk_size", 0) or 0)
    anchor_token_num = int(config.anchor_token_num)
    chunk_count = text_len // chunk_size if chunk_size > 0 else 0
    persistent_count = 1 if _persistent_memory_enabled(config) else 0
    expanded_len = text_len + persistent_count + chunk_count * anchor_token_num
    anchor_mask = torch.zeros(batch_size, expanded_len, dtype=torch.bool, device=device)
    persistent_memory_mask = torch.zeros_like(anchor_mask)

    write = 0
    if persistent_count:
        anchor_mask[:, 0] = True
        persistent_memory_mask[:, 0] = True
        write = 1
    for _ in range(chunk_count):
        write += chunk_size
        anchor_mask[:, write : write + anchor_token_num] = True
        write += anchor_token_num
    return anchor_mask, persistent_memory_mask


def _is_left_padding_mask(attention_mask: torch.Tensor) -> bool:
    if attention_mask.ndim != 2:
        return False
    mask = attention_mask.to(dtype=torch.bool)
    if mask.shape[1] <= 1:
        return True
    return not bool((mask[:, :-1] & ~mask[:, 1:]).any().item())


def _expand_2d_with_anchor(
    tensor: torch.Tensor,
    anchor_mask: torch.Tensor,
    fill_value: int | float = 0,
    anchor_fill: bool = False,
) -> torch.Tensor:
    batch_size, text_len = tensor.shape
    expanded = torch.full(
        (batch_size, anchor_mask.shape[1]),
        fill_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    text_positions = (~anchor_mask[0]).nonzero(as_tuple=False).flatten().to(tensor.device)
    expanded.index_copy_(1, text_positions, tensor)
    if anchor_fill and bool(anchor_mask.any().item()):
        anchors = _anchor_text_indices(anchor_mask).to(tensor.device)
        expanded[anchor_mask] = tensor.gather(1, anchors).reshape(-1)
    return expanded


def _expand_embeds_with_anchor(inputs_embeds: torch.Tensor, anchor_mask: torch.Tensor) -> torch.Tensor:
    batch_size, _, hidden_size = inputs_embeds.shape
    expanded = inputs_embeds.new_zeros(batch_size, anchor_mask.shape[1], hidden_size)
    text_positions = (~anchor_mask[0]).nonzero(as_tuple=False).flatten().to(inputs_embeds.device)
    expanded.index_copy_(1, text_positions, inputs_embeds)
    return expanded


def _anchor_text_indices(anchor_mask: torch.Tensor) -> torch.Tensor:
    text_seen = (~anchor_mask).long().cumsum(dim=1).sub(1)
    anchors = text_seen[anchor_mask].view(anchor_mask.shape[0], -1)
    # A persistent-memory anchor intentionally precedes the first text token.
    # For chunk-end embedding/position compatibility, anchor that one virtual
    # token to text position zero.  Dynamic anchors retain their old anchors.
    return anchors.clamp_min(0)


def _expand_position_ids(
    position_ids: torch.Tensor | None,
    text_len: int,
    anchor_mask: torch.Tensor,
    config: Safin1MoeTextConfig,
    device: torch.device,
) -> torch.Tensor | None:
    if anchor_mask.shape[1] == text_len and not bool(anchor_mask.any().item()):
        return position_ids

    anchor_position_type = str(getattr(config, "anchor_position_ids_type", "zeros"))
    anchor_fill = anchor_position_type == "chunk_end"
    if anchor_position_type not in {"zeros", "chunk_end"}:
        raise ValueError("anchor_position_ids_type must be either 'zeros' or 'chunk_end'.")

    batch_size = anchor_mask.shape[0]
    if position_ids is None:
        base = torch.arange(text_len, dtype=torch.long, device=device).view(1, -1).expand(batch_size, -1)
        expanded = _expand_2d_with_anchor(base, anchor_mask, fill_value=0, anchor_fill=anchor_fill)
        return expanded.view(1, batch_size, -1).expand(4, batch_size, -1)

    if position_ids.ndim == 2:
        return _expand_2d_with_anchor(position_ids, anchor_mask, fill_value=0, anchor_fill=anchor_fill)
    if position_ids.ndim == 3:
        rows = []
        for row in position_ids:
            rows.append(_expand_2d_with_anchor(row, anchor_mask, fill_value=0, anchor_fill=anchor_fill))
        return torch.stack(rows, dim=0)
    raise ValueError(f"Unsupported position_ids shape for HSD anchor insertion: {tuple(position_ids.shape)}")


def _insert_anchor_inputs_with_layout(
    *,
    config: Safin1MoeTextConfig,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    mm_token_type_ids: torch.Tensor | None = None,
    force_layout: bool = False,
) -> dict[str, torch.Tensor | None]:
    if input_ids is None and inputs_embeds is None:
        raise ValueError("HSD anchor insertion requires input_ids or inputs_embeds.")
    ref = input_ids if input_ids is not None else inputs_embeds
    batch_size, text_len = ref.shape[:2]
    if attention_mask is None:
        text_attention_mask = torch.ones(batch_size, text_len, dtype=torch.long, device=ref.device)
    elif attention_mask.ndim == 2 and attention_mask.shape[-1] == text_len:
        if not _is_left_padding_mask(attention_mask):
            raise NotImplementedError("Safin-1 HSD supports left-padded 2D attention masks only.")
        text_attention_mask = attention_mask.to(device=ref.device, dtype=torch.long)
    elif attention_mask is not None and attention_mask.ndim == 2:
        raise ValueError(f"attention_mask length {attention_mask.shape[-1]} must match text length {text_len}.")
    else:
        raise NotImplementedError("Safin-1 HSD anchor insertion supports 2D padding masks only.")

    text_valid_source = text_attention_mask.to(dtype=torch.bool)
    has_padding = not bool(text_valid_source.all().item())
    anchor_mask, persistent_memory_mask = _build_anchor_layout_masks(
        batch_size, text_len, config, ref.device
    )
    has_anchor_tokens = bool(anchor_mask.any().item())
    if not has_anchor_tokens and not force_layout:
        return {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "mm_token_type_ids": mm_token_type_ids,
            "anchor_mask": anchor_mask,
            "hsd_valid_mask": None,
            "text_valid_mask": None,
            "anchor_valid_mask": None,
            "persistent_memory_mask": persistent_memory_mask,
        }

    if has_padding:
        return _insert_left_padded_anchor_inputs(
            config=config,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=text_attention_mask,
            position_ids=position_ids,
            mm_token_type_ids=mm_token_type_ids,
            text_valid_source=text_valid_source,
        )

    if input_ids is not None:
        input_ids = _expand_2d_with_anchor(
            input_ids,
            anchor_mask,
            fill_value=int(getattr(config, "anchor_placeholder_token_id", 0)),
        )
    if inputs_embeds is not None:
        inputs_embeds = _expand_embeds_with_anchor(inputs_embeds, anchor_mask)
    if mm_token_type_ids is not None:
        mm_token_type_ids = _expand_2d_with_anchor(mm_token_type_ids, anchor_mask, fill_value=0)
    position_ids = _expand_position_ids(position_ids, text_len, anchor_mask, config, ref.device)
    attention_mask = torch.ones(batch_size, anchor_mask.shape[1], dtype=torch.long, device=ref.device)
    return {
        "input_ids": input_ids,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "mm_token_type_ids": mm_token_type_ids,
        "anchor_mask": anchor_mask,
        "hsd_valid_mask": attention_mask.to(dtype=torch.bool),
        "text_valid_mask": (~anchor_mask).to(dtype=torch.bool),
        "anchor_valid_mask": anchor_mask.to(dtype=torch.bool),
        "persistent_memory_mask": persistent_memory_mask,
    }


def _insert_left_padded_anchor_inputs(
    *,
    config: Safin1MoeTextConfig,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None,
    mm_token_type_ids: torch.Tensor | None,
    text_valid_source: torch.Tensor,
) -> dict[str, torch.Tensor | None]:
    ref = input_ids if input_ids is not None else inputs_embeds
    batch_size, text_len = ref.shape[:2]
    device = ref.device
    chunk_size = int(getattr(config, "anchor_chunk_size", 0) or 0)
    anchor_token_num = int(config.anchor_token_num)
    if chunk_size > 0:
        anchor_counts = (
            torch.div(text_valid_source.long().sum(dim=1), chunk_size, rounding_mode="floor")
            * anchor_token_num
        )
    else:
        anchor_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
    if _persistent_memory_enabled(config):
        anchor_counts = anchor_counts + 1
    max_anchor_count = int(anchor_counts.max().item()) if anchor_counts.numel() else 0
    expanded_len = text_len + max_anchor_count
    placeholder_id = int(getattr(config, "anchor_placeholder_token_id", 0))

    anchor_mask = torch.zeros(batch_size, expanded_len, dtype=torch.bool, device=device)
    hsd_valid_mask = torch.zeros(batch_size, expanded_len, dtype=torch.bool, device=device)
    text_valid_mask = torch.zeros(batch_size, expanded_len, dtype=torch.bool, device=device)
    anchor_valid_mask = torch.zeros(batch_size, expanded_len, dtype=torch.bool, device=device)
    persistent_memory_mask = torch.zeros(batch_size, expanded_len, dtype=torch.bool, device=device)
    expanded_attention = torch.zeros(batch_size, expanded_len, dtype=attention_mask.dtype, device=device)

    expanded_ids = None
    if input_ids is not None:
        expanded_ids = torch.full(
            (batch_size, expanded_len),
            placeholder_id,
            dtype=input_ids.dtype,
            device=device,
        )
    expanded_embeds = None
    if inputs_embeds is not None:
        expanded_embeds = inputs_embeds.new_zeros(batch_size, expanded_len, inputs_embeds.shape[-1])
    expanded_mm_ids = None
    if mm_token_type_ids is not None:
        expanded_mm_ids = torch.zeros(batch_size, expanded_len, dtype=mm_token_type_ids.dtype, device=device)

    expanded_position_ids = None
    if position_ids is None:
        base_position_ids = attention_mask.to(device=device, dtype=torch.long).cumsum(dim=1) - 1
        base_position_ids = base_position_ids.masked_fill(attention_mask.to(device=device, dtype=torch.bool).logical_not(), 0)
        position_source = base_position_ids
    else:
        position_source = position_ids.to(device=device)
    if position_source.ndim == 2:
        expanded_position_ids = torch.zeros(batch_size, expanded_len, dtype=position_source.dtype, device=device)
    elif position_source.ndim == 3:
        expanded_position_ids = torch.zeros(
            position_source.shape[0],
            batch_size,
            expanded_len,
            dtype=position_source.dtype,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported position_ids shape for HSD anchor insertion: {tuple(position_source.shape)}")

    anchor_position_type = str(getattr(config, "anchor_position_ids_type", "zeros"))
    if anchor_position_type not in {"zeros", "chunk_end"}:
        raise ValueError("anchor_position_ids_type must be either 'zeros' or 'chunk_end'.")

    for batch_idx in range(batch_size):
        write = 0
        real_seen = 0
        anchor_seen = 0
        for text_idx in range(text_len):
            if (
                _persistent_memory_enabled(config)
                and anchor_seen == 0
                and bool(text_valid_source[batch_idx, text_idx].item())
            ):
                anchor_mask[batch_idx, write] = True
                anchor_valid_mask[batch_idx, write] = True
                persistent_memory_mask[batch_idx, write] = True
                hsd_valid_mask[batch_idx, write] = True
                expanded_attention[batch_idx, write] = 1
                write += 1
                anchor_seen += 1
            if expanded_ids is not None:
                expanded_ids[batch_idx, write] = input_ids[batch_idx, text_idx]
            if expanded_embeds is not None:
                expanded_embeds[batch_idx, write] = inputs_embeds[batch_idx, text_idx]
            if expanded_mm_ids is not None:
                expanded_mm_ids[batch_idx, write] = mm_token_type_ids[batch_idx, text_idx]
            if expanded_position_ids.ndim == 2:
                expanded_position_ids[batch_idx, write] = position_source[batch_idx, text_idx]
            else:
                expanded_position_ids[:, batch_idx, write] = position_source[:, batch_idx, text_idx]

            if bool(text_valid_source[batch_idx, text_idx].item()):
                text_valid_mask[batch_idx, write] = True
                hsd_valid_mask[batch_idx, write] = True
                expanded_attention[batch_idx, write] = 1
                real_seen += 1
            write += 1

            if (
                chunk_size > 0
                and real_seen > 0
                and real_seen % chunk_size == 0
                and bool(text_valid_source[batch_idx, text_idx].item())
            ):
                for _ in range(anchor_token_num):
                    anchor_mask[batch_idx, write] = True
                    anchor_valid_mask[batch_idx, write] = True
                    hsd_valid_mask[batch_idx, write] = True
                    expanded_attention[batch_idx, write] = 1
                    if expanded_position_ids.ndim == 2:
                        if anchor_position_type == "chunk_end":
                            expanded_position_ids[batch_idx, write] = position_source[batch_idx, text_idx]
                    else:
                        if anchor_position_type == "chunk_end":
                            expanded_position_ids[:, batch_idx, write] = position_source[:, batch_idx, text_idx]
                    write += 1
                    anchor_seen += 1

        while anchor_seen < max_anchor_count:
            anchor_mask[batch_idx, write] = True
            write += 1
            anchor_seen += 1
        if write != expanded_len:
            raise RuntimeError(f"internal HSD left-padding layout error: wrote {write} slots, expected {expanded_len}.")

    return {
        "input_ids": expanded_ids,
        "inputs_embeds": expanded_embeds,
        "attention_mask": expanded_attention,
        "position_ids": expanded_position_ids,
        "mm_token_type_ids": expanded_mm_ids,
        "anchor_mask": anchor_mask,
        "hsd_valid_mask": hsd_valid_mask,
        "text_valid_mask": text_valid_mask,
        "anchor_valid_mask": anchor_valid_mask,
        "persistent_memory_mask": persistent_memory_mask,
    }


def _insert_anchor_inputs(
    *,
    config: Safin1MoeTextConfig,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    mm_token_type_ids: torch.Tensor | None = None,
    force_layout: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    layout = _insert_anchor_inputs_with_layout(
        config=config,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        mm_token_type_ids=mm_token_type_ids,
        force_layout=force_layout,
    )
    return (
        layout["input_ids"],
        layout["inputs_embeds"],
        layout["attention_mask"],
        layout["position_ids"],
        layout["mm_token_type_ids"],
        layout["anchor_mask"],
    )


def _filter_anchor_hidden(hidden_states: torch.Tensor, anchor_mask: torch.Tensor | None) -> torch.Tensor:
    if anchor_mask is None or not bool(anchor_mask.any().item()):
        return hidden_states
    anchor_mask = anchor_mask.to(device=hidden_states.device, dtype=torch.bool)
    text_mask = ~anchor_mask
    if not torch.equal(text_mask.sum(dim=1), text_mask.sum(dim=1)[:1].expand(text_mask.shape[0])):
        raise NotImplementedError("Safin-1 HSD V1 requires the same anchor layout for every batch row.")
    return hidden_states[text_mask].view(hidden_states.shape[0], int(text_mask[0].sum().item()), hidden_states.shape[-1])


def _anchor_has_tokens(anchor_mask: torch.Tensor | None) -> bool:
    return anchor_mask is not None and bool(anchor_mask.any().item())


def _text_positions_from_anchor_mask(anchor_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    anchor_mask = anchor_mask.to(device=device, dtype=torch.bool)
    if not torch.equal(anchor_mask, anchor_mask[:1].expand_as(anchor_mask)):
        raise NotImplementedError("Safin-1 HSD V1 requires the same anchor layout for every batch row.")
    return (~anchor_mask[0]).nonzero(as_tuple=False).flatten()


def _filter_batch_sequence_to_text(tensor: torch.Tensor | None, anchor_mask: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None or not _anchor_has_tokens(anchor_mask):
        return tensor
    if tensor.ndim < 2:
        return tensor
    anchor_mask = anchor_mask.to(device=tensor.device, dtype=torch.bool)
    text_mask = ~anchor_mask
    if tensor.ndim >= 2 and tuple(tensor.shape[:2]) == tuple(anchor_mask.shape):
        if not torch.equal(text_mask.sum(dim=1), text_mask.sum(dim=1)[:1].expand(text_mask.shape[0])):
            raise NotImplementedError("Safin-1 HSD requires equal text slot counts across batch rows.")
        text_count = int(text_mask[0].sum().item())
        return tensor[text_mask].view(tensor.shape[0], text_count, *tensor.shape[2:])
    if tensor.ndim >= 3 and tuple(tensor.shape[1:3]) == tuple(anchor_mask.shape):
        if not torch.equal(text_mask.sum(dim=1), text_mask.sum(dim=1)[:1].expand(text_mask.shape[0])):
            raise NotImplementedError("Safin-1 HSD requires equal text slot counts across batch rows.")
        rows = []
        for item in tensor:
            rows.append(item[text_mask].view(anchor_mask.shape[0], int(text_mask[0].sum().item()), *item.shape[2:]))
        return torch.stack(rows, dim=0)
    return tensor


def _filter_position_embeddings_to_text(
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    anchor_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if position_embeddings is None or not _anchor_has_tokens(anchor_mask):
        return position_embeddings
    return tuple(_filter_batch_sequence_to_text(item, anchor_mask) for item in position_embeddings)


def _scatter_text_hidden(
    base_hidden_states: torch.Tensor,
    text_hidden_states: torch.Tensor,
    anchor_mask: torch.Tensor,
) -> torch.Tensor:
    anchor_mask = anchor_mask.to(device=base_hidden_states.device, dtype=torch.bool)
    text_mask = ~anchor_mask
    text_counts = text_mask.sum(dim=1)
    if not torch.equal(text_counts, text_counts[:1].expand_as(text_counts)):
        raise NotImplementedError("Safin-1 HSD requires equal text slot counts across batch rows.")
    text_count = int(text_counts[0].item())
    expected_shape = (base_hidden_states.shape[0], text_count, base_hidden_states.shape[-1])
    if tuple(text_hidden_states.shape) != expected_shape:
        raise ValueError(
            "HSD full-attention text output shape mismatch: "
            f"got {tuple(text_hidden_states.shape)}, expected {expected_shape}."
        )
    output = base_hidden_states.clone()
    output[text_mask] = text_hidden_states.reshape(-1, base_hidden_states.shape[-1]).to(dtype=output.dtype)
    return output


def _full_attention_mask_source(attention_mask: torch.Tensor | None, anchor_mask: torch.Tensor | None) -> torch.Tensor | None:
    if anchor_mask is None:
        return attention_mask
    if attention_mask is None:
        return None
    if not bool(anchor_mask.any().item()):
        if attention_mask.ndim == 2 and bool(attention_mask.all().item()):
            return None
        return attention_mask
    if attention_mask.ndim == 2 and bool(attention_mask.all().item()):
        return None
    if attention_mask.ndim == 2 and tuple(attention_mask.shape) == tuple(anchor_mask.shape):
        return _filter_batch_sequence_to_text(attention_mask, anchor_mask)
    if attention_mask.ndim == 2:
        return attention_mask
    raise NotImplementedError("Safin-1 HSD anchor tokens do not support custom or padded attention masks.")


def _as_tuple_output(output):
    return tuple(value for value in output.to_tuple() if value is not None)


def _is_topk_router(module: nn.Module) -> bool:
    return module.__class__.__name__ == "Qwen3_5MoeTopKRouter"


def _filter_router_logits(logits: torch.Tensor, anchor_mask: torch.Tensor | None) -> torch.Tensor:
    if anchor_mask is None or not bool(anchor_mask.any().item()):
        return logits
    anchor_mask = anchor_mask.to(device=logits.device, dtype=torch.bool)
    text_mask = ~anchor_mask
    text_counts = text_mask.sum(dim=1)
    if not torch.equal(text_counts, text_counts[:1].expand_as(text_counts)):
        raise NotImplementedError("Safin-1 HSD requires equal text slot counts across batch rows.")
    flat_text_mask = text_mask.reshape(-1)
    if logits.ndim >= 2 and logits.shape[0] == flat_text_mask.numel():
        return logits[flat_text_mask]
    if logits.ndim >= 3 and tuple(logits.shape[:2]) == tuple(anchor_mask.shape):
        return logits[text_mask].view(anchor_mask.shape[0], int(text_counts[0].item()), *logits.shape[2:])
    return logits


def _record_router_logits(model: nn.Module, enabled: bool, anchor_mask: torch.Tensor | None):
    if not enabled:
        return [], []

    router_logits = []
    handles = []

    def hook(_module, _inputs, output):
        if isinstance(output, tuple) and output:
            router_logits.append(_filter_router_logits(output[0], anchor_mask))
        elif torch.is_tensor(output):
            router_logits.append(_filter_router_logits(output, anchor_mask))

    for module in model.modules():
        if _is_topk_router(module):
            handles.append(module.register_forward_hook(hook))
    return router_logits, handles


class Safin1MoeAnchorMLP(nn.Module):
    """Dense SwiGLU MLP used only for HSD anchor-token rows."""

    def __init__(self, config: Safin1MoeTextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.intermediate_size = int(getattr(config, "hsd_anchor_mlp_hidden_size", None) or self.hidden_size)
        if self.intermediate_size <= 0:
            raise ValueError(f"hsd_anchor_mlp_hidden_size must be positive, got {self.intermediate_size}.")

        bias = bool(getattr(config, "add_bias_linear", False))
        self.linear_fc1 = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=bias)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        nn.init.normal_(self.linear_fc1.weight, mean=0.0, std=config.initializer_range)
        nn.init.normal_(self.linear_fc2.weight, mean=0.0, std=config.initializer_range)
        if self.linear_fc1.bias is not None:
            nn.init.zeros_(self.linear_fc1.bias)
        if self.linear_fc2.bias is not None:
            nn.init.zeros_(self.linear_fc2.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        intermediate = self.linear_fc1(hidden_states)
        gate, value = intermediate.chunk(2, dim=-1)
        clamp_value = getattr(self.config, "activation_func_clamp_value", None)
        if clamp_value is not None:
            gate = gate.clamp(max=float(clamp_value))
            value = value.clamp(min=-float(clamp_value), max=float(clamp_value))
        value = value + float(getattr(self.config, "glu_linear_offset", 0.0) or 0.0)
        return self.linear_fc2(F.silu(gate) * value)


def _run_layer_mlp_delta(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    anchor_mask: torch.Tensor | None,
    anchor_mlp_input_states: torch.Tensor | None = None,
    hsd_valid_mask: torch.Tensor | None = None,
    text_valid_mask: torch.Tensor | None = None,
    anchor_valid_mask: torch.Tensor | None = None,
):
    config = getattr(layer, "_historical_hsd_config", getattr(layer, "config", None))

    def _run_standard_mlp(states: torch.Tensor) -> torch.Tensor:
        output = layer.mlp(states)
        return output[0] if isinstance(output, tuple) else output

    def _run_standard_mlp_by_mask(mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=hidden_states.device, dtype=torch.bool)
        if tuple(mask.shape) != tuple(hidden_states.shape[:2]):
            raise ValueError(f"MLP valid mask shape {tuple(mask.shape)} must match hidden states {tuple(hidden_states.shape[:2])}.")
        output = hidden_states.new_zeros(hidden_states.shape)
        for batch_idx in range(hidden_states.shape[0]):
            row_mask = mask[batch_idx]
            if not bool(row_mask.any().item()):
                continue
            row_hidden = hidden_states[batch_idx : batch_idx + 1, row_mask]
            row_output = _run_standard_mlp(row_hidden)
            if not torch.is_tensor(row_output):
                raise TypeError("HSD MLP requires the layer MLP to return a tensor.")
            output[batch_idx, row_mask] = row_output.reshape(-1, hidden_states.shape[-1]).to(dtype=output.dtype)
        return output

    if hsd_valid_mask is not None:
        hsd_valid_mask = hsd_valid_mask.to(device=hidden_states.device, dtype=torch.bool)
        if tuple(hsd_valid_mask.shape) != tuple(hidden_states.shape[:2]):
            raise ValueError(
                f"hsd_valid_mask shape {tuple(hsd_valid_mask.shape)} must match hidden states {tuple(hidden_states.shape[:2])}."
            )
        if not bool(hsd_valid_mask.all().item()) and (config is None or not _text_only_mlp_enabled(config)):
            return _run_standard_mlp_by_mask(hsd_valid_mask)

    if config is None or not _text_only_mlp_enabled(config):
        return _run_standard_mlp(hidden_states)
    if anchor_mask is None or not bool(anchor_mask.any().item()):
        if text_valid_mask is not None and not bool(text_valid_mask.to(device=hidden_states.device, dtype=torch.bool).all().item()):
            return _run_standard_mlp_by_mask(text_valid_mask)
        return _run_standard_mlp(hidden_states)

    anchor_mask = anchor_mask.to(device=hidden_states.device, dtype=torch.bool)
    if tuple(anchor_mask.shape) != tuple(hidden_states.shape[:2]):
        raise ValueError(f"anchor_mask shape {tuple(anchor_mask.shape)} must match hidden states {tuple(hidden_states.shape[:2])}.")

    batch_size, _, hidden_size = hidden_states.shape
    if text_valid_mask is None:
        text_mask = ~anchor_mask if hsd_valid_mask is None else (~anchor_mask) & hsd_valid_mask
    else:
        text_mask = text_valid_mask.to(device=hidden_states.device, dtype=torch.bool)
    if anchor_valid_mask is None:
        active_anchor_mask = anchor_mask if hsd_valid_mask is None else anchor_mask & hsd_valid_mask
    else:
        active_anchor_mask = anchor_valid_mask.to(device=hidden_states.device, dtype=torch.bool)
    text_counts = text_mask.sum(dim=1)
    anchor_counts = active_anchor_mask.sum(dim=1)
    uniform_text = torch.equal(text_counts, text_counts[:1].expand_as(text_counts))
    uniform_anchor = torch.equal(anchor_counts, anchor_counts[:1].expand_as(anchor_counts))
    if not uniform_text or not uniform_anchor:
        output = hidden_states.new_zeros(hidden_states.shape)
        for batch_idx in range(batch_size):
            row_text_mask = text_mask[batch_idx]
            if bool(row_text_mask.any().item()):
                row_hidden = hidden_states[batch_idx : batch_idx + 1, row_text_mask]
                row_output = _run_standard_mlp(row_hidden)
                if not torch.is_tensor(row_output):
                    raise TypeError("HSD text-only MLP requires the layer MLP to return a tensor.")
                output[batch_idx, row_text_mask] = row_output.reshape(-1, hidden_size).to(dtype=output.dtype)
        if _anchor_mlp_enabled(config) and bool(active_anchor_mask.any().item()):
            anchor_mlp = getattr(layer, "hsd_anchor_mlp", None)
            if anchor_mlp is None:
                raise RuntimeError("HSD anchor MLP is enabled but layer.hsd_anchor_mlp was not initialized.")
            anchor_input = hidden_states if anchor_mlp_input_states is None else anchor_mlp_input_states
            if tuple(anchor_input.shape) != tuple(hidden_states.shape):
                raise ValueError(
                    "HSD anchor MLP input states must match MLP hidden states, "
                    f"got {tuple(anchor_input.shape)} vs {tuple(hidden_states.shape)}."
                )
            for batch_idx in range(batch_size):
                row_anchor_mask = active_anchor_mask[batch_idx]
                if not bool(row_anchor_mask.any().item()):
                    continue
                anchor_hidden = anchor_input[batch_idx : batch_idx + 1, row_anchor_mask]
                anchor_output = anchor_mlp(anchor_hidden)
                if tuple(anchor_output.shape) != tuple(anchor_hidden.shape):
                    raise ValueError(
                        "HSD anchor MLP output shape must match input shape, "
                        f"got {tuple(anchor_output.shape)} vs {tuple(anchor_hidden.shape)}."
                    )
                output[batch_idx, row_anchor_mask] = anchor_output.reshape(-1, hidden_size).to(dtype=output.dtype)
        return output
    text_count = int(text_counts[0].item())
    anchor_count = int(anchor_counts[0].item())
    anchor_mlp_enabled = _anchor_mlp_enabled(config)
    if text_count == 0 and not anchor_mlp_enabled:
        raise ValueError("HSD text-only MLP requires at least one text token.")

    output = hidden_states.new_zeros(hidden_states.shape)
    if text_count > 0:
        text_hidden = hidden_states[text_mask].view(batch_size, text_count, hidden_size)
        text_output = layer.mlp(text_hidden)
        if isinstance(text_output, tuple):
            text_output = text_output[0]
        if not torch.is_tensor(text_output):
            raise TypeError("HSD text-only MLP requires the layer MLP to return a tensor.")
        output[text_mask] = text_output.reshape(-1, hidden_size).to(dtype=output.dtype)

    if anchor_mlp_enabled and anchor_count > 0:
        anchor_mlp = getattr(layer, "hsd_anchor_mlp", None)
        if anchor_mlp is None:
            raise RuntimeError("HSD anchor MLP is enabled but layer.hsd_anchor_mlp was not initialized.")
        anchor_input = hidden_states if anchor_mlp_input_states is None else anchor_mlp_input_states
        if tuple(anchor_input.shape) != tuple(hidden_states.shape):
            raise ValueError(
                "HSD anchor MLP input states must match MLP hidden states, "
                f"got {tuple(anchor_input.shape)} vs {tuple(hidden_states.shape)}."
            )
        anchor_hidden = anchor_input[active_anchor_mask].view(batch_size, anchor_count, hidden_size)
        anchor_output = anchor_mlp(anchor_hidden)
        if tuple(anchor_output.shape) != tuple(anchor_hidden.shape):
            raise ValueError(
                "HSD anchor MLP output shape must match input shape, "
                f"got {tuple(anchor_output.shape)} vs {tuple(anchor_hidden.shape)}."
            )
        output[active_anchor_mask] = anchor_output.reshape(-1, hidden_size).to(dtype=output.dtype)
    return output


class _RowFullAttentionCache:
    def __init__(self, layer_idx: int, keys: torch.Tensor | None = None, values: torch.Tensor | None = None):
        self.layer_idx = int(layer_idx)
        self.keys = keys
        self.values = values

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args, **kwargs):
        if int(layer_idx) != self.layer_idx:
            raise ValueError(f"row full-attention cache received layer {layer_idx}, expected {self.layer_idx}.")
        if self.keys is None or self.values is None or self.keys.numel() == 0:
            self.keys = key_states
            self.values = value_states
        else:
            self.keys = torch.cat([self.keys.to(key_states.device), key_states], dim=-2)
            self.values = torch.cat([self.values.to(value_states.device), value_states], dim=-2)
        return self.keys, self.values


def _full_attention_layer_idx(layer: nn.Module) -> int:
    attn = getattr(layer, "self_attn", None)
    return int(getattr(attn, "layer_idx", getattr(layer, "_historical_hsd_layer_idx", getattr(layer, "layer_idx", 0))))


def _slice_row_full_attention_cache(past_key_values, layer_idx: int, batch_idx: int, batch_size: int) -> _RowFullAttentionCache | None:
    if past_key_values is None:
        return None
    layers = getattr(past_key_values, "layers", None)
    if layers is None or len(layers) <= layer_idx:
        return _RowFullAttentionCache(layer_idx)
    layer_cache = layers[layer_idx]
    keys = getattr(layer_cache, "keys", None)
    values = getattr(layer_cache, "values", None)
    if not torch.is_tensor(keys) or not torch.is_tensor(values) or keys.numel() == 0:
        return _RowFullAttentionCache(layer_idx)
    lengths = getattr(layer_cache, "hsd_text_cache_lengths_per_row", None)
    if torch.is_tensor(lengths) and lengths.numel() > batch_idx:
        row_len = int(lengths.reshape(-1)[batch_idx].item())
    else:
        row_len = int(keys.shape[-2])
    return _RowFullAttentionCache(
        layer_idx,
        keys[batch_idx : batch_idx + 1, :, :row_len].contiguous(),
        values[batch_idx : batch_idx + 1, :, :row_len].contiguous(),
    )


def _merge_row_full_attention_caches(past_key_values, layer_idx: int, row_caches: list[_RowFullAttentionCache | None]) -> None:
    if past_key_values is None:
        return
    layers = getattr(past_key_values, "layers", None)
    if layers is None or len(layers) <= layer_idx:
        return
    keyed = [cache for cache in row_caches if cache is not None and torch.is_tensor(cache.keys)]
    if not keyed:
        return
    sample_keys = keyed[0].keys
    sample_values = keyed[0].values
    max_len = max(int(cache.keys.shape[-2]) if cache is not None and torch.is_tensor(cache.keys) else 0 for cache in row_caches)
    batch_size = len(row_caches)
    merged_keys = sample_keys.new_zeros(batch_size, sample_keys.shape[1], max_len, sample_keys.shape[-1])
    merged_values = sample_values.new_zeros(batch_size, sample_values.shape[1], max_len, sample_values.shape[-1])
    lengths = torch.zeros(batch_size, dtype=torch.long, device=merged_keys.device)
    for batch_idx, cache in enumerate(row_caches):
        if cache is None or not torch.is_tensor(cache.keys) or not torch.is_tensor(cache.values):
            continue
        row_len = int(cache.keys.shape[-2])
        merged_keys[batch_idx : batch_idx + 1, :, :row_len] = cache.keys.to(merged_keys.dtype)
        merged_values[batch_idx : batch_idx + 1, :, :row_len] = cache.values.to(merged_values.dtype)
        lengths[batch_idx] = row_len
    layer_cache = layers[layer_idx]
    layer_cache.keys = merged_keys
    layer_cache.values = merged_values
    layer_cache.is_initialized = True
    layer_cache.dtype = merged_keys.dtype
    layer_cache.device = merged_keys.device
    layer_cache.hsd_text_cache_lengths_per_row = lengths


def _run_full_attention_padded_fallback(
    layer: nn.Module,
    *,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    past_key_values,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    anchor_mask: torch.Tensor,
    text_valid_mask: torch.Tensor | None,
    **kwargs,
) -> torch.Tensor | None:
    if text_valid_mask is None:
        return None
    text_hidden_states = _filter_anchor_hidden(hidden_states, anchor_mask)
    text_valid_slots = _filter_batch_sequence_to_text(text_valid_mask, anchor_mask).to(
        device=text_hidden_states.device,
        dtype=torch.bool,
    )
    layer_idx = _full_attention_layer_idx(layer)
    cached_per_row = False
    cache_layer = None
    if past_key_values is not None:
        layers = getattr(past_key_values, "layers", None)
        cache_layer = layers[layer_idx] if layers is not None and len(layers) > layer_idx else None
        cached_per_row = torch.is_tensor(getattr(cache_layer, "hsd_text_cache_lengths_per_row", None))
    if bool(text_valid_slots.all().item()) and not cached_per_row:
        return None

    text_position_ids = _filter_batch_sequence_to_text(position_ids, anchor_mask)
    text_position_embeddings = _filter_position_embeddings_to_text(position_embeddings, anchor_mask)
    output = text_hidden_states.new_zeros(text_hidden_states.shape)
    row_caches: list[_RowFullAttentionCache | None] = []
    for batch_idx in range(text_hidden_states.shape[0]):
        valid = text_valid_slots[batch_idx]
        if not bool(valid.any().item()):
            raise ValueError("Safin-1 full-attention fallback requires at least one valid text token.")
        row_hidden = text_hidden_states[batch_idx : batch_idx + 1, valid]
        row_position_ids = None
        if torch.is_tensor(text_position_ids):
            row_position_ids = text_position_ids[batch_idx : batch_idx + 1, valid]
        row_position_embeddings = tuple(item[batch_idx : batch_idx + 1, valid] for item in text_position_embeddings)
        row_cache = _slice_row_full_attention_cache(past_key_values, layer_idx, batch_idx, text_hidden_states.shape[0])
        row_output, _ = layer.self_attn(
            hidden_states=row_hidden,
            attention_mask=None,
            position_ids=row_position_ids,
            past_key_values=row_cache,
            position_embeddings=row_position_embeddings,
            **kwargs,
        )
        output[batch_idx, valid] = row_output.reshape(-1, output.shape[-1]).to(output.dtype)
        row_caches.append(row_cache)
    _merge_row_full_attention_caches(past_key_values, layer_idx, row_caches)
    return _scatter_text_hidden(hidden_states, output, anchor_mask)


def _historical_hsd_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    anchor_mask: torch.Tensor | None = None,
    hsd_valid_mask: torch.Tensor | None = None,
    text_valid_mask: torch.Tensor | None = None,
    anchor_valid_mask: torch.Tensor | None = None,
    persistent_memory_mask: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    layer_idx = int(getattr(self, "_historical_hsd_layer_idx", getattr(self, "layer_idx", 0)))
    trace_anchor_mask = anchor_mask if anchor_mask is not None and tuple(anchor_mask.shape) == tuple(hidden_states.shape[:2]) else None
    trace_hsd_internal(layer_idx, "decoder_input", _filter_anchor_hidden(hidden_states, trace_anchor_mask))
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    trace_hsd_internal(layer_idx, "input_layernorm_out", _filter_anchor_hidden(hidden_states, trace_anchor_mask))

    if self.layer_type == "linear_attention" or getattr(self, "_historical_hsd_self_attn_replaced", False):
        hsd_attn = self.linear_attn if self.layer_type == "linear_attention" else self.self_attn
        hidden_states = hsd_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
            anchor_mask=anchor_mask,
            hsd_valid_mask=hsd_valid_mask,
            text_valid_mask=text_valid_mask,
            anchor_valid_mask=anchor_valid_mask,
            persistent_memory_mask=persistent_memory_mask,
            **kwargs,
        )
    elif self.layer_type == "full_attention":
        full_attention_anchor_tokens = bool(
            getattr(getattr(self, "_historical_hsd_config", None), "hsd_full_attention_anchor_tokens", False)
        )
        padded_full_attention = None
        if anchor_mask is not None and not full_attention_anchor_tokens:
            padded_full_attention = _run_full_attention_padded_fallback(
                self,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                anchor_mask=anchor_mask,
                text_valid_mask=text_valid_mask,
                **kwargs,
            )
        if padded_full_attention is not None:
            hidden_states = padded_full_attention
        elif _anchor_has_tokens(anchor_mask) and not full_attention_anchor_tokens:
            text_hidden_states = _filter_anchor_hidden(hidden_states, anchor_mask)
            text_position_ids = _filter_batch_sequence_to_text(position_ids, anchor_mask)
            text_position_embeddings = _filter_position_embeddings_to_text(position_embeddings, anchor_mask)
            text_hidden_states, _ = self.self_attn(
                hidden_states=text_hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                position_embeddings=text_position_embeddings,
                **kwargs,
            )
            hidden_states = _scatter_text_hidden(hidden_states, text_hidden_states, anchor_mask)
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )

    trace_hsd_internal(layer_idx, "attention_raw_out", _filter_anchor_hidden(hidden_states, trace_anchor_mask))
    hidden_states = residual + hidden_states
    trace_hsd_internal(layer_idx, "after_attn_residual", _filter_anchor_hidden(hidden_states, trace_anchor_mask))
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    trace_hsd_internal(layer_idx, "post_attention_layernorm_out", _filter_anchor_hidden(hidden_states, trace_anchor_mask))
    hidden_states = _run_layer_mlp_delta(
        self,
        hidden_states,
        anchor_mask,
        anchor_mlp_input_states=residual,
        hsd_valid_mask=hsd_valid_mask,
        text_valid_mask=text_valid_mask,
        anchor_valid_mask=anchor_valid_mask,
    )
    trace_hsd_internal(layer_idx, "mlp_out", _filter_anchor_hidden(hidden_states, trace_anchor_mask))
    hidden_states = residual + hidden_states
    trace_hsd_internal(layer_idx, "decoder_output", _filter_anchor_hidden(hidden_states, trace_anchor_mask))
    return hidden_states


def _reorder_batch_tensor(value, beam_idx: torch.LongTensor):
    if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == beam_idx.shape[0]:
        return value.index_select(0, beam_idx.to(value.device))
    if isinstance(value, list):
        return [_reorder_batch_tensor(item, beam_idx) for item in value]
    if isinstance(value, tuple):
        return tuple(_reorder_batch_tensor(item, beam_idx) for item in value)
    return value


def _reorder_hsd_sidecar_cache(past_key_values, beam_idx: torch.LongTensor):
    if past_key_values is None:
        return past_key_values

    for attr_name in ("hsd_seen_text_tokens_per_row", "hsd_seen_expanded_tokens_per_row"):
        if hasattr(past_key_values, attr_name):
            setattr(past_key_values, attr_name, _reorder_batch_tensor(getattr(past_key_values, attr_name), beam_idx))

    for attr_name in ("_hsd_conv_states", "_hsd_recurrent_states"):
        states = getattr(past_key_values, attr_name, None)
        if states is None:
            continue
        for idx, value in enumerate(states):
            states[idx] = _reorder_batch_tensor(value, beam_idx)

    hsd_layers = getattr(past_key_values, "_hsd_layer_states", None)
    if hsd_layers is not None:
        for layer_state in hsd_layers:
            for attr_name in (
                "hsd_seen_text_tokens_per_row",
                "hsd_seen_expanded_tokens_per_row",
                "hsd_anchor_states",
                "hsd_anchor_keys",
                "hsd_anchor_boundaries",
                "hsd_anchor_valid_mask",
                "hsd_anchor_persistent_memory_mask",
            ):
                if hasattr(layer_state, attr_name):
                    setattr(layer_state, attr_name, _reorder_batch_tensor(getattr(layer_state, attr_name), beam_idx))

    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        for layer in layers:
            for attr_name in (
                "conv_states",
                "recurrent_states",
                "hsd_seen_text_tokens_per_row",
                "hsd_seen_expanded_tokens_per_row",
                "hsd_anchor_states",
                "hsd_anchor_keys",
                "hsd_anchor_boundaries",
                "hsd_anchor_valid_mask",
                "hsd_anchor_persistent_memory_mask",
                "hsd_text_cache_lengths_per_row",
            ):
                if hasattr(layer, attr_name):
                    setattr(layer, attr_name, _reorder_batch_tensor(getattr(layer, attr_name), beam_idx))
    return past_key_values


def replace_linear_attention_layers(language_model: Qwen3_5MoeTextModel, config: Safin1MoeTextConfig) -> None:
    for layer_idx, layer in enumerate(language_model.layers):
        layer_type = config.layer_types[layer_idx]
        layer._historical_hsd_config = config
        layer._historical_hsd_layer_idx = int(layer_idx)
        if config.layer_types[layer_idx] == "linear_attention":
            layer.linear_attn = Safin1MoeGatedDeltaNet(config, layer_idx)
            if _anchor_mlp_enabled(config) and not hasattr(layer, "hsd_anchor_mlp"):
                layer.hsd_anchor_mlp = Safin1MoeAnchorMLP(config)
            if _anchor_enabled(config) and not getattr(layer, "_historical_hsd_linear_patched", False):
                layer.forward = MethodType(_historical_hsd_decoder_layer_forward, layer)
                layer._historical_hsd_linear_patched = True
                layer._historical_hsd_accepts_anchor_mask = True
        elif layer_type == "full_attention" and _anchor_enabled(config):
            if _anchor_mlp_enabled(config) and not hasattr(layer, "hsd_anchor_mlp"):
                layer.hsd_anchor_mlp = Safin1MoeAnchorMLP(config)
            if not getattr(layer, "_historical_hsd_full_attention_patched", False):
                layer.forward = MethodType(_historical_hsd_decoder_layer_forward, layer)
                layer._historical_hsd_full_attention_patched = True
                layer._historical_hsd_accepts_anchor_mask = True


def _update_linear_attn_mask(model: Qwen3_5MoeTextModel, attention_mask, cache_position, past_key_values):
    try:
        parameters = list(inspect.signature(model._update_linear_attn_mask).parameters)
        if len(parameters) >= 2 and parameters[1] == "cache_position":
            return model._update_linear_attn_mask(attention_mask, cache_position)
        return model._update_linear_attn_mask(attention_mask, past_key_values)
    except (TypeError, ValueError):
        pass
    try:
        return model._update_linear_attn_mask(attention_mask, cache_position)
    except (TypeError, AttributeError):
        return model._update_linear_attn_mask(attention_mask, past_key_values)


def _create_qwen_causal_mask(config, inputs_embeds, attention_mask, cache_position, past_key_values, position_ids):
    try:
        parameters = inspect.signature(create_causal_mask).parameters
    except (TypeError, ValueError):
        parameters = {}
    embed_kwarg = "inputs_embeds" if "inputs_embeds" in parameters else "input_embeds"
    try:
        return create_causal_mask(
            config=config,
            **{embed_kwarg: inputs_embeds},
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
    except TypeError:
        return create_causal_mask(
            config=config,
            **{"input_embeds" if embed_kwarg == "inputs_embeds" else "inputs_embeds": inputs_embeds},
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )


def _decoder_layer_accepts_cache_position(decoder_layer) -> bool:
    try:
        param = inspect.signature(decoder_layer.forward).parameters.get("cache_position")
    except (TypeError, ValueError):
        return False
    return param is not None and param.kind is not inspect.Parameter.VAR_KEYWORD


class Safin1MoeTextModel(Qwen3_5MoeTextModel):
    """Official text model with optional HSD anchor-token support."""

    config: Safin1MoeTextConfig
    config_class = Safin1MoeTextConfig

    def __init__(self, config: Safin1MoeTextConfig):
        super().__init__(config)
        if _anchor_enabled(config) and str(getattr(config, "anchor_embedding_type", "learned")) == "learned":
            self.anchor_embedding = nn.Embedding(int(config.anchor_token_num), config.hidden_size)
            with torch.no_grad():
                self._initialize_anchor_embedding_weight()
        if getattr(config, "hist_replace_linear_attn", True):
            replace_linear_attention_layers(self, config)

    def _initialize_anchor_embedding_weight(self) -> None:
        init_from = str(getattr(self.config, "anchor_embedding_init_from", "random"))
        if init_from == "random":
            nn.init.normal_(self.anchor_embedding.weight, mean=0.0, std=self.config.initializer_range)
        else:
            embeddings = self.get_input_embeddings()
            weight = getattr(embeddings, "weight", None)
            if weight is None:
                raise RuntimeError("Cannot initialize HSD anchor embedding because token embeddings are missing.")
            if init_from in {"placeholder", "bos"}:
                if init_from == "bos":
                    bos_token_id = getattr(self.config, "anchor_bos_token_id", None)
                    if bos_token_id is None:
                        bos_token_id = getattr(self.config, "bos_token_id", None)
                    if bos_token_id is None:
                        raise ValueError(
                            "anchor_embedding_init_from='bos' requires config.anchor_bos_token_id "
                            "or config.bos_token_id."
                        )
                    token_id = int(bos_token_id)
                else:
                    token_id = int(getattr(self.config, "anchor_placeholder_token_id", 0) or 0)
                if token_id < 0 or token_id >= weight.shape[0]:
                    raise ValueError(
                        f"anchor {init_from} token id {token_id} is outside the embedding table "
                        f"with vocab size {weight.shape[0]}."
                    )
                vector = weight[token_id].detach()
            elif init_from == "vocab_mean":
                vector = weight.detach().float().mean(dim=0)
            else:
                raise ValueError("anchor_embedding_init_from must be one of random, placeholder, bos, or vocab_mean.")
            vector = vector.to(device=self.anchor_embedding.weight.device, dtype=self.anchor_embedding.weight.dtype)
            self.anchor_embedding.weight.copy_(vector.view(1, -1).expand_as(self.anchor_embedding.weight))

        init_scale = getattr(self.config, "anchor_embedding_init_scale", None)
        if init_scale is not None:
            self.anchor_embedding.weight.mul_(float(init_scale))

    def _apply_anchor_embedding(self, inputs_embeds: torch.Tensor, anchor_mask: torch.Tensor | None) -> torch.Tensor:
        if anchor_mask is None or not bool(anchor_mask.any().item()):
            return inputs_embeds
        anchor_mask = anchor_mask.to(device=inputs_embeds.device, dtype=torch.bool)
        anchor_embedding_type = str(getattr(self.config, "anchor_embedding_type", "learned"))
        if anchor_embedding_type == "placeholder":
            return inputs_embeds
        output = inputs_embeds.clone()
        if anchor_embedding_type == "learned":
            if not hasattr(self, "anchor_embedding"):
                raise ValueError("learned HSD anchor embeddings require anchor_token_num > 0.")
            anchor_ordinals = anchor_mask.long().cumsum(dim=1).sub(1)
            anchor_ids = anchor_ordinals[anchor_mask].remainder(int(self.config.anchor_token_num))
            replacements = self.anchor_embedding(anchor_ids.to(device=self.anchor_embedding.weight.device))
            output[anchor_mask] = replacements.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            return output
        if anchor_embedding_type == "chunk_end":
            anchors = _anchor_text_indices(anchor_mask).to(inputs_embeds.device)
            anchor_embeds = inputs_embeds.gather(
                1,
                anchors.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1]),
            )
            output[anchor_mask] = anchor_embeds.reshape(-1, inputs_embeds.shape[-1])
            return output
        raise ValueError(f"Unsupported anchor_embedding_type: {anchor_embedding_type}")

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        anchor_mask: torch.Tensor | None = None,
        hsd_valid_mask: torch.Tensor | None = None,
        text_valid_mask: torch.Tensor | None = None,
        anchor_valid_mask: torch.Tensor | None = None,
        persistent_memory_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> Qwen3_5MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_hidden_states = kwargs.pop("output_hidden_states", None)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else getattr(self.config, "output_hidden_states", False)
        )
        output_hidden_states = bool(output_hidden_states)
        output_attentions = bool(kwargs.pop("output_attentions", False))
        if output_attentions:
            raise NotImplementedError("Safin-1 HSD does not expose attentions for linear layers.")
        output_router_logits = kwargs.pop("output_router_logits", None)
        output_router_logits = (
            output_router_logits if output_router_logits is not None else getattr(self.config, "output_router_logits", False)
        )
        output_router_logits = bool(output_router_logits)
        return_dict = kwargs.pop("return_dict", None)
        return_dict = return_dict if return_dict is not None else _config_return_dict(self.config)
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        cache_position = kwargs.pop("cache_position", cache_position)

        force_layout = bool(use_cache and _anchor_enabled(self.config))
        if anchor_mask is None and _anchor_enabled(self.config) and not _cache_has_previous_state(past_key_values):
            layout = _insert_anchor_inputs_with_layout(
                config=self.config,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                force_layout=force_layout,
            )
            input_ids = layout["input_ids"]
            inputs_embeds = layout["inputs_embeds"]
            attention_mask = layout["attention_mask"]
            position_ids = layout["position_ids"]
            anchor_mask = layout["anchor_mask"]
            hsd_valid_mask = layout["hsd_valid_mask"]
            text_valid_mask = layout["text_valid_mask"]
            anchor_valid_mask = layout["anchor_valid_mask"]
            persistent_memory_mask = layout["persistent_memory_mask"]
            expected_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
            if cache_position is not None and int(cache_position.numel()) != int(expected_len):
                cache_position = None

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if anchor_mask is None and use_cache and _anchor_enabled(self.config) and _cache_has_previous_state(past_key_values):
            anchor_mask = torch.zeros(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
            hsd_valid_mask = torch.ones_like(anchor_mask, dtype=torch.bool)
            text_valid_mask = ~anchor_mask
            anchor_valid_mask = anchor_mask
            persistent_memory_mask = torch.zeros_like(anchor_mask)
        if anchor_mask is not None:
            anchor_mask = anchor_mask.to(device=inputs_embeds.device, dtype=torch.bool)
            if hsd_valid_mask is not None:
                hsd_valid_mask = hsd_valid_mask.to(device=inputs_embeds.device, dtype=torch.bool)
            if text_valid_mask is not None:
                text_valid_mask = text_valid_mask.to(device=inputs_embeds.device, dtype=torch.bool)
            if anchor_valid_mask is not None:
                anchor_valid_mask = anchor_valid_mask.to(device=inputs_embeds.device, dtype=torch.bool)
            if persistent_memory_mask is not None:
                persistent_memory_mask = persistent_memory_mask.to(device=inputs_embeds.device, dtype=torch.bool)
            inputs_embeds = self._apply_anchor_embedding(inputs_embeds, anchor_mask)

        if use_cache and past_key_values is None:
            past_key_values = _new_qwen35_cache(self.config)
        elif use_cache:
            past_key_values = _patch_hsd_cache_get_seq_length(past_key_values)
        batch_size = int(inputs_embeds.shape[0])
        past_text_before_per_row = (
            _cache_text_lengths(past_key_values, batch_size, inputs_embeds.device)
            if use_cache and past_key_values is not None
            else torch.zeros(batch_size, dtype=torch.long, device=inputs_embeds.device)
        )
        past_expanded_before_per_row = (
            _cache_expanded_lengths(past_key_values, batch_size, inputs_embeds.device)
            if use_cache and past_key_values is not None
            else torch.zeros(batch_size, dtype=torch.long, device=inputs_embeds.device)
        )
        past_expanded_before = int(past_expanded_before_per_row.max().item()) if past_expanded_before_per_row.numel() else 0

        if cache_position is None:
            cache_position = torch.arange(
                past_expanded_before,
                past_expanded_before + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids[1:]
        else:
            text_position_ids = None
            rope_position_ids = position_ids

        active_layer_types = self.config.layer_types[: self.config.num_hidden_layers]
        has_full_attention = any(layer_type != "linear_attention" for layer_type in active_layer_types)
        causal_mask = None
        full_attention_anchor_tokens = bool(
            getattr(self.config, "hsd_full_attention_anchor_tokens", False)
        )
        full_attention_inputs_embeds = (
            inputs_embeds
            if full_attention_anchor_tokens
            else _filter_anchor_hidden(inputs_embeds, anchor_mask)
        )
        full_attention_position_ids = (
            text_position_ids
            if full_attention_anchor_tokens
            else _filter_batch_sequence_to_text(text_position_ids, anchor_mask)
        )
        full_attention_mask = (
            attention_mask
            if full_attention_anchor_tokens
            else _full_attention_mask_source(attention_mask, anchor_mask)
        )
        full_attention_cache_position = cache_position
        if (not full_attention_anchor_tokens and cache_position is not None
                and int(cache_position.numel()) != int(full_attention_inputs_embeds.shape[1])):
            if torch.is_tensor(full_attention_mask) and full_attention_mask.ndim == 2:
                full_start = int(full_attention_mask.shape[-1]) - int(full_attention_inputs_embeds.shape[1])
                full_attention_cache_position = torch.arange(
                    full_start,
                    full_start + int(full_attention_inputs_embeds.shape[1]),
                    dtype=cache_position.dtype,
                    device=cache_position.device,
                )
            elif _anchor_has_tokens(anchor_mask):
                try:
                    text_positions = _text_positions_from_anchor_mask(anchor_mask, cache_position.device)
                    full_attention_cache_position = cache_position.index_select(0, text_positions)
                except NotImplementedError:
                    full_start = int(past_text_before_per_row.max().item()) if past_text_before_per_row.numel() else 0
                    full_attention_cache_position = torch.arange(
                        full_start,
                        full_start + int(full_attention_inputs_embeds.shape[1]),
                        dtype=cache_position.dtype,
                        device=cache_position.device,
                    )
        if has_full_attention:
            causal_mask = _create_qwen_causal_mask(
                config=self.config,
                inputs_embeds=full_attention_inputs_embeds,
                attention_mask=full_attention_mask,
                cache_position=full_attention_cache_position,
                past_key_values=past_key_values,
                position_ids=full_attention_position_ids,
            )
        linear_attention_mask_source = attention_mask
        if (
            torch.is_tensor(linear_attention_mask_source)
            and linear_attention_mask_source.ndim == 2
            and linear_attention_mask_source.shape[-1] != inputs_embeds.shape[1]
        ):
            linear_attention_mask_source = None
        linear_attn_mask = _update_linear_attn_mask(self, linear_attention_mask_source, cache_position, past_key_values)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, rope_position_ids)
        all_hidden_states = () if output_hidden_states else None
        router_logits, router_handles = _record_router_logits(self, output_router_logits, anchor_mask)
        try:
            for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
                if output_hidden_states:
                    all_hidden_states += (_filter_anchor_hidden(hidden_states, anchor_mask),)
                is_linear_attention = self.config.layer_types[i] == "linear_attention"
                layer_mask = linear_attn_mask if is_linear_attention else causal_mask
                layer_position_ids = text_position_ids if is_linear_attention else full_attention_position_ids
                layer_cache_position = cache_position if is_linear_attention else full_attention_cache_position
                layer_kwargs = dict(
                    position_embeddings=position_embeddings,
                    attention_mask=layer_mask,
                    position_ids=layer_position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    **kwargs,
                )
                if is_linear_attention or getattr(decoder_layer, "_historical_hsd_accepts_anchor_mask", False):
                    layer_kwargs["anchor_mask"] = anchor_mask
                    layer_kwargs["hsd_valid_mask"] = hsd_valid_mask
                    layer_kwargs["text_valid_mask"] = text_valid_mask
                    layer_kwargs["anchor_valid_mask"] = anchor_valid_mask
                    layer_kwargs["persistent_memory_mask"] = persistent_memory_mask
                if _decoder_layer_accepts_cache_position(decoder_layer):
                    layer_kwargs["cache_position"] = layer_cache_position
                hidden_states = decoder_layer(hidden_states, **layer_kwargs)
        finally:
            for handle in router_handles:
                handle.remove()

        final_trace_layer = int(self.config.num_hidden_layers)
        trace_hsd_internal(
            final_trace_layer,
            "final_norm_input",
            _filter_anchor_hidden(hidden_states, anchor_mask),
        )
        hidden_states = self.norm(hidden_states)
        hidden_states = _filter_anchor_hidden(hidden_states, anchor_mask)
        trace_hsd_internal(final_trace_layer, "final_norm_out", hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if use_cache and past_key_values is not None and anchor_mask is not None:
            if text_valid_mask is None:
                current_text_per_row = (~anchor_mask).long().sum(dim=1)
            else:
                current_text_per_row = text_valid_mask.long().sum(dim=1)
            if hsd_valid_mask is None:
                current_expanded_per_row = torch.full_like(current_text_per_row, int(inputs_embeds.shape[1]))
            else:
                current_expanded_per_row = hsd_valid_mask.long().sum(dim=1)
            _set_root_cache_lengths(
                past_key_values,
                text_lengths=past_text_before_per_row + current_text_per_row.to(past_text_before_per_row.device),
                expanded_lengths=past_expanded_before_per_row
                + current_expanded_per_row.to(past_expanded_before_per_row.device),
            )

        outputs = Qwen3_5MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=None,
            router_logits=tuple(router_logits) if output_router_logits else None,
        )
        if not return_dict:
            return _as_tuple_output(outputs)
        return outputs


class Safin1MoeModel(Qwen3_5MoeModel):
    """Top-level multimodal model with the historical text backbone."""

    config: Safin1MoeConfig
    config_class = Safin1MoeConfig

    def __init__(self, config: Safin1MoeConfig):
        Qwen3_5MoePreTrainedModel.__init__(self, config)
        self.visual = Qwen3_5MoeVisionModel._from_config(config.vision_config)
        self.language_model = Safin1MoeTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        if getattr(module, "_hsd_zero_init", False):
            nn.init.zeros_(module.weight)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        anchor_mask: torch.Tensor | None = None,
        hsd_valid_mask: torch.Tensor | None = None,
        text_valid_mask: torch.Tensor | None = None,
        anchor_valid_mask: torch.Tensor | None = None,
        persistent_memory_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple | Qwen3_5MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        return_dict = kwargs.pop("return_dict", None)
        return_dict = return_dict if return_dict is not None else _config_return_dict(self.config)
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, image_grid_thw, return_dict=True, **kwargs)
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        if pixel_values_videos is not None:
            video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True, **kwargs)
            video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        use_cache = kwargs.get("use_cache", None)
        force_layout = bool(use_cache and _anchor_enabled(self.config.text_config))
        if anchor_mask is None and _anchor_enabled(self.config.text_config) and not _cache_has_previous_state(past_key_values):
            layout = _insert_anchor_inputs_with_layout(
                config=self.config.text_config,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                mm_token_type_ids=mm_token_type_ids,
                force_layout=force_layout,
            )
            input_ids = layout["input_ids"]
            inputs_embeds = layout["inputs_embeds"]
            attention_mask = layout["attention_mask"]
            position_ids = layout["position_ids"]
            mm_token_type_ids = layout["mm_token_type_ids"]
            anchor_mask = layout["anchor_mask"]
            hsd_valid_mask = layout["hsd_valid_mask"]
            text_valid_mask = layout["text_valid_mask"]
            anchor_valid_mask = layout["anchor_valid_mask"]
            persistent_memory_mask = layout["persistent_memory_mask"]
            expected_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
            if cache_position is not None and int(cache_position.numel()) != int(expected_len):
                cache_position = None

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            anchor_mask=anchor_mask,
            hsd_valid_mask=hsd_valid_mask,
            text_valid_mask=text_valid_mask,
            anchor_valid_mask=anchor_valid_mask,
            persistent_memory_mask=persistent_memory_mask,
            return_dict=True,
            **kwargs,
        )
        output = Qwen3_5MoeModelOutputWithPast(**outputs, rope_deltas=self.rope_deltas)
        if not return_dict:
            return _as_tuple_output(output)
        return output


class Safin1MoeForConditionalGeneration(Qwen3_5MoeForConditionalGeneration):
    """Causal generation wrapper preserving the official Qwen3.5-MoE API."""

    config: Safin1MoeConfig
    config_class = Safin1MoeConfig

    def __init__(self, config: Safin1MoeConfig):
        Qwen3_5MoePreTrainedModel.__init__(self, config)
        self.model = Safin1MoeModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        if getattr(module, "_hsd_zero_init", False):
            nn.init.zeros_(module.weight)
        state = getattr(module, "persistent_memory_state", None)
        if isinstance(state, nn.Parameter):
            nn.init.zeros_(state)

    def save_persistent_memory(self, output_dir):
        return save_persistent_memory_artifact(self, output_dir)

    def load_persistent_memory(self, artifact, *, strict: bool = True):
        return load_persistent_memory_artifact(self, artifact, strict=strict)

    def get_persistent_memory_metrics(
        self, *, cumulative: bool = False, reset: bool = False, cpu: bool = True
    ):
        return collect_persistent_memory_metrics(
            self.model,
            cumulative=cumulative,
            reset=reset,
            cpu=cpu,
        )

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, generation_mode, batch_size, max_cache_length):
        result = super()._prepare_cache_for_generation(
            generation_config,
            model_kwargs,
            generation_mode,
            batch_size,
            max_cache_length,
        )
        if _anchor_enabled(self.config.text_config):
            cache = model_kwargs.get("past_key_values")
            if cache is not None:
                model_kwargs["past_key_values"] = _patch_hsd_cache_get_seq_length(cache)
        return result

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        logits_to_keep=None,
        cache_position=None,
        **kwargs,
    ):
        if _anchor_enabled(self.config.text_config) and past_key_values is not None and not _cache_has_previous_state(past_key_values):
            model_kwargs = dict(kwargs)
            model_kwargs.pop("next_sequence_length", None)
            mm_token_type_ids = model_kwargs.pop("mm_token_type_ids", None)
            model_inputs = {
                "input_ids": input_ids.contiguous() if input_ids is not None and inputs_embeds is None else None,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": _patch_hsd_cache_get_seq_length(past_key_values),
                "use_cache": use_cache,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "mm_token_type_ids": mm_token_type_ids,
            }
            if cache_position is not None:
                model_inputs["cache_position"] = cache_position
            if logits_to_keep is not None:
                model_inputs["logits_to_keep"] = logits_to_keep
            model_inputs.update(model_kwargs)
            return model_inputs

        if not _anchor_enabled(self.config.text_config) or past_key_values is None or not _cache_has_previous_state(past_key_values):
            model_inputs = super().prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                use_cache=use_cache,
                is_first_iteration=is_first_iteration,
                logits_to_keep=logits_to_keep,
                cache_position=cache_position,
                **kwargs,
            )
            return model_inputs

        if input_ids is None:
            raise ValueError("Safin-1 HSD cached generation requires input_ids during decode.")
        batch_size = input_ids.shape[0]
        past_text_lens = _cache_text_lengths(past_key_values, batch_size, input_ids.device)
        past_expanded_lens = _cache_expanded_lengths(past_key_values, batch_size, input_ids.device)
        input_ids = input_ids[:, -1:]
        if input_ids.shape[1] != 1:
            raise NotImplementedError("Safin-1 HSD V1 supports one new decode token per step.")

        anchor_token_num = int(self.config.text_config.anchor_token_num)
        chunk_size = int(self.config.text_config.anchor_chunk_size)
        append_anchor = (past_text_lens + 1).remainder(chunk_size) == 0
        current_len = 1 + (anchor_token_num if bool(append_anchor.any().item()) else 0)
        if current_len > 1:
            anchor_ids = torch.full(
                (batch_size, anchor_token_num),
                int(getattr(self.config.text_config, "anchor_placeholder_token_id", 0)),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            input_ids = torch.cat([input_ids, anchor_ids], dim=1)
        anchor_mask = torch.zeros(batch_size, input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
        anchor_valid_mask = torch.zeros_like(anchor_mask)
        persistent_memory_mask = torch.zeros_like(anchor_mask)
        text_valid_mask = torch.zeros_like(anchor_mask)
        hsd_valid_mask = torch.zeros_like(anchor_mask)
        text_valid_mask[:, 0] = True
        hsd_valid_mask[:, 0] = True
        if current_len > 1:
            anchor_mask[:, 1:] = True
            anchor_valid_mask[:, 1:] = append_anchor.view(batch_size, 1)
            hsd_valid_mask[:, 1:] = anchor_valid_mask[:, 1:]

        current_position_ids = self._build_decode_position_ids(
            batch_size=batch_size,
            past_text_len=past_text_lens,
            current_len=input_ids.shape[1],
            append_anchor=append_anchor,
            device=input_ids.device,
        )
        past_expanded_len = int(past_expanded_lens.max().item()) if past_expanded_lens.numel() else _cache_expanded_length(past_key_values)
        if cache_position is not None:
            cache_position = torch.arange(
                past_expanded_len,
                past_expanded_len + input_ids.shape[1],
                dtype=cache_position.dtype,
                device=input_ids.device,
            )
        decode_attention_mask = attention_mask
        if torch.is_tensor(decode_attention_mask) and bool(decode_attention_mask.all().item()):
            decode_attention_mask = None

        model_inputs = {
            "input_ids": input_ids.contiguous(),
            "inputs_embeds": None,
            "attention_mask": decode_attention_mask,
            "position_ids": current_position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "anchor_mask": anchor_mask,
            "hsd_valid_mask": hsd_valid_mask,
            "text_valid_mask": text_valid_mask,
            "anchor_valid_mask": anchor_valid_mask,
            "persistent_memory_mask": persistent_memory_mask,
            "pixel_values": None,
            "pixel_values_videos": None,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "mm_token_type_ids": None,
        }
        if cache_position is not None:
            model_inputs["cache_position"] = cache_position
        if logits_to_keep is not None:
            model_inputs["logits_to_keep"] = logits_to_keep
        model_inputs.update(kwargs)
        return model_inputs

    def _build_decode_position_ids(
        self,
        batch_size: int,
        past_text_len: int | torch.Tensor,
        current_len: int,
        append_anchor: bool | torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if torch.is_tensor(past_text_len):
            past_text_tensor = past_text_len.to(device=device, dtype=torch.long).reshape(batch_size)
        else:
            past_text_tensor = torch.full((batch_size,), int(past_text_len), dtype=torch.long, device=device)
        if torch.is_tensor(append_anchor):
            append_anchor_tensor = append_anchor.to(device=device, dtype=torch.bool).reshape(batch_size)
        else:
            append_anchor_tensor = torch.full((batch_size,), bool(append_anchor), dtype=torch.bool, device=device)
        text_pos = past_text_tensor.view(batch_size, 1).expand(batch_size, current_len).clone()
        if current_len > 1:
            anchor_position_type = str(getattr(self.config.text_config, "anchor_position_ids_type", "zeros"))
            if anchor_position_type == "chunk_end":
                text_pos[:, 1:] = past_text_tensor.view(batch_size, 1)
            else:
                text_pos[:, 1:] = 0
            text_pos[:, 1:] = torch.where(
                append_anchor_tensor.view(batch_size, 1),
                text_pos[:, 1:],
                torch.zeros_like(text_pos[:, 1:]),
            )
        if self.model.rope_deltas is not None:
            delta = self.model.rope_deltas.to(device=device, dtype=torch.long)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            if current_len > 1 and str(
                getattr(self.config.text_config, "anchor_position_ids_type", "zeros")
            ) == "zeros":
                text_pos[:, :1] = text_pos[:, :1] + delta
            else:
                text_pos = text_pos + delta
        return text_pos.view(1, batch_size, current_len).expand(4, batch_size, current_len)

    def _reorder_cache(self, past_key_values, beam_idx):
        reorder_cache = getattr(super(), "_reorder_cache", None)
        if callable(reorder_cache):
            reordered = reorder_cache(past_key_values, beam_idx)
            if reordered is not None:
                past_key_values = reordered
        elif hasattr(past_key_values, "reorder_cache"):
            past_key_values.reorder_cache(beam_idx)
        return _reorder_hsd_sidecar_cache(past_key_values, beam_idx)


__all__ = [
    "Safin1MoeAnchorMLP",
    "Safin1MoeTextModel",
    "Safin1MoeModel",
    "Safin1MoeForConditionalGeneration",
    "replace_linear_attention_layers",
]
