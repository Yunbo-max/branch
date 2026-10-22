"""Dense MARCH / Safin-1 remote-code model."""

from __future__ import annotations

from types import MethodType

import torch
from torch import nn

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5PreTrainedModel,
    Qwen3_5TextModel,
    Qwen3_5VisionModel,
)

# Transformers local remote-code caching copies direct relative imports.
# Declare transitive dependencies here so a fresh cache is self-contained.
from .fused_history_read import fused_prefill_top4_history_read as _remote_history_read  # noqa: F401
from .fused_march_segments import FLA_CHUNK_SIZE as _remote_fla_chunk_size  # noqa: F401
from .configuration_safin1_moe import Safin1MoeConfig as _remote_moe_config  # noqa: F401
from .ops import route_softmax_weights as _remote_route_softmax  # noqa: F401
from .configuration_safin1 import Safin1Config, Safin1TextConfig
from .march_gdn import (
    Safin1GatedDeltaNet,
    collect_persistent_memory_metrics,
    load_persistent_memory_artifact,
    save_persistent_memory_artifact,
)
from .modeling_safin1_moe import (
    Safin1MoeAnchorMLP as Safin1AnchorMLP,
    _as_tuple_output,
    _cache_expanded_length,
    _cache_expanded_lengths,
    _cache_has_previous_state,
    _cache_text_lengths,
    _config_return_dict,
    _create_qwen_causal_mask,
    _decoder_layer_accepts_cache_position,
    _filter_batch_sequence_to_text,
    _filter_anchor_hidden,
    _full_attention_mask_source,
    _historical_hsd_decoder_layer_forward,
    _insert_anchor_inputs_with_layout,
    _new_qwen35_cache,
    _patch_hsd_cache_get_seq_length,
    _reorder_hsd_sidecar_cache,
    _set_root_cache_lengths,
    _anchor_text_indices,
    _anchor_has_tokens,
    _anchor_mlp_enabled,
    _text_positions_from_anchor_mask,
    _update_linear_attn_mask,
)


def _anchor_enabled(config: Safin1TextConfig) -> bool:
    return (
        bool(getattr(config, "hist_replace_linear_attn", True))
        and int(getattr(config, "anchor_token_num", 0) or 0) > 0
        and (
            int(getattr(config, "anchor_chunk_size", 0) or 0) > 0
            or bool(getattr(config, "hsd_persistent_memory_enabled", False))
        )
    )


def replace_linear_attention_layers(language_model: Qwen3_5TextModel, config: Safin1TextConfig) -> None:
    for layer_idx, layer in enumerate(language_model.layers):
        layer_type = config.layer_types[layer_idx]
        layer._historical_hsd_config = config
        layer._historical_hsd_layer_idx = int(layer_idx)
        if layer_type == "linear_attention":
            layer.linear_attn = Safin1GatedDeltaNet(config, layer_idx)
            if _anchor_mlp_enabled(config) and not hasattr(layer, "hsd_anchor_mlp"):
                layer.hsd_anchor_mlp = Safin1AnchorMLP(config)
            if _anchor_enabled(config) and not getattr(layer, "_historical_hsd_linear_patched", False):
                layer.forward = MethodType(_historical_hsd_decoder_layer_forward, layer)
                layer._historical_hsd_linear_patched = True
                layer._historical_hsd_accepts_anchor_mask = True
        elif layer_type == "full_attention" and _anchor_enabled(config):
            if _anchor_mlp_enabled(config) and not hasattr(layer, "hsd_anchor_mlp"):
                layer.hsd_anchor_mlp = Safin1AnchorMLP(config)
            if not getattr(layer, "_historical_hsd_full_attention_patched", False):
                layer.forward = MethodType(_historical_hsd_decoder_layer_forward, layer)
                layer._historical_hsd_full_attention_patched = True
                layer._historical_hsd_accepts_anchor_mask = True


class Safin1TextModel(Qwen3_5TextModel):
    """Dense Qwen3.5 text model with optional HSD anchor-token support."""

    config: Safin1TextConfig
    config_class = Safin1TextConfig

    def __init__(self, config: Safin1TextConfig):
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
    ) -> Qwen3_5ModelOutputWithPast:
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
        kwargs.pop("output_router_logits", None)
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

        hidden_states = self.norm(hidden_states)
        hidden_states = _filter_anchor_hidden(hidden_states, anchor_mask)
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

        outputs = Qwen3_5ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=None,
        )
        if not return_dict:
            return _as_tuple_output(outputs)
        return outputs


class Safin1Model(Qwen3_5Model):
    """Top-level dense multimodal model with the historical text backbone."""

    config: Safin1Config
    config_class = Safin1Config

    def __init__(self, config: Safin1Config):
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.visual = Qwen3_5VisionModel._from_config(config.vision_config)
        self.language_model = Safin1TextModel._from_config(config.text_config)
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
    ) -> tuple | Qwen3_5ModelOutputWithPast:
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
        output = Qwen3_5ModelOutputWithPast(**outputs, rope_deltas=self.rope_deltas)
        if not return_dict:
            return _as_tuple_output(output)
        return output


class Safin1ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Causal generation wrapper preserving the official dense Qwen3.5 API."""

    config: Safin1Config
    config_class = Safin1Config

    def __init__(self, config: Safin1Config):
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.model = Safin1Model(config)
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
    "Safin1AnchorMLP",
    "Safin1TextModel",
    "Safin1Model",
    "Safin1ForConditionalGeneration",
    "replace_linear_attention_layers",
]
