"""Historical-state replacement for Qwen3.5-MoE linear attention layers."""

from __future__ import annotations

import os
import json
from math import isfinite
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from transformers.cache_utils import Cache
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet,
    apply_mask_to_padding_states,
    l2norm,
)

from .fused_history_read import fused_decode_top4_history_read, fused_prefill_top4_history_read
from .fused_march_segments import (
    FLA_CHUNK_SIZE,
    chunk_gated_delta_rule_with_chunk_states_forward,
    chunk_gated_delta_rule_with_anchor_states_forward,
)
from .ops import route_softmax_weights


_HSD_TRACE_COLLECTOR: dict[str, torch.Tensor] | None = None
_HSD_FUSED_HISTORY_DEBUG_COUNT = 0


def _persistent_memory_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.endswith("persistent_memory_state")
    }


def _persistent_memory_route_key_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.endswith("persistent_memory_route_key")
    }


def save_persistent_memory_artifact(
    model: nn.Module,
    output_dir: str | os.PathLike[str],
) -> tuple[Path, Path]:
    """Save the compact persistent state and optional dedicated route keys."""

    from safetensors.torch import save_file

    state_parameters = _persistent_memory_parameters(model)
    route_key_parameters = _persistent_memory_route_key_parameters(model)
    parameters = {**state_parameters, **route_key_parameters}
    if not parameters:
        raise ValueError(
            "model has no persistent_memory_state parameters; construct it with "
            "hsd_persistent_memory_enabled=true."
        )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    tensor_path = root / "persistent_memory.safetensors"
    config_path = root / "persistent_memory_config.json"
    tensors = {
        name: parameter.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for name, parameter in parameters.items()
    }
    artifact_format = "hsd_persistent_memory_v2" if route_key_parameters else "hsd_persistent_memory_v1"
    save_file(tensors, str(tensor_path), metadata={"format": artifact_format})
    model_config = getattr(model, "config", None)
    text_config = getattr(model_config, "text_config", model_config)
    layer_names = sorted(name.rsplit(".persistent_memory_state", 1)[0] for name in state_parameters)
    metadata = {
        "format": artifact_format,
        "key_source": (
            "per_layer_dedicated_parameter"
            if route_key_parameters
            else "anchor_hidden_projection"
        ),
        "enabled": bool(getattr(text_config, "hsd_persistent_memory_enabled", True)),
        "layers": layer_names,
        "anchor_chunk_size": getattr(text_config, "anchor_chunk_size", None),
        "hsd_read_topk": getattr(text_config, "hsd_read_topk", None),
        "hsd_null_read_gate": getattr(text_config, "hsd_null_read_gate", None),
        "tensor_count": len(tensors),
        "state_tensor_count": len(state_parameters),
        "route_key_tensor_count": len(route_key_parameters),
        "parameter_count": sum(tensor.numel() for tensor in tensors.values()),
        "dtype": "float32",
        "tensors": {name: list(tensor.shape) for name, tensor in tensors.items()},
    }
    config_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return tensor_path, config_path


def load_persistent_memory_artifact(
    model: nn.Module,
    artifact: str | os.PathLike[str],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a compact state/key artifact into an already memory-enabled HF model."""

    from safetensors.torch import load_file

    artifact_path = Path(artifact)
    tensor_path = artifact_path / "persistent_memory.safetensors" if artifact_path.is_dir() else artifact_path
    if not tensor_path.is_file():
        raise FileNotFoundError(f"persistent memory tensor file not found: {tensor_path}")
    loaded = load_file(str(tensor_path), device="cpu")
    parameters = {
        **_persistent_memory_parameters(model),
        **_persistent_memory_route_key_parameters(model),
    }
    expected, actual = set(parameters), set(loaded)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if strict and (missing or unexpected):
        raise RuntimeError(
            "persistent memory artifact key mismatch: "
            f"missing={missing} unexpected={unexpected}."
        )
    copied = []
    with torch.no_grad():
        for name in sorted(expected & actual):
            source = loaded[name]
            target = parameters[name]
            if tuple(source.shape) != tuple(target.shape):
                raise RuntimeError(
                    f"persistent memory shape mismatch for {name}: "
                    f"artifact={tuple(source.shape)} model={tuple(target.shape)}."
                )
            target.copy_(source.to(device=target.device, dtype=target.dtype))
            copied.append(name)
    return {
        "tensor_path": str(tensor_path),
        "loaded": copied,
        "missing": missing,
        "unexpected": unexpected,
        "parameter_count": sum(parameters[name].numel() for name in copied),
    }


def collect_persistent_memory_metrics(
    model: nn.Module,
    *,
    cumulative: bool = False,
    reset: bool = False,
    cpu: bool = True,
) -> dict[str, dict[str, torch.Tensor | float]]:
    """Collect per-layer persistent-memory routing metrics from an HF model."""

    result = {}
    for name, module in model.named_modules():
        getter = getattr(module, "get_persistent_memory_metrics", None)
        if not callable(getter):
            continue
        metrics = getter(cumulative=cumulative, reset=reset, cpu=cpu)
        if metrics:
            result[name] = metrics
    return result


def set_hsd_trace_collector(collector: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor] | None:
    global _HSD_TRACE_COLLECTOR
    previous = _HSD_TRACE_COLLECTOR
    _HSD_TRACE_COLLECTOR = collector
    return previous


def trace_hsd_internal(layer_idx: int, name: str, tensor: torch.Tensor | None) -> None:
    if _HSD_TRACE_COLLECTOR is None or not torch.is_tensor(tensor):
        return
    selected_layers = _parse_layer_selector(os.environ.get("HSD_TRACE_LAYERS"))
    if selected_layers is not None and (int(layer_idx) + 1) not in selected_layers:
        return
    selected_stages = os.environ.get("HSD_TRACE_STAGES")
    if selected_stages:
        stage_names = {
            stage.strip()
            for stage in selected_stages.split(",")
            if stage.strip()
        }
        if name not in stage_names:
            return
    _HSD_TRACE_COLLECTOR[f"layer_{int(layer_idx) + 1}/{name}"] = tensor.detach().cpu()


def _parse_layer_selector(value: str | list[int] | tuple[int, ...] | set[int] | int | None) -> set[int] | None:
    """Parse Megatron-style 1-based layer selectors such as ``"2-39"``."""

    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"", "*", "all"}:
            return None
        if raw in {"0", "false", "none", "off"}:
            return set()
        layers: set[int] = set()
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                start, end = int(start_s.strip()), int(end_s.strip())
                if start <= 0 or end < start:
                    raise ValueError(f"Invalid HSD layer selector range: {part!r}")
                layers.update(range(start, end + 1))
            else:
                layer = int(part)
                if layer <= 0:
                    raise ValueError(f"Invalid HSD layer number: {part!r}")
                layers.add(layer)
        return layers
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"Invalid HSD layer number: {value}")
        return {int(value)}
    layers = {int(v) for v in value}
    invalid = sorted(layer for layer in layers if layer <= 0)
    if invalid:
        raise ValueError(f"Invalid HSD layer number(s): {invalid}")
    return layers


def _env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return int(default)
    return int(raw)


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_layer_selected(name: str, layer_idx: int, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    selected = _parse_layer_selector(raw)
    if selected is None:
        return True
    return (int(layer_idx) + 1) in selected


def _debug_fused_history_read(message: str) -> None:
    global _HSD_FUSED_HISTORY_DEBUG_COUNT
    if not _env_truthy("HSD_FUSED_HISTORY_DEBUG", default=False):
        return
    limit = _env_int("HSD_FUSED_HISTORY_DEBUG_LIMIT", 200)
    if limit >= 0 and _HSD_FUSED_HISTORY_DEBUG_COUNT >= limit:
        return
    _HSD_FUSED_HISTORY_DEBUG_COUNT += 1
    print(f"[hsd-fused-history] {message}", flush=True)


def _env_optional_float(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None if default is None else float(default)
    return float(raw)


class Safin1MoeGatedDeltaNet(Qwen3_5MoeGatedDeltaNet):
    """Drop-in replacement for ``Qwen3_5MoeGatedDeltaNet``.

    The no-anchor path delegates to the official implementation.  When a
    ``anchor_mask`` is present, text tokens are run through the original GDN
    recurrence, anchor tokens collect/read recurrent states, and optional
    global history read uses anchor route keys.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

        self.anchor_chunk_size = int(
            getattr(config, "anchor_chunk_size", getattr(config, "hist_chunk_size", 0)) or 0
        )
        self.anchor_token_num = int(getattr(config, "anchor_token_num", 0) or 0)
        self.hsd_read_block_size = int(
            getattr(config, "hsd_read_block_size", getattr(config, "hist_read_block_size", 1024)) or 1024
        )
        self.hsd_read_chunk_block_size = int(
            getattr(config, "hsd_read_chunk_block_size", getattr(config, "hist_read_chunk_block_size", 16)) or 16
        )
        self.hsd_use_global_read = bool(getattr(config, "hsd_use_global_read", True)) and bool(
            getattr(config, "hist_use_history_read", True)
        )
        self.hist_disable_cache = bool(getattr(config, "hist_disable_cache", False))
        self.hsd_global_read_layers = _parse_layer_selector(getattr(config, "hsd_global_read_layers", None))
        self.hsd_layer_uses_global_read = self._global_read_enabled_for_layer()
        self.hsd_history_read_q_mode = str(
            getattr(config, "hsd_history_read_q_mode", "shared_local")
        ).strip().lower()
        self.hsd_history_read_q_lora_rank = int(
            getattr(config, "hsd_history_read_q_lora_rank", 32) or 32
        )
        if self.hsd_history_read_q_mode not in {"shared_local", "hidden_residual_lora"}:
            raise ValueError(
                "hsd_history_read_q_mode must be either 'shared_local' or 'hidden_residual_lora'."
            )
        if self.hsd_history_read_q_lora_rank <= 0:
            raise ValueError("hsd_history_read_q_lora_rank must be positive.")

        self.hsd_use_route_scale = bool(getattr(config, "hsd_use_route_scale", True))
        self.hsd_route_logit_scale = float(getattr(config, "hsd_route_logit_scale", 1.0))
        self.hsd_route_head_dim = int(getattr(config, "hsd_route_head_dim", None) or self.head_k_dim)
        if self.hsd_route_head_dim <= 0:
            raise ValueError("hsd_route_head_dim must be positive.")
        self.hsd_route_query_heads = int(getattr(config, "hsd_route_query_heads", None) or self.num_v_heads)
        if self.hsd_route_query_heads not in {1, self.num_v_heads}:
            raise NotImplementedError("hsd_route_query_heads currently supports 1 or num_v_heads.")
        self.hsd_route_key_heads = int(getattr(config, "hsd_route_key_heads", None) or self.num_v_heads)
        if self.hsd_route_key_heads <= 0:
            raise ValueError("hsd_route_key_heads must be positive.")
        if self.num_v_heads % self.hsd_route_key_heads != 0:
            raise ValueError(
                "hsd_route_key_heads must divide num_v_heads so each route key is shared by an equal "
                f"query-head group: got key_heads={self.hsd_route_key_heads}, num_v_heads={self.num_v_heads}."
            )
        self.hsd_route_qk_l2norm = bool(getattr(config, "hsd_route_qk_l2norm", False))
        self.hsd_route_position_encoding = str(getattr(config, "hsd_route_position_encoding", "none")).lower()
        self.hsd_route_position_mode = str(getattr(config, "hsd_route_position_mode", "chunk_end")).lower()
        if self.hsd_route_position_encoding != "none":
            raise NotImplementedError("Transformers HSD V1 supports hsd_route_position_encoding='none' only.")
        if self.hsd_route_position_mode != "chunk_end":
            raise NotImplementedError("Transformers HSD V1 supports hsd_route_position_mode='chunk_end' only.")
        self.hsd_read_topk = max(0, int(getattr(config, "hsd_read_topk", 0) or 0))
        self.hsd_null_read_gate = bool(getattr(config, "hsd_null_read_gate", False))
        self.hsd_null_read_gate_init_bias = float(getattr(config, "hsd_null_read_gate_init_bias", 0.0))
        self.hsd_persistent_memory_enabled = bool(
            getattr(config, "hsd_persistent_memory_enabled", False)
        )
        self.hsd_persistent_memory_route_key_enabled = bool(
            getattr(config, "hsd_persistent_memory_route_key_enabled", False)
        )
        self.hsd_persistent_memory_route_key_mode = str(
            getattr(config, "hsd_persistent_memory_route_key_mode", "legacy")
        ).strip().lower()
        if self.hsd_persistent_memory_route_key_mode not in {"dedicated", "legacy"}:
            raise ValueError(
                "hsd_persistent_memory_route_key_mode must be 'dedicated' or 'legacy'."
            )
        if self.hsd_persistent_memory_route_key_enabled and not self.hsd_persistent_memory_enabled:
            raise ValueError(
                "hsd_persistent_memory_route_key_enabled requires hsd_persistent_memory_enabled."
            )
        if (
            self.hsd_persistent_memory_route_key_mode == "dedicated"
            and not self.hsd_persistent_memory_route_key_enabled
        ):
            raise ValueError(
                "route-key mode 'dedicated' requires hsd_persistent_memory_route_key_enabled."
            )
        if self.hsd_persistent_memory_route_key_enabled and (
            self.hsd_route_key_heads != self.num_v_heads
            or self.hsd_route_head_dim != self.head_k_dim
        ):
            raise ValueError(
                "Dedicated persistent-memory route keys require one full key per value head: "
                f"route_key_heads={self.hsd_route_key_heads}, value_heads={self.num_v_heads}, "
                f"route_head_dim={self.hsd_route_head_dim}, key_head_dim={self.head_k_dim}."
            )
        if not isfinite(self.hsd_null_read_gate_init_bias):
            raise ValueError("hsd_null_read_gate_init_bias must be finite.")
        self.hsd_local_state_mode = str(getattr(config, "hsd_local_state_mode", "full")).lower()
        if self.hsd_local_state_mode not in {"full", "chunk"}:
            raise ValueError("hsd_local_state_mode must be either 'full' or 'chunk'.")
        self.hsd_text_single_pass_delta = bool(getattr(config, "hsd_text_single_pass_delta", False))
        if self.hsd_text_single_pass_delta and self.hsd_local_state_mode != "full":
            raise ValueError("hsd_text_single_pass_delta requires hsd_local_state_mode='full'.")
        self.hsd_history_scale = float(getattr(config, "hsd_history_scale", 1.0))
        if self.hsd_history_scale < 0:
            raise ValueError("hsd_history_scale must be non-negative.")
        self.hsd_history_out_rmsnorm = bool(getattr(config, "hsd_history_out_rmsnorm", False))
        self.hsd_history_out_rmsnorm_eps = float(getattr(config, "hsd_history_out_rmsnorm_eps", 1e-6))
        if self.hsd_history_out_rmsnorm_eps <= 0:
            raise ValueError("hsd_history_out_rmsnorm_eps must be positive.")

        self.hsd_history_gate_schedule_start = _env_optional_float(
            "HSD_HISTORY_GATE_SCHEDULE_START",
            getattr(config, "hsd_history_gate_schedule_start", None),
        )
        self.hsd_history_gate_schedule_end = _env_optional_float(
            "HSD_HISTORY_GATE_SCHEDULE_END",
            getattr(config, "hsd_history_gate_schedule_end", None),
        )
        self.hsd_history_gate_schedule_steps = _env_int(
            "HSD_HISTORY_GATE_SCHEDULE_STEPS",
            int(getattr(config, "hsd_history_gate_schedule_steps", 0) or 0),
        )
        if self.hsd_history_gate_schedule_steps < 0:
            raise ValueError("hsd_history_gate_schedule_steps must be non-negative.")
        for name, value in (
            ("hsd_history_gate_schedule_start", self.hsd_history_gate_schedule_start),
            ("hsd_history_gate_schedule_end", self.hsd_history_gate_schedule_end),
        ):
            if value is None:
                continue
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
        self.hsd_history_gate_schedule_enabled = (
            self.hsd_history_gate_schedule_steps > 0
            and self.hsd_history_gate_schedule_start is not None
            and self.hsd_history_gate_schedule_end is not None
        )
        if self.hsd_history_gate_schedule_steps > 0 and not self.hsd_history_gate_schedule_enabled:
            raise ValueError(
                "hsd_history_gate_schedule_start and hsd_history_gate_schedule_end are required when "
                "hsd_history_gate_schedule_steps > 0."
            )

        anchor_q_dim = self.num_v_heads * self.head_k_dim
        self.anchor_q_proj = nn.Linear(self.hidden_size, anchor_q_dim, bias=False)
        if self.hsd_layer_uses_global_read and self.hsd_history_read_q_mode == "hidden_residual_lora":
            self.history_read_q_down_proj = nn.Linear(
                self.hidden_size, self.hsd_history_read_q_lora_rank, bias=False
            )
            self.history_read_q_up_proj = nn.Linear(
                self.hsd_history_read_q_lora_rank,
                self.num_v_heads * self.head_k_dim,
                bias=False,
            )
            nn.init.normal_(
                self.history_read_q_down_proj.weight,
                mean=0.0,
                std=float(getattr(config, "initializer_range", 0.02)),
            )
            nn.init.zeros_(self.history_read_q_up_proj.weight)
            # Parent multimodal wrappers may call post_init() again after the
            # historical text model has replaced its attention modules.
            self.history_read_q_up_proj._hsd_zero_init = True
        else:
            self.history_read_q_down_proj = None
            self.history_read_q_up_proj = None
        if self.hsd_layer_uses_global_read:
            self.anchor_k_proj = nn.Linear(
                self.hidden_size, self.hsd_route_key_heads * self.hsd_route_head_dim, bias=False)
            self.router_q_proj = nn.Linear(
                self.hidden_size, self.hsd_route_query_heads * self.hsd_route_head_dim, bias=False)
            if self.hsd_null_read_gate:
                self.null_gate_proj = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
                nn.init.zeros_(self.null_gate_proj.weight)
                self.hsd_null_read_gate_bias = nn.Parameter(
                    torch.full((self.num_v_heads,), self.hsd_null_read_gate_init_bias, dtype=torch.float32)
                )
            else:
                self.null_gate_proj = None
                self.register_parameter("hsd_null_read_gate_bias", None)
        else:
            self.anchor_k_proj = None
            self.router_q_proj = None
            self.null_gate_proj = None
            self.register_parameter("hsd_null_read_gate_bias", None)
        if self.hsd_layer_uses_global_read and self.hsd_history_out_rmsnorm:
            self.hsd_history_out_rmsnorm_weight = nn.Parameter(torch.ones(self.head_v_dim))
        else:
            self.register_parameter("hsd_history_out_rmsnorm_weight", None)
        if self._persistent_memory_state_required(config):
            # Keep the trainable memory in FP32 even when the model's forward
            # dtype is BF16.  It is cast only at the final residual boundary.
            self.persistent_memory_state = nn.Parameter(
                torch.zeros(
                    self.num_v_heads,
                    self.head_k_dim,
                    self.head_v_dim,
                    dtype=torch.float32,
                )
            )
            if self.hsd_persistent_memory_route_key_enabled:
                self.persistent_memory_route_key = nn.Parameter(
                    torch.zeros(
                        self.num_v_heads,
                        self.head_k_dim,
                        dtype=torch.float32,
                    )
                )
            else:
                self.register_parameter("persistent_memory_route_key", None)
        else:
            self.register_parameter("persistent_memory_state", None)
            self.register_parameter("persistent_memory_route_key", None)
        self._hsd_persistent_memory_metrics: dict[str, torch.Tensor] = {}
        self._hsd_persistent_memory_metric_sums: dict[str, torch.Tensor] = {}
        self._hsd_persistent_memory_metric_count = 0

    def _apply(self, fn, recurse: bool = True):
        """Apply device moves while keeping the persistent state in FP32."""

        result = super()._apply(fn, recurse=recurse)
        for name in ("persistent_memory_state", "persistent_memory_route_key"):
            parameter = self._parameters.get(name)
            if parameter is not None and parameter.dtype != torch.float32:
                with torch.no_grad():
                    parameter.data = parameter.data.float()
                    if parameter.grad is not None:
                        parameter.grad.data = parameter.grad.data.float()
        return result

    def _replace_persistent_anchor_route_key(
        self,
        anchor_k: torch.Tensor,
        persistent_anchor_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Replace only boundary-zero keys; dynamic anchor keys remain projected."""

        if self.hsd_persistent_memory_route_key_mode != "dedicated":
            return anchor_k
        parameter = self.persistent_memory_route_key
        if parameter is None:
            raise RuntimeError(
                "Dedicated persistent-memory routing requires persistent_memory_route_key."
            )
        if persistent_anchor_mask.shape != anchor_k.shape[:2]:
            raise ValueError(
                "persistent anchor route-key mask shape mismatch: "
                f"mask={tuple(persistent_anchor_mask.shape)}, keys={tuple(anchor_k.shape)}."
            )
        if tuple(parameter.shape) != tuple(anchor_k.shape[2:]):
            raise RuntimeError(
                "persistent_memory_route_key shape mismatch: "
                f"parameter={tuple(parameter.shape)}, keys={tuple(anchor_k.shape[2:])}."
            )
        expanded = parameter.to(device=anchor_k.device).view(1, 1, *parameter.shape)
        expanded = expanded.expand(anchor_k.shape[0], anchor_k.shape[1], *parameter.shape)
        result = torch.where(
            persistent_anchor_mask.view(*persistent_anchor_mask.shape, 1, 1),
            expanded.to(dtype=anchor_k.dtype),
            anchor_k,
        )
        self._trace_internal("persistent_memory_route_key", parameter)
        return result

    def _trace_internal(self, name: str, tensor: torch.Tensor | None) -> None:
        trace_hsd_internal(int(self.layer_idx), name, tensor)

    @property
    def route_q_proj(self):
        """Legacy alias for the first prototype; not emitted as a state-dict key."""

        return self.router_q_proj

    @property
    def route_k_proj(self):
        """Legacy alias for the first prototype; not emitted as a state-dict key."""

        return self.anchor_k_proj

    @staticmethod
    def _route_keys_for_query_heads(anchor_k: torch.Tensor, query_heads: int) -> torch.Tensor:
        key_heads = int(anchor_k.shape[2])
        query_heads = int(query_heads)
        if key_heads == query_heads:
            return anchor_k
        if key_heads == 1:
            return anchor_k.expand(anchor_k.shape[0], anchor_k.shape[1], query_heads, anchor_k.shape[3])
        if query_heads % key_heads == 0:
            return anchor_k.repeat_interleave(query_heads // key_heads, dim=2)
        raise RuntimeError(
            "HSD anchor route key heads must divide query heads: "
            f"got key_heads={key_heads}, query_heads={query_heads}."
        )

    def _route_logits(self, route_q: torch.Tensor, route_keys: torch.Tensor, value_heads: int) -> torch.Tensor:
        route_q_heads = int(route_q.shape[2])
        route_key_heads = int(route_keys.shape[2])
        if route_q_heads not in {1, int(value_heads)}:
            raise RuntimeError(
                "HSD route query heads must be 1 or match value heads: "
                f"got query_heads={route_q_heads}, value_heads={int(value_heads)}."
            )
        if int(value_heads) % route_key_heads != 0:
            raise RuntimeError(
                "HSD anchor route key heads must divide value heads: "
                f"got key_heads={route_key_heads}, value_heads={int(value_heads)}."
            )
        if route_q.shape[-1] != route_keys.shape[-1]:
            raise RuntimeError(
                "HSD route q/k dim mismatch: "
                f"router_q dim={route_q.shape[-1]}, anchor_k dim={route_keys.shape[-1]}."
            )
        if route_q_heads == value_heads and route_key_heads == value_heads:
            return torch.einsum("bthk,bmhk->bthm", route_q, route_keys)
        if route_q_heads == value_heads and route_key_heads == 1:
            return torch.einsum("bthk,bmk->bthm", route_q, route_keys[:, :, 0])
        if route_q_heads == value_heads:
            grouped_keys = self._route_keys_for_query_heads(route_keys, value_heads)
            return torch.einsum("bthk,bmhk->bthm", route_q, grouped_keys)
        if route_key_heads == value_heads:
            return torch.einsum("btk,bmhk->bthm", route_q[:, :, 0], route_keys)
        if route_key_heads == 1:
            shared_logits = torch.einsum("btk,bmk->btm", route_q[:, :, 0], route_keys[:, :, 0])
            return shared_logits.view(route_q.shape[0], route_q.shape[1], 1, route_keys.shape[1]).expand(
                route_q.shape[0], route_q.shape[1], value_heads, route_keys.shape[1])
        grouped_keys = self._route_keys_for_query_heads(route_keys, value_heads)
        return torch.einsum("btk,bmhk->bthm", route_q[:, :, 0], grouped_keys)

    def _project_null_gate_logits(self, text_hidden: torch.Tensor) -> torch.Tensor | None:
        if self.null_gate_proj is None:
            return None
        if self.hsd_null_read_gate_bias is None:
            raise RuntimeError("hsd_null_read_gate_bias is required when HSD null read gate is enabled.")
        logits = self.null_gate_proj(text_hidden)
        if logits.shape[-1] != self.hsd_null_read_gate_bias.numel():
            raise RuntimeError(
                "HSD null read gate output size must match bias size: "
                f"logits={logits.shape[-1]}, bias={self.hsd_null_read_gate_bias.numel()}."
            )
        bias = self.hsd_null_read_gate_bias.float().to(device=logits.device).view(1, 1, -1)
        return logits.float() + bias

    def _global_read_enabled_for_layer(self) -> bool:
        if not self.hsd_use_global_read:
            return False
        if self.hsd_global_read_layers is None:
            return True
        return (int(self.layer_idx) + 1) in self.hsd_global_read_layers

    def _persistent_memory_state_required(self, config) -> bool:
        """Return whether this GDN layer has a causal path to a global read."""

        if not self.hsd_persistent_memory_enabled or not self.hsd_use_global_read:
            return False
        # An absolute dedicated key severs the legacy path through which an
        # early layer's anchor hidden state changed a later reader's key.
        # Consequently, only layers that perform a real global read may own a
        # trainable state in the dedicated-key architecture.  Keep the wider
        # causal-path allocation for state-only/legacy checkpoints.
        if bool(getattr(self, "hsd_persistent_memory_route_key_enabled", False)):
            return self.hsd_layer_uses_global_read
        layer_types = list(getattr(config, "layer_types", ()) or ())
        num_layers = int(getattr(config, "num_hidden_layers", len(layer_types)) or len(layer_types))
        if not layer_types:
            # Custom/minimal configs cannot expose the future layer topology;
            # retaining the state is safer than silently dropping memory.
            return True
        candidates = []
        for idx, layer_type in enumerate(layer_types[:num_layers]):
            if layer_type != "linear_attention":
                continue
            if self.hsd_global_read_layers is not None and (idx + 1) not in self.hsd_global_read_layers:
                continue
            candidates.append(idx)
        return bool(candidates) and int(self.layer_idx) <= max(candidates)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        kwargs.pop("use_cache", None)
        kwargs.pop("cache_position", None)
        anchor_mask = kwargs.pop("anchor_mask", None)
        hsd_valid_mask = kwargs.pop("hsd_valid_mask", None)
        text_valid_mask = kwargs.pop("text_valid_mask", None)
        anchor_valid_mask = kwargs.pop("anchor_valid_mask", None)
        persistent_memory_mask = kwargs.pop("persistent_memory_mask", None)
        has_anchor_layout = anchor_mask is not None
        has_anchor_tokens = bool(anchor_mask.any().item()) if isinstance(anchor_mask, torch.Tensor) else False
        has_hsd_cache = self._cache_seen_text_tokens(cache_params) > 0
        should_run_hsd = (self.anchor_chunk_size > 0 or self.hsd_persistent_memory_enabled) and (
            has_anchor_tokens or has_hsd_cache or (cache_params is not None and has_anchor_layout)
            or _HSD_TRACE_COLLECTOR is not None
        )

        if not should_run_hsd:
            return super().forward(
                hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
            )
        if cache_params is not None and self.hist_disable_cache:
            raise NotImplementedError("Safin-1 HSD cache is disabled by config.")

        if anchor_mask is None:
            anchor_mask = torch.zeros(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        anchor_mask = anchor_mask.to(device=hidden_states.device, dtype=torch.bool)
        return self.forward_hsd(
            hidden_states=hidden_states,
            anchor_mask=anchor_mask,
            cache_params=cache_params,
            attention_mask=attention_mask,
            hsd_valid_mask=hsd_valid_mask,
            text_valid_mask=text_valid_mask,
            anchor_valid_mask=anchor_valid_mask,
            persistent_memory_mask=persistent_memory_mask,
            **kwargs,
        )

    def forward_hsd(
        self,
        hidden_states: torch.Tensor,
        anchor_mask: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        hsd_valid_mask: torch.Tensor | None = None,
        text_valid_mask: torch.Tensor | None = None,
        anchor_valid_mask: torch.Tensor | None = None,
        persistent_memory_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        if anchor_mask.shape != (batch_size, seq_len):
            raise ValueError(f"anchor_mask shape {tuple(anchor_mask.shape)} must match hidden states {(batch_size, seq_len)}.")
        if hsd_valid_mask is None:
            if attention_mask is not None and attention_mask.ndim == 2 and attention_mask.shape == anchor_mask.shape:
                hsd_valid_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
            else:
                hsd_valid_mask = torch.ones_like(anchor_mask, dtype=torch.bool, device=hidden_states.device)
        else:
            hsd_valid_mask = hsd_valid_mask.to(device=hidden_states.device, dtype=torch.bool)
        if text_valid_mask is None:
            text_valid_mask = (~anchor_mask) & hsd_valid_mask
        else:
            text_valid_mask = text_valid_mask.to(device=hidden_states.device, dtype=torch.bool)
        if anchor_valid_mask is None:
            anchor_valid_mask = anchor_mask & hsd_valid_mask
        else:
            anchor_valid_mask = anchor_valid_mask.to(device=hidden_states.device, dtype=torch.bool)
        if persistent_memory_mask is None:
            persistent_memory_mask = torch.zeros_like(anchor_mask)
        else:
            persistent_memory_mask = persistent_memory_mask.to(device=hidden_states.device, dtype=torch.bool)
        if persistent_memory_mask.shape != anchor_mask.shape:
            raise ValueError(
                "persistent_memory_mask shape must match anchor_mask: "
                f"got {tuple(persistent_memory_mask.shape)} vs {tuple(anchor_mask.shape)}."
            )
        if bool((persistent_memory_mask & ~anchor_valid_mask).any().item()):
            raise ValueError("persistent_memory_mask must be a subset of valid anchor tokens.")
        if bool(persistent_memory_mask.any().item()) and not self.hsd_persistent_memory_enabled:
            raise ValueError("persistent_memory_mask was provided while hsd_persistent_memory_enabled is false.")
        uniform_anchor_layout = torch.equal(anchor_mask, anchor_mask[:1].expand_as(anchor_mask))
        hsd_state = self._get_hsd_state(cache_params) if cache_params is not None else None
        cached_per_row_layout = False
        if hsd_state is not None:
            cached_boundaries = getattr(hsd_state, "hsd_anchor_boundaries", None)
            cached_valid = getattr(hsd_state, "hsd_anchor_valid_mask", None)
            cached_per_row_layout = (
                (torch.is_tensor(cached_boundaries) and cached_boundaries.ndim == 2)
                or torch.is_tensor(cached_valid)
                or torch.is_tensor(getattr(hsd_state, "hsd_seen_text_tokens_per_row", None))
            )
        fast_layout = (
            uniform_anchor_layout
            and bool(hsd_valid_mask.all().item())
            and torch.equal(anchor_valid_mask, anchor_mask)
            and torch.equal(text_valid_mask, ~anchor_mask)
            and torch.equal(persistent_memory_mask, persistent_memory_mask[:1].expand_as(persistent_memory_mask))
            and not cached_per_row_layout
        )
        if not fast_layout:
            return self._forward_hsd_padded_fallback(
                hidden_states=hidden_states,
                anchor_mask=anchor_mask,
                cache_params=cache_params,
                hsd_valid_mask=hsd_valid_mask,
                text_valid_mask=text_valid_mask,
                anchor_valid_mask=anchor_valid_mask,
                persistent_memory_mask=persistent_memory_mask,
                **kwargs,
            )
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        if cache_params is not None and self.hsd_local_state_mode == "chunk":
            raise NotImplementedError("Safin-1 HSD cached decode currently requires hsd_local_state_mode='full'.")

        text_mask = ~anchor_mask
        text_len = int(text_mask[0].sum().item())
        if text_len == 0:
            raise ValueError("Safin-1 HSD requires at least one text token.")
        anchor_count = int(anchor_mask[0].sum().item())

        text_hidden = self._select_btd(hidden_states, text_mask)
        anchor_hidden = self._select_btd(hidden_states, anchor_mask)
        persistent_anchor_mask = self._select_btd(persistent_memory_mask, anchor_mask)
        self._trace_internal("hidden_states_text_before_proj", text_hidden)
        self._trace_internal("anchor_hidden_pre", anchor_hidden)
        mixed_qkv_all = self.in_proj_qkv(hidden_states)
        z_all = self.in_proj_z(hidden_states)
        z = z_all.reshape(batch_size, seq_len, -1, self.head_v_dim)
        mixed_qkv_text = self._select_btd(mixed_qkv_all, text_mask)
        z_text = self._select_btd(z_all, text_mask)
        b = self._select_btd(self.in_proj_b(hidden_states), text_mask)
        a = self._select_btd(self.in_proj_a(hidden_states), text_mask)
        qkvzba_text = torch.cat([mixed_qkv_text, z_text, b, a], dim=-1)
        if qkvzba_text.shape[-1] % int(self.num_k_heads) == 0:
            qkvzba_text = qkvzba_text.view(batch_size, text_len, int(self.num_k_heads), -1)
        self._trace_internal("qkvzba_text_after_proj", qkvzba_text)
        self._trace_internal("conv1d_weight", self.conv1d.weight)
        self._trace_internal("gate_text", self._select_btd(z, text_mask))

        use_precomputed_states = self._has_previous_state(cache_params)
        past_text_len = self._cache_seen_text_tokens(cache_params) if use_precomputed_states else 0
        conv_state = self._get_conv_state(cache_params) if use_precomputed_states else None
        recurrent_state = self._get_recurrent_state(cache_params) if use_precomputed_states else None

        mixed_qkv = self._run_text_conv(
            mixed_qkv_text=mixed_qkv_text,
            cache_params=cache_params,
            conv_state=conv_state,
            use_precomputed_states=use_precomputed_states,
            seq_idx=kwargs.get("seq_idx"),
        )
        self._trace_internal("qkvzba_text_after_conv", mixed_qkv)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, text_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, text_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, text_len, -1, self.head_v_dim)
        query = l2norm(query.contiguous(), dim=-1, eps=1e-6)
        key = l2norm(key.contiguous(), dim=-1, eps=1e-6)

        beta = b.contiguous().sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        self._trace_internal("q_text", query)
        self._trace_internal("k_text", key)
        self._trace_internal("v_text", value)
        self._trace_internal("beta_text", beta)
        self._trace_internal("g_text", g)

        anchor_q = self.anchor_q_proj(anchor_hidden).view(
            batch_size, anchor_count, self.num_v_heads, self.head_k_dim
        )
        anchor_q = l2norm(anchor_q.contiguous(), dim=-1, eps=1e-6) if anchor_count else anchor_q
        self._trace_internal("anchor_q", anchor_q)

        anchor_k = None
        if self.hsd_layer_uses_global_read:
            if self.anchor_k_proj is None:
                raise RuntimeError("anchor_k_proj is required when HSD global read is enabled.")
            anchor_k = self.anchor_k_proj(anchor_hidden).view(
                batch_size, anchor_count, self.hsd_route_key_heads, self.hsd_route_head_dim
            )
            anchor_k = self._replace_persistent_anchor_route_key(
                anchor_k,
                persistent_anchor_mask,
            )
            if self.hsd_route_qk_l2norm and anchor_count:
                anchor_k = l2norm(anchor_k.contiguous(), dim=-1, eps=1e-6)
            self._trace_internal("anchor_k", anchor_k)

        boundary_counts = self._anchor_text_counts(anchor_mask) + int(past_text_len)
        if use_precomputed_states:
            text_o, anchor_o, new_anchor_states, recurrent_state = self._run_hsd_decode_step(
                query=query,
                key=key,
                value=value,
                beta=beta,
                g=g,
                anchor_q=anchor_q,
                boundary_counts=boundary_counts,
                boundary_start=int(past_text_len),
                initial_state=recurrent_state,
            )
        else:
            text_o, anchor_o, new_anchor_states, recurrent_state = self._run_hsd_segments(
                query=query,
                key=key,
                value=value,
                beta=beta,
                g=g,
                anchor_q=anchor_q,
                boundary_counts=boundary_counts,
                boundary_start=int(past_text_len),
                initial_state=recurrent_state,
            )
        anchor_o, new_anchor_states = self._replace_persistent_anchor_state(
            anchor_q=anchor_q,
            anchor_o=anchor_o,
            anchor_states=new_anchor_states,
            persistent_anchor_mask=persistent_anchor_mask,
        )
        if self.hsd_text_single_pass_delta and not use_precomputed_states:
            text_o = self._run_delta_single_pass_text(query=query, key=key, value=value, beta=beta, g=g)
        self._trace_internal("text_o_before_history", text_o)
        self._trace_internal("anchor_o", anchor_o)
        self._trace_internal("anchor_states", new_anchor_states)

        anchor_states, anchor_keys, anchor_boundaries, anchor_memory_mask = None, None, None, None
        if self.hsd_layer_uses_global_read:
            anchor_states, anchor_keys, anchor_boundaries, anchor_memory_mask = self._merge_anchor_cache(
                cache_params=cache_params,
                new_anchor_states=new_anchor_states,
                new_anchor_keys=anchor_k,
                new_anchor_boundaries=boundary_counts,
                new_persistent_memory_mask=persistent_anchor_mask,
            )

        if self.hsd_layer_uses_global_read and anchor_states is not None and anchor_states.shape[1] > 0:
            if self.router_q_proj is None or anchor_keys is None or anchor_boundaries is None:
                raise RuntimeError("router_q_proj and anchor_k cache are required when HSD global read is enabled.")
            router_q = self.router_q_proj(text_hidden).view(
                batch_size, text_len, self.hsd_route_query_heads, self.hsd_route_head_dim
            )
            if self.hsd_route_qk_l2norm:
                router_q = l2norm(router_q.contiguous(), dim=-1, eps=1e-6)
            if self.hsd_history_read_q_mode == "hidden_residual_lora":
                if self.history_read_q_down_proj is None or self.history_read_q_up_proj is None:
                    raise RuntimeError("hidden-residual history-read query LoRA projections are missing.")
                history_read_q_delta = self.history_read_q_up_proj(
                    self.history_read_q_down_proj(text_hidden)
                ).view(batch_size, text_len, self.num_v_heads, self.head_k_dim)
                self._trace_internal("history_read_q_delta", history_read_q_delta)
                read_q = l2norm(
                    (query + history_read_q_delta).contiguous(), dim=-1, eps=1e-6
                )
            else:
                read_q = query
            null_logits = None
            null_logits_raw = self._project_null_gate_logits(text_hidden)
            if null_logits_raw is not None:
                null_logits = null_logits_raw.view(batch_size, text_len, self.num_v_heads, 1).contiguous()
            self._trace_internal("router_q", router_q)
            self._trace_internal("read_q", read_q)
            self._trace_internal("null_gate_logits", null_logits)
            history_out = self._history_read(
                read_q=read_q,
                router_q=router_q,
                states=anchor_states,
                anchor_k=anchor_keys,
                boundary_counts=anchor_boundaries,
                text_len=text_len,
                start_index=int(past_text_len),
                null_logits=null_logits,
                persistent_memory_mask=anchor_memory_mask,
            )
            history_scale = self._history_scale_tensor(device=history_out.device)
            self._trace_internal("history_out_text", history_out)
            history_out = self._apply_history_out_rmsnorm(history_out)
            if self.hsd_history_out_rmsnorm_weight is not None:
                self._trace_internal("history_out_rmsnorm_text", history_out)
            self._trace_internal(
                "history_scaled_text",
                (history_out * history_scale.to(dtype=history_out.dtype)).to(text_o.dtype),
            )
            text_o = text_o + (history_out * history_scale.to(dtype=history_out.dtype)).to(text_o.dtype)
        self._trace_internal("delta_o_text", text_o)

        if cache_params is not None:
            self._update_recurrent_state(cache_params, recurrent_state)
            hsd_state = self._get_hsd_state(cache_params)
            if hsd_state is not None:
                hsd_state.hsd_seen_text_tokens = int(past_text_len) + int(text_len)
                if self.hsd_layer_uses_global_read:
                    hsd_state.hsd_anchor_states = anchor_states
                    hsd_state.hsd_anchor_keys = anchor_keys
                    hsd_state.hsd_anchor_boundaries = anchor_boundaries
                    hsd_state.hsd_anchor_persistent_memory_mask = anchor_memory_mask

        core_attn_out = hidden_states.new_zeros(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        core_attn_out[text_mask] = text_o.reshape(-1, self.num_v_heads, self.head_v_dim)
        if anchor_count:
            core_attn_out[anchor_mask] = anchor_o.reshape(-1, self.num_v_heads, self.head_v_dim)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        self._trace_internal("gated_norm_text", self._select_btd(core_attn_out, text_mask))
        out = self.out_proj(core_attn_out)
        self._trace_internal("out_proj_text", self._select_btd(out, text_mask))
        self._trace_internal("attention_out_text", self._select_btd(out, text_mask))
        return out

    def _forward_hsd_padded_fallback(
        self,
        *,
        hidden_states: torch.Tensor,
        anchor_mask: torch.Tensor,
        cache_params: Cache | None,
        hsd_valid_mask: torch.Tensor,
        text_valid_mask: torch.Tensor,
        anchor_valid_mask: torch.Tensor,
        persistent_memory_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        out = hidden_states.new_zeros(batch_size, seq_len, hidden_size)
        row_caches: list[Any | None] = []
        for batch_idx in range(batch_size):
            valid = hsd_valid_mask[batch_idx]
            if not bool(valid.any().item()):
                raise ValueError("Safin-1 HSD requires at least one valid token per batch row.")
            row_hidden = hidden_states[batch_idx : batch_idx + 1, valid]
            row_anchor_mask = anchor_valid_mask[batch_idx, valid].view(1, -1)
            row_persistent_memory_mask = persistent_memory_mask[batch_idx, valid].view(1, -1)
            row_cache = self._slice_cache_for_row(cache_params, batch_idx, batch_size)
            row_has_anchor = bool(row_anchor_mask.any().item())
            if not row_has_anchor and row_cache is None and _HSD_TRACE_COLLECTOR is None:
                row_out = super().forward(row_hidden, cache_params=None, attention_mask=None)
            else:
                row_out = self.forward_hsd(
                    hidden_states=row_hidden,
                    anchor_mask=row_anchor_mask,
                    cache_params=row_cache,
                    attention_mask=None,
                    persistent_memory_mask=row_persistent_memory_mask,
                    **kwargs,
                )
            out[batch_idx, valid] = row_out.reshape(-1, hidden_size).to(dtype=out.dtype)
            row_caches.append(row_cache)
        if cache_params is not None:
            self._merge_row_caches(cache_params, row_caches, batch_size, hidden_states.device)
        return out

    @staticmethod
    def _slice_batch_tensor(value: torch.Tensor | None, batch_idx: int, batch_size: int) -> torch.Tensor | None:
        if not torch.is_tensor(value) or value.ndim == 0 or int(value.shape[0]) != int(batch_size):
            return value
        return value[batch_idx : batch_idx + 1].contiguous()

    @staticmethod
    def _length_for_row(value, batch_idx: int) -> int:
        if value is None:
            return 0
        if torch.is_tensor(value):
            if value.numel() == 0:
                return 0
            flat = value.reshape(-1)
            if flat.numel() == 1:
                return int(flat.item())
            return int(flat[batch_idx].item())
        return int(value or 0)

    def _slice_cache_for_row(self, cache_params: Cache | None, batch_idx: int, batch_size: int) -> Any | None:
        if cache_params is None:
            return None
        row_cache = SimpleNamespace()

        conv_state = self._get_conv_state(cache_params)
        if torch.is_tensor(conv_state):
            row_cache._hsd_conv_states = [None for _ in range(self.layer_idx + 1)]
            row_cache._hsd_conv_states[self.layer_idx] = self._slice_batch_tensor(conv_state, batch_idx, batch_size)
        recurrent_state = self._get_recurrent_state(cache_params)
        if torch.is_tensor(recurrent_state):
            row_cache._hsd_recurrent_states = [None for _ in range(self.layer_idx + 1)]
            row_cache._hsd_recurrent_states[self.layer_idx] = self._slice_batch_tensor(
                recurrent_state,
                batch_idx,
                batch_size,
            )

        source_state = self._get_hsd_state(cache_params)
        row_state = SimpleNamespace()
        if source_state is not None:
            row_state.hsd_seen_text_tokens = self._length_for_row(
                getattr(source_state, "hsd_seen_text_tokens_per_row", getattr(source_state, "hsd_seen_text_tokens", 0)),
                batch_idx,
            )
            old_states = getattr(source_state, "hsd_anchor_states", None)
            old_keys = getattr(source_state, "hsd_anchor_keys", None)
            old_boundaries = getattr(source_state, "hsd_anchor_boundaries", None)
            old_valid = getattr(source_state, "hsd_anchor_valid_mask", None)
            old_memory_mask = getattr(source_state, "hsd_anchor_persistent_memory_mask", None)
            if torch.is_tensor(old_states):
                if torch.is_tensor(old_valid) and old_valid.ndim == 2 and old_valid.shape[0] == batch_size:
                    valid = old_valid[batch_idx].to(device=old_states.device, dtype=torch.bool)
                else:
                    valid = torch.ones(old_states.shape[1], dtype=torch.bool, device=old_states.device)
                row_state.hsd_anchor_states = old_states[batch_idx : batch_idx + 1, valid].contiguous()
                if torch.is_tensor(old_keys):
                    row_state.hsd_anchor_keys = old_keys[batch_idx : batch_idx + 1, valid].contiguous()
                if torch.is_tensor(old_boundaries):
                    if old_boundaries.ndim == 2 and old_boundaries.shape[0] == batch_size:
                        row_state.hsd_anchor_boundaries = old_boundaries[batch_idx, valid].to(old_states.device)
                    else:
                        row_state.hsd_anchor_boundaries = old_boundaries.to(old_states.device)
                if torch.is_tensor(old_memory_mask):
                    if old_memory_mask.ndim == 2 and old_memory_mask.shape[0] == batch_size:
                        row_state.hsd_anchor_persistent_memory_mask = old_memory_mask[
                            batch_idx : batch_idx + 1, valid
                        ].contiguous()
                    else:
                        row_state.hsd_anchor_persistent_memory_mask = old_memory_mask[..., valid].contiguous()

        row_cache._hsd_layer_states = [SimpleNamespace() for _ in range(self.layer_idx + 1)]
        row_cache._hsd_layer_states[self.layer_idx] = row_state
        return row_cache

    def _merge_row_caches(
        self,
        cache_params: Cache,
        row_caches: list[Any | None],
        batch_size: int,
        device: torch.device,
    ) -> None:
        row_states = [self._get_hsd_state(row_cache) for row_cache in row_caches]
        seen = torch.tensor(
            [int(getattr(row_state, "hsd_seen_text_tokens", 0) or 0) for row_state in row_states],
            dtype=torch.long,
            device=device,
        )

        recurrent_rows = [self._get_recurrent_state(row_cache) for row_cache in row_caches]
        if all(torch.is_tensor(item) for item in recurrent_rows):
            self._update_recurrent_state(cache_params, torch.cat(recurrent_rows, dim=0))
        conv_rows = [self._get_conv_state(row_cache) for row_cache in row_caches]
        if all(torch.is_tensor(item) for item in conv_rows):
            self._update_conv_state(cache_params, torch.cat(conv_rows, dim=0))

        hsd_state = self._get_hsd_state(cache_params)
        if hsd_state is None:
            return
        hsd_state.hsd_seen_text_tokens_per_row = seen
        hsd_state.hsd_seen_text_tokens = int(seen.max().item()) if seen.numel() else 0
        if not self.hsd_layer_uses_global_read:
            return
        state_rows = [getattr(row_state, "hsd_anchor_states", None) for row_state in row_states]
        key_rows = [getattr(row_state, "hsd_anchor_keys", None) for row_state in row_states]
        boundary_rows = [getattr(row_state, "hsd_anchor_boundaries", None) for row_state in row_states]
        memory_rows = [
            getattr(row_state, "hsd_anchor_persistent_memory_mask", None) for row_state in row_states
        ]
        tensor_state_rows = [item for item in state_rows if torch.is_tensor(item)]
        if not tensor_state_rows:
            hsd_state.hsd_anchor_states = None
            hsd_state.hsd_anchor_keys = None
            hsd_state.hsd_anchor_boundaries = None
            hsd_state.hsd_anchor_valid_mask = None
            hsd_state.hsd_anchor_persistent_memory_mask = None
            return

        sample_states = tensor_state_rows[0]
        max_anchor = max(int(item.shape[1]) if torch.is_tensor(item) else 0 for item in state_rows)
        states = sample_states.new_zeros(batch_size, max_anchor, *sample_states.shape[2:])
        valid_mask = torch.zeros(batch_size, max_anchor, dtype=torch.bool, device=states.device)
        boundaries = torch.full((batch_size, max_anchor), -1, dtype=torch.long, device=states.device)
        memory_mask = torch.zeros(batch_size, max_anchor, dtype=torch.bool, device=states.device)
        keys = None
        tensor_key_rows = [item for item in key_rows if torch.is_tensor(item)]
        if tensor_key_rows:
            sample_keys = tensor_key_rows[0]
            keys = sample_keys.new_zeros(batch_size, max_anchor, *sample_keys.shape[2:])

        for batch_idx, row_state in enumerate(row_states):
            row_states_tensor = getattr(row_state, "hsd_anchor_states", None)
            if not torch.is_tensor(row_states_tensor):
                continue
            count = int(row_states_tensor.shape[1])
            if count == 0:
                continue
            states[batch_idx : batch_idx + 1, :count] = row_states_tensor.to(states.dtype)
            valid_mask[batch_idx, :count] = True
            row_keys_tensor = getattr(row_state, "hsd_anchor_keys", None)
            if keys is not None and torch.is_tensor(row_keys_tensor):
                keys[batch_idx : batch_idx + 1, :count] = row_keys_tensor.to(keys.dtype)
            row_boundaries_tensor = getattr(row_state, "hsd_anchor_boundaries", None)
            if torch.is_tensor(row_boundaries_tensor):
                boundaries[batch_idx, :count] = row_boundaries_tensor.to(device=boundaries.device, dtype=torch.long)
            row_memory_tensor = memory_rows[batch_idx]
            if torch.is_tensor(row_memory_tensor):
                memory_mask[batch_idx, :count] = row_memory_tensor.reshape(-1)[:count].to(
                    device=memory_mask.device, dtype=torch.bool
                )

        hsd_state.hsd_anchor_states = states
        hsd_state.hsd_anchor_keys = keys
        hsd_state.hsd_anchor_boundaries = boundaries
        hsd_state.hsd_anchor_valid_mask = valid_mask
        hsd_state.hsd_anchor_persistent_memory_mask = memory_mask

    @staticmethod
    def _select_btd(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size = tensor.shape[0]
        count = int(mask[0].sum().item())
        return tensor[mask].view(batch_size, count, *tensor.shape[2:])

    @staticmethod
    def _anchor_text_counts(anchor_mask: torch.Tensor) -> torch.Tensor:
        text_seen = (~anchor_mask[0]).long().cumsum(dim=0)
        return text_seen[anchor_mask[0]]

    def _replace_persistent_anchor_state(
        self,
        *,
        anchor_q: torch.Tensor,
        anchor_o: torch.Tensor,
        anchor_states: torch.Tensor,
        persistent_anchor_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(persistent_anchor_mask.any().item()):
            return anchor_o, anchor_states
        if self.persistent_memory_state is None:
            # Layers after the last global read intentionally have no trainable
            # state: their anchor output cannot reach a text token.
            return anchor_o, anchor_states
        if persistent_anchor_mask.shape != anchor_states.shape[:2]:
            raise ValueError(
                "persistent anchor mask shape mismatch: "
                f"got {tuple(persistent_anchor_mask.shape)}, expected {tuple(anchor_states.shape[:2])}."
            )
        counts = persistent_anchor_mask.long().sum(dim=1)
        if not bool((counts == 1).all().item()):
            raise ValueError(
                "HF HSD persistent memory requires exactly one memory anchor per active document row; "
                f"got counts={counts.detach().cpu().tolist()}."
            )

        state = self.persistent_memory_state.to(device=anchor_states.device)
        state_expanded = state.view(1, 1, *state.shape).expand(
            anchor_states.shape[0], anchor_states.shape[1], *state.shape
        )
        mask = persistent_anchor_mask.view(*persistent_anchor_mask.shape, 1, 1, 1)
        anchor_states = torch.where(mask, state_expanded.to(anchor_states.dtype), anchor_states)
        memory_o = (
            torch.einsum(
                "bmhk,bmhkv->bmhv",
                anchor_q.to(state_expanded.dtype),
                state_expanded,
            )
            * (self.head_k_dim**-0.5)
        )
        anchor_o = torch.where(
            persistent_anchor_mask.view(*persistent_anchor_mask.shape, 1, 1),
            memory_o.to(anchor_o.dtype),
            anchor_o,
        )
        self._trace_internal("persistent_memory_state", state)
        self._trace_internal("persistent_memory_anchor_o", memory_o)
        return anchor_o, anchor_states

    def _run_text_conv(
        self,
        mixed_qkv_text: torch.Tensor,
        cache_params: Cache | None,
        conv_state: torch.Tensor | None,
        use_precomputed_states: bool,
        seq_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_len = mixed_qkv_text.shape[1]
        mixed_qkv = mixed_qkv_text.transpose(1, 2)
        if use_precomputed_states and text_len == 1 and conv_state is not None:
            if self.causal_conv1d_update is not None:
                mixed_qkv = self.causal_conv1d_update(
                    mixed_qkv,
                    conv_state,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.activation,
                )
                return mixed_qkv.transpose(1, 2)
            conv_input = torch.cat([conv_state, mixed_qkv], dim=-1)
            self._update_conv_state(cache_params, F.pad(conv_input, (self.conv_kernel_size - conv_input.shape[-1], 0)))
            mixed_qkv = F.silu(self.conv1d(conv_input)[:, :, -text_len:])
            return mixed_qkv.transpose(1, 2)

        conv_input = mixed_qkv
        if use_precomputed_states and conv_state is not None:
            conv_input = torch.cat([conv_state, mixed_qkv], dim=-1)
        if cache_params is not None:
            self._update_conv_state(cache_params, F.pad(conv_input, (self.conv_kernel_size - conv_input.shape[-1], 0)))
        if self.causal_conv1d_fn is not None:
            mixed_qkv = self.causal_conv1d_fn(
                x=conv_input,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=seq_idx,
            )
        else:
            mixed_qkv = F.silu(self.conv1d(conv_input)[:, :, : conv_input.shape[-1]])
        if use_precomputed_states and conv_state is not None:
            mixed_qkv = mixed_qkv[:, :, -text_len:]
        return mixed_qkv.transpose(1, 2)

    def _run_hsd_decode_step(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        anchor_q: torch.Tensor,
        boundary_counts: torch.Tensor,
        boundary_start: int,
        initial_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, text_len, heads, key_dim = query.shape
        value_dim = value.shape[-1]
        anchor_count = anchor_q.shape[1]
        if text_len != 1:
            raise NotImplementedError("Safin-1 HSD cached decode supports one text token per step.")

        self._trace_internal("tail_initial_recurrent_state", initial_state)
        text_o, state = self.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        self._trace_internal("tail_final_recurrent_state", state)

        if anchor_count == 0:
            anchor_o = query.new_zeros(batch_size, 0, heads, value_dim)
            states = query.new_zeros(batch_size, 0, heads, key_dim, value_dim)
            return text_o, anchor_o, states, state

        expected_boundary = int(boundary_start) + 1
        expected = torch.full_like(boundary_counts, expected_boundary)
        if not torch.equal(boundary_counts, expected):
            raise ValueError(
                "Safin-1 HSD cached decode anchor tokens must follow the current text token: "
                f"got={boundary_counts.detach().cpu().tolist()} expected={expected.detach().cpu().tolist()}"
            )

        states = state.unsqueeze(1).expand(batch_size, anchor_count, heads, key_dim, value_dim).contiguous()
        anchor_o = (
            torch.einsum(
                "bmhk,bmhkv->bmhv",
                anchor_q.to(states.dtype),
                states,
            )
            * (self.head_k_dim**-0.5)
        ).to(text_o.dtype)
        return text_o, anchor_o, states, state

    def _run_hsd_segments(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        anchor_q: torch.Tensor,
        boundary_counts: torch.Tensor,
        boundary_start: int = 0,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, text_len, heads, key_dim = query.shape
        value_dim = value.shape[-1]
        fast = self._run_hsd_segments_fastpath(
            query=query,
            key=key,
            value=value,
            beta=beta,
            g=g,
            anchor_q=anchor_q,
            boundary_counts=boundary_counts,
            boundary_start=boundary_start,
            initial_state=initial_state,
        )
        if fast is not None:
            return fast

        text_outputs = []
        anchor_outputs = []
        anchor_states = []
        state = initial_state
        cursor = 0
        zero_state = query.new_zeros(batch_size, heads, key_dim, value_dim)

        for anchor_idx, boundary in enumerate(boundary_counts.tolist()):
            local_boundary = int(boundary) - int(boundary_start)
            if local_boundary < cursor:
                raise ValueError("anchor positions must be ordered after non-decreasing text prefixes.")
            if local_boundary > text_len:
                raise ValueError("anchor positions must not exceed current text length.")
            if local_boundary > cursor:
                if self.hsd_local_state_mode == "chunk":
                    out, _ = self._run_delta_segment(query, key, value, beta, g, cursor, local_boundary, None)
                    _, state = self._run_delta_segment(query, key, value, beta, g, cursor, local_boundary, state)
                else:
                    out, state = self._run_delta_segment(query, key, value, beta, g, cursor, local_boundary, state)
                text_outputs.append(out)
                cursor = local_boundary

            state_for_anchor = zero_state if state is None else state
            anchor_states.append(state_for_anchor)
            anchor_outputs.append(
                torch.einsum(
                    "bhk,bhkv->bhv",
                    anchor_q[:, anchor_idx].to(state_for_anchor.dtype),
                    state_for_anchor,
                )
                * (self.head_k_dim**-0.5)
            )

        if cursor < text_len:
            initial_state = None if self.hsd_local_state_mode == "chunk" else state
            self._trace_internal("tail_initial_recurrent_state", initial_state)
            out, state = self._run_delta_segment(query, key, value, beta, g, cursor, text_len, initial_state)
            self._trace_internal("tail_final_recurrent_state", state)
            text_outputs.append(out)

        text_o = torch.cat(text_outputs, dim=1) if text_outputs else query.new_zeros(
            batch_size, 0, heads, value_dim
        )
        anchor_o = torch.stack(anchor_outputs, dim=1).to(text_o.dtype) if anchor_outputs else query.new_zeros(
            batch_size, 0, heads, value_dim
        )
        states = torch.stack(anchor_states, dim=1) if anchor_states else query.new_zeros(
            batch_size, 0, heads, key_dim, value_dim
        )
        if state is None:
            state = zero_state
        return text_o, anchor_o, states, state

    def _run_hsd_segments_fastpath(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        anchor_q: torch.Tensor,
        boundary_counts: torch.Tensor,
        boundary_start: int = 0,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if self.hsd_local_state_mode != "full":
            return None
        if anchor_q.ndim != 4 or boundary_counts.ndim != 1:
            return None
        batch_size, text_len, heads, key_dim = query.shape
        anchor_count = int(anchor_q.shape[1])
        value_dim = int(value.shape[-1])
        if anchor_count <= 0 or int(boundary_counts.shape[0]) != anchor_count:
            return None
        if anchor_q.shape != (batch_size, anchor_count, heads, key_dim):
            return None
        if torch.is_grad_enabled() and any(
            tensor.requires_grad
            for tensor in (query, key, value, beta, g, anchor_q, boundary_counts, initial_state)
            if torch.is_tensor(tensor)
        ):
            return None

        local_boundaries = boundary_counts.to(device=query.device, dtype=torch.long) - int(boundary_start)
        if local_boundaries.numel() != anchor_count:
            return None
        if bool((local_boundaries < 0).any().item()):
            return None
        if bool((local_boundaries > int(text_len)).any().item()):
            return None
        if anchor_count > 1 and bool((local_boundaries[1:] < local_boundaries[:-1]).any().item()):
            return None
        is_final_boundary = local_boundaries == int(text_len)
        aligned_or_final = is_final_boundary | ((local_boundaries % FLA_CHUNK_SIZE) == 0)
        if not bool(aligned_or_final.all().item()):
            return None

        anchor_chunk_size = int(getattr(self, "anchor_chunk_size", 0) or getattr(self, "hist_chunk_size", 0) or 0)
        if anchor_chunk_size > 0 and anchor_chunk_size % FLA_CHUNK_SIZE == 0:
            expected_boundaries = (
                torch.arange(1, anchor_count + 1, device=query.device, dtype=torch.long) * int(anchor_chunk_size)
            )
            if torch.equal(local_boundaries, expected_boundaries):
                selective = chunk_gated_delta_rule_with_anchor_states_forward(
                    query=query,
                    key=key,
                    value=value,
                    g=g,
                    beta=beta,
                    anchor_count=anchor_count,
                    anchor_chunk_size=anchor_chunk_size,
                    initial_state=initial_state,
                    scale=self.head_k_dim**-0.5,
                    anchor_q=anchor_q,
                )
                if selective is not None:
                    text_o, fused_anchor_o, states, final_state = selective
                    if bool(is_final_boundary.any().item()):
                        final_expanded = final_state.unsqueeze(1).expand(
                            batch_size, anchor_count, heads, key_dim, value_dim
                        )
                        states = torch.where(
                            is_final_boundary.view(1, anchor_count, 1, 1, 1),
                            final_expanded,
                            states,
                        )
                    if fused_anchor_o is not None:
                        anchor_o = fused_anchor_o.to(text_o.dtype)
                    else:
                        anchor_outputs = []
                        scale = self.head_k_dim**-0.5
                        for anchor_idx in range(anchor_count):
                            state_for_anchor = states[:, anchor_idx]
                            anchor_outputs.append(
                                torch.einsum(
                                    "bhk,bhkv->bhv",
                                    anchor_q[:, anchor_idx].to(state_for_anchor.dtype),
                                    state_for_anchor,
                                )
                                * scale
                            )
                        anchor_o = torch.stack(anchor_outputs, dim=1).to(text_o.dtype)
                    self._trace_internal("tail_final_recurrent_state", final_state)
                    return text_o, anchor_o, states, final_state

        fast = chunk_gated_delta_rule_with_chunk_states_forward(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            scale=self.head_k_dim**-0.5,
        )
        if fast is None:
            return None
        text_o, chunk_states, final_state = fast
        if final_state is None:
            return None
        if chunk_states.ndim != 5 or int(chunk_states.shape[1]) <= 0:
            return None

        chunk_indices = torch.div(local_boundaries, FLA_CHUNK_SIZE, rounding_mode="floor")
        max_chunk_index = int(chunk_states.shape[1]) - 1
        if bool((chunk_indices[~is_final_boundary] > max_chunk_index).any().item()):
            return None
        gather_indices = chunk_indices.clamp(min=0, max=max_chunk_index)
        states = chunk_states.index_select(1, gather_indices).to(final_state.dtype)
        if bool(is_final_boundary.any().item()):
            final_expanded = final_state.unsqueeze(1).expand(batch_size, anchor_count, heads, key_dim, value_dim)
            states = torch.where(is_final_boundary.view(1, anchor_count, 1, 1, 1), final_expanded, states)

        anchor_outputs = []
        scale = self.head_k_dim**-0.5
        for anchor_idx in range(anchor_count):
            state_for_anchor = states[:, anchor_idx]
            anchor_outputs.append(
                torch.einsum(
                    "bhk,bhkv->bhv",
                    anchor_q[:, anchor_idx].to(state_for_anchor.dtype),
                    state_for_anchor,
                )
                * scale
            )
        anchor_o = torch.stack(anchor_outputs, dim=1).to(text_o.dtype)
        self._trace_internal("tail_final_recurrent_state", final_state)
        return text_o, anchor_o, states, final_state

    def _run_delta_segment(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        start: int,
        end: int,
        initial_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if initial_state is not None and end - start == 1:
            return self.recurrent_gated_delta_rule(
                query[:, start:end],
                key[:, start:end],
                value[:, start:end],
                g=g[:, start:end],
                beta=beta[:, start:end],
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=False,
            )
        return self.chunk_gated_delta_rule(
            query[:, start:end],
            key[:, start:end],
            value[:, start:end],
            g=g[:, start:end],
            beta=beta[:, start:end],
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )

    def _run_delta_single_pass_text(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        text_o, _ = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
        )
        return text_o

    def _history_read(
        self,
        read_q: torch.Tensor,
        router_q: torch.Tensor,
        states: torch.Tensor,
        anchor_k: torch.Tensor,
        boundary_counts: torch.Tensor,
        text_len: int,
        start_index: int = 0,
        null_logits: torch.Tensor | None = None,
        persistent_memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, _, heads, _ = read_q.shape
        anchor_count = states.shape[1]
        value_dim = states.shape[-1]
        if anchor_count == 0:
            return read_q.new_zeros(batch_size, text_len, heads, value_dim)
        if null_logits is not None and tuple(null_logits.shape) != (batch_size, text_len, heads, 1):
            raise RuntimeError(
                "HSD null read logits shape mismatch: "
                f"got {tuple(null_logits.shape)}, expected {(batch_size, text_len, heads, 1)}."
            )
        if router_q.shape[:2] != (batch_size, text_len):
            raise RuntimeError(
                "HSD router query shape mismatch: "
                f"got {tuple(router_q.shape[:2])}, expected {(batch_size, text_len)}."
            )
        if anchor_k.shape[:2] != (batch_size, anchor_count):
            raise RuntimeError(
                "HSD anchor route key shape mismatch: "
                f"got {tuple(anchor_k.shape[:2])}, expected {(batch_size, anchor_count)}."
            )

        if persistent_memory_mask is not None and bool(persistent_memory_mask.any().item()):
            return self._persistent_history_read(
                read_q=read_q,
                router_q=router_q,
                states=states,
                anchor_k=anchor_k,
                boundary_counts=boundary_counts,
                text_len=text_len,
                start_index=start_index,
                null_logits=null_logits,
                persistent_memory_mask=persistent_memory_mask,
            )

        boundaries = boundary_counts.to(device=read_q.device, dtype=torch.long)
        route_scale = self.hsd_route_head_dim**-0.5 if self.hsd_use_route_scale else 1.0
        route_scale = route_scale * self.hsd_route_logit_scale
        topk = self.hsd_read_topk
        block_size = max(1, int(getattr(self, "hsd_read_block_size", 0) or text_len))
        out = read_q.new_empty(batch_size, text_len, heads, value_dim)
        full_head_route = (
            int(router_q.shape[2]) == heads
            and int(anchor_k.shape[2]) == heads
            and int(router_q.shape[-1]) == self.head_k_dim
            and int(anchor_k.shape[-1]) == self.head_k_dim
        )

        for block_start in range(0, text_len, block_size):
            block_end = min(text_len, block_start + block_size)
            block_len = block_end - block_start
            block_router_q = router_q[:, block_start:block_end]

            if full_head_route and null_logits is None and 0 < topk < anchor_count and block_len == 1:
                if topk == 4:
                    fused_out = fused_decode_top4_history_read(
                        read_q=read_q[:, block_start:block_end],
                        router_q=block_router_q,
                        states=states,
                        anchor_k=anchor_k,
                        boundary_counts=boundaries,
                        start_index=int(start_index + block_start),
                        route_scale=float(route_scale),
                        read_scale=float(self.head_k_dim**-0.5),
                    )
                    if fused_out is not None:
                        out[:, block_start:block_end] = fused_out.to(out.dtype)
                        continue
                token_index = torch.arange(
                    start_index + block_start,
                    start_index + block_end,
                    device=read_q.device,
                    dtype=torch.long,
                )
                valid_2d = boundaries.view(1, anchor_count) <= token_index.view(block_len, 1)
                valid = valid_2d.view(1, block_len, 1, anchor_count).expand(
                    batch_size, block_len, heads, anchor_count
                )
                logits = self._route_logits(block_router_q.float(), anchor_k.float(), heads) * route_scale
                _, topk_idx = logits.masked_fill(~valid, float("-inf")).topk(k=topk, dim=-1)
                selected_logits = logits.gather(dim=-1, index=topk_idx)
                selected_valid = valid.gather(dim=-1, index=topk_idx)
                weights = route_softmax_weights(selected_logits, selected_valid, dim=-1)

                batch_idx = torch.arange(batch_size, device=states.device).view(batch_size, 1, 1, 1)
                head_idx = torch.arange(heads, device=states.device).view(1, 1, heads, 1)
                selected_states = states[batch_idx, topk_idx.to(states.device), head_idx]
                block_reads = torch.einsum(
                    "bthk,bthmkv->bthmv",
                    read_q[:, block_start:block_end].to(states.dtype),
                    selected_states,
                )
                block_reads = block_reads * (self.head_k_dim**-0.5)
                block_reads = block_reads.masked_fill(~selected_valid.unsqueeze(-1), 0)
                out[:, block_start:block_end] = torch.einsum(
                    "bthm,bthmv->bthv",
                    weights.to(block_reads.dtype),
                    block_reads,
                ).to(out.dtype)
            else:
                if (
                    0 < topk < anchor_count
                    and null_logits is None
                    and full_head_route
                    and topk == 4
                    and block_len > 1
                    and _env_truthy("HSD_ENABLE_FUSED_HISTORY_READ_PREFILL", default=True)
                    and _env_layer_selected("HSD_FUSED_HISTORY_READ_PREFILL_LAYERS", int(self.layer_idx), default=True)
                ):
                    _debug_fused_history_read(
                        "prefill "
                        f"layer={int(self.layer_idx) + 1} block_start={block_start} block_len={block_len} "
                        f"text_len={text_len} anchor_count={anchor_count} batch={batch_size} heads={heads} "
                        f"key_dim={read_q.shape[-1]} value_dim={value_dim} device={read_q.device}"
                    )
                    fused_out = fused_prefill_top4_history_read(
                        read_q=read_q[:, block_start:block_end],
                        router_q=block_router_q,
                        states=states,
                        anchor_k=anchor_k,
                        boundary_counts=boundaries,
                        start_index=int(start_index + block_start),
                        route_scale=float(route_scale),
                        read_scale=float(self.head_k_dim**-0.5),
                    )
                    if fused_out is not None:
                        if _env_truthy("HSD_FUSED_HISTORY_SYNC", default=False):
                            torch.cuda.synchronize(read_q.device)
                        out[:, block_start:block_end] = fused_out.to(out.dtype)
                        if _env_truthy("HSD_FUSED_HISTORY_SYNC", default=False):
                            torch.cuda.synchronize(read_q.device)
                        continue
                token_index = torch.arange(
                    start_index + block_start,
                    start_index + block_end,
                    device=read_q.device,
                    dtype=torch.long,
                )
                valid_2d = boundaries.view(1, anchor_count) <= token_index.view(block_len, 1)
                valid = valid_2d.view(1, block_len, 1, anchor_count).expand(
                    batch_size, block_len, heads, anchor_count
                )
                logits = self._route_logits(block_router_q.float(), anchor_k.float(), heads) * route_scale
                if 0 < topk < anchor_count:
                    _, topk_idx = logits.masked_fill(~valid, float("-inf")).topk(k=topk, dim=-1)
                    topk_mask = torch.zeros_like(logits, dtype=torch.bool)
                    topk_mask.scatter_(-1, topk_idx, True)
                    valid = valid & topk_mask
                if null_logits is not None:
                    block_null_logits = null_logits[:, block_start:block_end].float()
                    null_valid = torch.ones(
                        batch_size,
                        block_len,
                        heads,
                        1,
                        dtype=torch.bool,
                        device=valid.device,
                    )
                    logits_with_null = torch.cat([logits, block_null_logits], dim=-1)
                    valid_with_null = torch.cat([valid, null_valid], dim=-1)
                    weights = route_softmax_weights(logits_with_null, valid_with_null, dim=-1)[
                        ..., :anchor_count
                    ]
                else:
                    weights = route_softmax_weights(logits, valid, dim=-1)
                block_reads = torch.einsum(
                    "bthk,bmhkv->bthmv",
                    read_q[:, block_start:block_end].to(states.dtype),
                    states,
                )
                block_reads = block_reads * (self.head_k_dim**-0.5)
                block_reads = block_reads.masked_fill(~valid.unsqueeze(-1), 0)
                out[:, block_start:block_end] = torch.einsum(
                    "bthm,bthmv->bthv",
                    weights.to(block_reads.dtype),
                    block_reads,
                ).to(out.dtype)

        return out

    def _persistent_history_read(
        self,
        *,
        read_q: torch.Tensor,
        router_q: torch.Tensor,
        states: torch.Tensor,
        anchor_k: torch.Tensor,
        boundary_counts: torch.Tensor,
        text_len: int,
        start_index: int,
        null_logits: torch.Tensor | None,
        persistent_memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Read memory plus dynamic Top-K using one joint softmax.

        The explicit memory index is concatenated with indices selected only
        from dynamic states, so it can never consume a dynamic Top-K slot.
        This path is differentiable for state/key training and uses sparse
        gathers whenever Top-K is smaller than the dynamic bank.
        """

        batch_size, _, heads, _ = read_q.shape
        anchor_count = int(states.shape[1])
        value_dim = int(states.shape[-1])
        memory_mask = persistent_memory_mask.to(device=states.device, dtype=torch.bool)
        if memory_mask.ndim == 1:
            memory_mask = memory_mask.view(1, -1).expand(batch_size, -1)
        if tuple(memory_mask.shape) != (batch_size, anchor_count):
            raise RuntimeError(
                "HSD persistent memory cache mask shape mismatch: "
                f"got {tuple(memory_mask.shape)}, expected {(batch_size, anchor_count)}."
            )
        memory_counts = memory_mask.long().sum(dim=-1)
        if not bool((memory_counts == 1).all().item()):
            raise RuntimeError(
                "HF HSD persistent history read requires exactly one memory per row; "
                f"got counts={memory_counts.detach().cpu().tolist()}."
            )

        boundaries = boundary_counts.to(device=read_q.device, dtype=torch.long)
        if boundaries.ndim == 1:
            boundaries = boundaries.view(1, -1).expand(batch_size, -1)
        if tuple(boundaries.shape) != (batch_size, anchor_count):
            raise RuntimeError(
                "HSD persistent boundary shape mismatch: "
                f"got {tuple(boundaries.shape)}, expected {(batch_size, anchor_count)}."
            )
        route_scale = self.hsd_route_head_dim**-0.5 if self.hsd_use_route_scale else 1.0
        route_scale = route_scale * self.hsd_route_logit_scale
        block_size = max(1, int(getattr(self, "hsd_read_block_size", 0) or text_len))
        out = read_q.new_empty(batch_size, text_len, heads, value_dim)
        metric_values: dict[str, list[torch.Tensor]] = {
            "memory_logit": [],
            "memory_probability": [],
            "memory_read_rms": [],
        }

        memory_idx_row = memory_mask.long().argmax(dim=-1)
        dynamic_count = max(0, anchor_count - 1)
        topk = int(self.hsd_read_topk)
        for block_start in range(0, text_len, block_size):
            block_end = min(text_len, block_start + block_size)
            block_len = block_end - block_start
            block_router_q = router_q[:, block_start:block_end]
            logits = self._route_logits(block_router_q.float(), anchor_k.float(), heads) * route_scale
            token_index = torch.arange(
                start_index + block_start,
                start_index + block_end,
                device=read_q.device,
                dtype=torch.long,
            )
            valid = boundaries[:, None, None, :] <= token_index[None, :, None, None]
            valid = valid.expand(batch_size, block_len, heads, anchor_count)
            dynamic_valid = valid & ~memory_mask[:, None, None, :]

            if 0 < topk < dynamic_count:
                dynamic_logits = logits.masked_fill(~dynamic_valid, float("-inf"))
                dynamic_idx = dynamic_logits.topk(k=topk, dim=-1).indices
            else:
                dynamic_idx = torch.arange(anchor_count, device=states.device, dtype=torch.long)
                dynamic_idx = dynamic_idx.view(1, 1, 1, anchor_count).expand(
                    batch_size, block_len, heads, anchor_count
                )
            dynamic_selected_valid = dynamic_valid.gather(-1, dynamic_idx.to(dynamic_valid.device))
            dynamic_selected_logits = logits.gather(-1, dynamic_idx.to(logits.device))

            memory_idx = memory_idx_row.view(batch_size, 1, 1, 1).expand(
                batch_size, block_len, heads, 1
            )
            memory_valid = valid.gather(-1, memory_idx.to(valid.device))
            memory_logits = logits.gather(-1, memory_idx.to(logits.device))
            candidate_idx = torch.cat([memory_idx.to(dynamic_idx.device), dynamic_idx], dim=-1)
            candidate_valid = torch.cat([memory_valid, dynamic_selected_valid], dim=-1)
            candidate_logits = torch.cat([memory_logits, dynamic_selected_logits], dim=-1)

            if null_logits is not None:
                block_null_logits = null_logits[:, block_start:block_end].float()
                null_valid = torch.ones_like(block_null_logits, dtype=torch.bool)
                joint_logits = torch.cat([candidate_logits, block_null_logits], dim=-1)
                joint_valid = torch.cat([candidate_valid, null_valid], dim=-1)
                joint_weights = route_softmax_weights(joint_logits, joint_valid, dim=-1)
                candidate_weights = joint_weights[..., : candidate_idx.shape[-1]]
            else:
                joint_weights = route_softmax_weights(candidate_logits, candidate_valid, dim=-1)
                candidate_weights = joint_weights

            batch_idx = torch.arange(batch_size, device=states.device).view(batch_size, 1, 1, 1)
            head_idx = torch.arange(heads, device=states.device).view(1, 1, heads, 1)
            selected_states = states[batch_idx, candidate_idx.to(states.device), head_idx]
            candidate_reads = torch.einsum(
                "bthk,bthmkv->bthmv",
                read_q[:, block_start:block_end].to(states.dtype),
                selected_states,
            )
            candidate_reads = candidate_reads * (self.head_k_dim**-0.5)
            candidate_reads = candidate_reads.masked_fill(~candidate_valid.unsqueeze(-1), 0)
            block_out = torch.einsum(
                "bthm,bthmv->bthv",
                candidate_weights.to(candidate_reads.dtype),
                candidate_reads,
            )
            out[:, block_start:block_end] = block_out.to(out.dtype)

            memory_contribution = candidate_reads[..., 0, :] * candidate_weights[..., 0, None].to(
                candidate_reads.dtype
            )
            metric_values["memory_logit"].append(memory_logits.detach().float().mean())
            metric_values["memory_probability"].append(candidate_weights[..., 0].detach().float().mean())
            metric_values["memory_read_rms"].append(
                memory_contribution.detach().float().square().mean().sqrt()
            )
            self._trace_internal("persistent_memory_logit", memory_logits)
            self._trace_internal("persistent_memory_probability", candidate_weights[..., :1])
            self._trace_internal("persistent_memory_raw_read", candidate_reads[..., 0, :])
            self._trace_internal("persistent_memory_read", memory_contribution)
            self._trace_internal("persistent_memory_joint_probability_sum", joint_weights.sum(dim=-1))

        state_rms = states.detach().float().square().mean(dim=(-3, -2, -1)).sqrt()
        dynamic_rms = state_rms.masked_select(~memory_mask)
        if dynamic_rms.numel():
            dynamic_median = dynamic_rms.median()
        else:
            dynamic_median = state_rms.new_zeros(())
        memory_rms = state_rms.masked_select(memory_mask).mean()
        metrics = {
            name: torch.stack(values).mean() for name, values in metric_values.items() if values
        }
        metrics["state_rms"] = memory_rms
        metrics["dynamic_state_rms_median"] = dynamic_median
        self._record_persistent_memory_metrics(metrics)
        return out

    def _record_persistent_memory_metrics(self, metrics: dict[str, torch.Tensor]) -> None:
        detached = {name: value.detach() for name, value in metrics.items()}
        self._hsd_persistent_memory_metrics = detached
        for name, value in detached.items():
            previous = self._hsd_persistent_memory_metric_sums.get(name)
            self._hsd_persistent_memory_metric_sums[name] = value if previous is None else previous + value
        self._hsd_persistent_memory_metric_count += 1

    def get_persistent_memory_metrics(
        self,
        *,
        cumulative: bool = False,
        reset: bool = False,
        cpu: bool = True,
    ) -> dict[str, torch.Tensor | float]:
        """Return the latest or running-mean memory routing diagnostics."""

        if cumulative and self._hsd_persistent_memory_metric_count:
            count = float(self._hsd_persistent_memory_metric_count)
            source = {name: value / count for name, value in self._hsd_persistent_memory_metric_sums.items()}
        else:
            source = dict(self._hsd_persistent_memory_metrics)
        if cpu:
            result: dict[str, torch.Tensor | float] = {
                name: float(value.detach().float().cpu().item()) for name, value in source.items()
            }
        else:
            result = source
        if reset:
            self._hsd_persistent_memory_metrics = {}
            self._hsd_persistent_memory_metric_sums = {}
            self._hsd_persistent_memory_metric_count = 0
        return result

    def _history_scale_tensor(self, device: torch.device) -> torch.Tensor:
        return self._history_gate_tensor(device=device) * float(self.hsd_history_scale)

    def _apply_history_out_rmsnorm(self, history_out: torch.Tensor) -> torch.Tensor:
        if self.hsd_history_out_rmsnorm_weight is None:
            return history_out
        rms = history_out.float().pow(2).mean(dim=-1, keepdim=True).add(self.hsd_history_out_rmsnorm_eps).rsqrt()
        weight = self.hsd_history_out_rmsnorm_weight.float().view(1, 1, 1, -1)
        return (history_out.float() * rms * weight).to(history_out.dtype)

    def _history_gate_tensor(self, device: torch.device) -> torch.Tensor:
        schedule_gate = self._history_gate_schedule_tensor(device=device)
        if schedule_gate is not None:
            return schedule_gate
        return torch.ones((), device=device, dtype=torch.float32)

    def _history_gate_schedule_tensor(self, device: torch.device) -> torch.Tensor | None:
        if not self.hsd_history_gate_schedule_enabled:
            return None
        if self.hsd_history_gate_schedule_steps <= 1:
            progress = 1.0
        else:
            step = max(0, _env_int("HSD_RUNTIME_STEP", 0))
            progress = min(1.0, float(step) / float(self.hsd_history_gate_schedule_steps - 1))
        start = float(self.hsd_history_gate_schedule_start)
        end = float(self.hsd_history_gate_schedule_end)
        value = start + (end - start) * progress
        return torch.tensor(value, device=device, dtype=torch.float32)

    def _merge_anchor_cache(
        self,
        cache_params: Cache | None,
        new_anchor_states: torch.Tensor,
        new_anchor_keys: torch.Tensor | None,
        new_anchor_boundaries: torch.Tensor,
        new_persistent_memory_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        old_states, old_keys, old_boundaries, old_memory_mask = None, None, None, None
        hsd_state = self._get_hsd_state(cache_params)
        if hsd_state is not None:
            old_states = getattr(hsd_state, "hsd_anchor_states", None)
            old_keys = getattr(hsd_state, "hsd_anchor_keys", None)
            old_boundaries = getattr(hsd_state, "hsd_anchor_boundaries", None)
            old_memory_mask = getattr(hsd_state, "hsd_anchor_persistent_memory_mask", None)

        new_memory_mask = new_persistent_memory_mask.to(
            device=new_anchor_states.device, dtype=torch.bool
        )
        if tuple(new_memory_mask.shape) != tuple(new_anchor_states.shape[:2]):
            raise RuntimeError(
                "new persistent memory cache mask shape mismatch: "
                f"got {tuple(new_memory_mask.shape)}, expected {tuple(new_anchor_states.shape[:2])}."
            )

        if old_states is None or old_states.shape[1] == 0:
            return new_anchor_states, new_anchor_keys, new_anchor_boundaries, new_memory_mask
        if new_anchor_states.shape[1] == 0:
            if old_memory_mask is None:
                old_memory_mask = torch.zeros(
                    old_states.shape[:2], dtype=torch.bool, device=old_states.device
                )
            return old_states, old_keys, old_boundaries, old_memory_mask
        if old_keys is None or new_anchor_keys is None:
            merged_keys = None
        else:
            merged_keys = torch.cat([old_keys, new_anchor_keys.to(old_keys.dtype)], dim=1)
        if old_memory_mask is None:
            old_memory_mask = torch.zeros(old_states.shape[:2], dtype=torch.bool, device=old_states.device)
        else:
            old_memory_mask = old_memory_mask.to(device=old_states.device, dtype=torch.bool)
            if old_memory_mask.ndim == 1:
                old_memory_mask = old_memory_mask.view(1, -1).expand(old_states.shape[0], -1)
        return (
            torch.cat([old_states, new_anchor_states.to(old_states.dtype)], dim=1),
            merged_keys,
            torch.cat([old_boundaries.to(new_anchor_boundaries.device), new_anchor_boundaries], dim=0),
            torch.cat([old_memory_mask, new_memory_mask.to(old_memory_mask.device)], dim=1),
        )

    def _get_layer_cache(self, cache_params: Cache | None) -> Any | None:
        if cache_params is None:
            return None
        layers = getattr(cache_params, "layers", None)
        if layers is not None and len(layers) > self.layer_idx:
            return layers[self.layer_idx]
        return cache_params

    def _get_hsd_state(self, cache_params: Cache | None) -> Any | None:
        if cache_params is None:
            return None
        hsd_layers = getattr(cache_params, "_hsd_layer_states", None)
        if hsd_layers is None:
            cache_layers = getattr(cache_params, "layers", None) or []
            num_layers = len(getattr(cache_params, "conv_states", []) or [])
            num_layers = max(num_layers, len(cache_layers))
            num_layers = max(num_layers, self.layer_idx + 1)
            hsd_layers = [SimpleNamespace() for _ in range(num_layers)]
            cache_params._hsd_layer_states = hsd_layers
        while len(hsd_layers) <= self.layer_idx:
            hsd_layers.append(SimpleNamespace())
        return hsd_layers[self.layer_idx]

    def _has_previous_state(self, cache_params: Cache | None) -> bool:
        if cache_params is None:
            return False
        has_previous_state = getattr(cache_params, "has_previous_state", None)
        if callable(has_previous_state):
            try:
                return bool(has_previous_state(self.layer_idx))
            except TypeError:
                return bool(has_previous_state())
        if isinstance(has_previous_state, bool):
            return has_previous_state
        return (
            self._cache_seen_text_tokens(cache_params) > 0
            or self._get_conv_state(cache_params) is not None
            or self._get_recurrent_state(cache_params) is not None
        )

    def _cache_seen_text_tokens(self, cache_params: Cache | None) -> int:
        hsd_state = self._get_hsd_state(cache_params)
        if hsd_state is None:
            return 0
        return int(getattr(hsd_state, "hsd_seen_text_tokens", 0) or 0)

    def _get_conv_state(self, cache_params: Cache | None) -> torch.Tensor | None:
        layer_cache = self._get_layer_cache(cache_params)
        if layer_cache is not None and hasattr(layer_cache, "conv_states"):
            conv_state = layer_cache.conv_states
            if isinstance(conv_state, (list, tuple)):
                return conv_state[self.layer_idx] if len(conv_state) > self.layer_idx else None
            return conv_state
        conv_states = getattr(cache_params, "conv_states", None) if cache_params is not None else None
        if conv_states is not None and len(conv_states) > self.layer_idx:
            return conv_states[self.layer_idx]
        hsd_conv_states = getattr(cache_params, "_hsd_conv_states", None) if cache_params is not None else None
        if hsd_conv_states is not None and len(hsd_conv_states) > self.layer_idx:
            return hsd_conv_states[self.layer_idx]
        return None

    def _get_recurrent_state(self, cache_params: Cache | None) -> torch.Tensor | None:
        hsd_recurrent_states = getattr(cache_params, "_hsd_recurrent_states", None) if cache_params is not None else None
        if hsd_recurrent_states is not None and len(hsd_recurrent_states) > self.layer_idx:
            recurrent_state = hsd_recurrent_states[self.layer_idx]
            if recurrent_state is not None:
                return recurrent_state
        layer_cache = self._get_layer_cache(cache_params)
        if layer_cache is not None and hasattr(layer_cache, "recurrent_states"):
            recurrent_state = layer_cache.recurrent_states
            if isinstance(recurrent_state, (list, tuple)):
                return recurrent_state[self.layer_idx] if len(recurrent_state) > self.layer_idx else None
            return recurrent_state
        recurrent_states = getattr(cache_params, "recurrent_states", None) if cache_params is not None else None
        if recurrent_states is not None and len(recurrent_states) > self.layer_idx:
            return recurrent_states[self.layer_idx]
        return None

    def _update_conv_state(self, cache_params: Cache, conv_state: torch.Tensor) -> None:
        update_conv_state = getattr(cache_params, "update_conv_state", None)
        if callable(update_conv_state):
            try:
                update_conv_state(conv_state, self.layer_idx)
                return
            except AttributeError:
                pass
        conv_states = getattr(cache_params, "conv_states", None)
        if conv_states is not None and len(conv_states) > self.layer_idx:
            conv_states[self.layer_idx] = conv_state
            return
        hsd_conv_states = getattr(cache_params, "_hsd_conv_states", None)
        if hsd_conv_states is None:
            num_layers = len(getattr(cache_params, "layers", None) or [])
            num_layers = max(num_layers, self.layer_idx + 1)
            hsd_conv_states = [None for _ in range(num_layers)]
            cache_params._hsd_conv_states = hsd_conv_states
        while len(hsd_conv_states) <= self.layer_idx:
            hsd_conv_states.append(None)
        hsd_conv_states[self.layer_idx] = conv_state

    def _update_recurrent_state(self, cache_params: Cache, recurrent_state: torch.Tensor) -> None:
        hsd_recurrent_states = getattr(cache_params, "_hsd_recurrent_states", None)
        if hsd_recurrent_states is None:
            num_layers = len(getattr(cache_params, "layers", None) or [])
            num_layers = max(num_layers, self.layer_idx + 1)
            hsd_recurrent_states = [None for _ in range(num_layers)]
            cache_params._hsd_recurrent_states = hsd_recurrent_states
        while len(hsd_recurrent_states) <= self.layer_idx:
            hsd_recurrent_states.append(None)
        hsd_recurrent_states[self.layer_idx] = recurrent_state

        update_recurrent_state = getattr(cache_params, "update_recurrent_state", None)
        if callable(update_recurrent_state):
            try:
                update_recurrent_state(recurrent_state, self.layer_idx)
                return
            except AttributeError:
                pass
        recurrent_states = getattr(cache_params, "recurrent_states", None)
        if recurrent_states is not None and len(recurrent_states) > self.layer_idx:
            recurrent_states[self.layer_idx] = recurrent_state
            return


class Safin1GatedDeltaNet(Safin1MoeGatedDeltaNet):
    """Dense Qwen3.5 alias for the shared HSD GatedDeltaNet implementation."""


__all__ = [
    "Safin1MoeGatedDeltaNet",
    "Safin1GatedDeltaNet",
    "collect_persistent_memory_metrics",
    "load_persistent_memory_artifact",
    "save_persistent_memory_artifact",
    "set_hsd_trace_collector",
    "trace_hsd_internal",
]
