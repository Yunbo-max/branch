"""HuggingFace-compatible wrapper so FLAME can build MARCH via AutoModelForCausalLM."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import MARCHConfig
from .model import MARCHModel


class MARCHForCausalLM(PreTrainedModel):
    config_class = MARCHConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    def __init__(self, config: MARCHConfig) -> None:
        super().__init__(config)
        self.model = MARCHModel(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed = value

    def forward(
        self,
        input_ids: Tensor | None = None,
        labels: Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if input_ids is None:
            raise ValueError("input_ids are required")
        logits = self.model(input_ids)
        loss = None
        if labels is not None:
            # Shift so token t predicts t+1, standard causal LM objective.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)


def register() -> None:
    """Register with transformers' automatic classes (idempotent)."""
    AutoConfig.register(MARCHConfig.model_type, MARCHConfig, exist_ok=True)
    AutoModelForCausalLM.register(MARCHConfig, MARCHForCausalLM, exist_ok=True)
