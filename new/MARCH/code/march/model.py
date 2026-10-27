"""Reference MARCH model: embed -> expand anchors -> N x MARCHLayer -> LM head."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .anchors import expand_with_anchors
from .config import MARCHConfig
from .layer import MARCHLayer


class MARCHModel(nn.Module):
    def __init__(self, config: MARCHConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MARCHLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm_out = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Args: input_ids (B, L). Returns logits at *text* positions, (B, L, V)."""
        cfg = self.config
        if cfg.anchor_token_id is None:
            raise ValueError("MARCHConfig.anchor_token_id must be set (reserve a vocab id)")
        expanded, text_mask, anchor_pos, boundaries = expand_with_anchors(
            input_ids, cfg.anchor_token_id, cfg.anchor_interval
        )
        h = self.embed(expanded)
        for layer in self.layers:
            h = layer(h, text_mask, anchor_pos, boundaries)
        h_text = h[text_mask].view(input_ids.shape[0], input_ids.shape[1], -1)
        return self.lm_head(self.norm_out(h_text))
