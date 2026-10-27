"""MARCH configuration.

Subclasses ``transformers.PretrainedConfig`` so FLAME's
``AutoModelForCausalLM.from_config(config)`` path works unchanged, exactly like the
vendored FLA and H-GDN baseline configs.
"""

from __future__ import annotations

from transformers import PretrainedConfig


class MARCHConfig(PretrainedConfig):
    model_type = "march"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 1536,
        num_hidden_layers: int = 21,
        num_heads: int = 6,
        head_dim: int = 192,
        max_position_embeddings: int = 16384,
        initializer_range: float = 0.006,
        tie_word_embeddings: bool = False,
        use_cache: bool = True,
        # MARCH (paper §4.1)
        routing_dim: int = 64,          # d_r
        anchor_interval: int = 512,     # C, text tokens between anchors
        top_k: int = -1,                # -1 = dense routing; K > 0 = Top-K sparse variant
        # Shared backbone (identical across all five models)
        intermediate_size: int | None = None,
        # Reference implementation conveniences
        anchor_token_id: int | None = None,
        rms_norm_eps: float = 1e-6,
        **kwargs,
    ) -> None:
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if anchor_interval <= 0:
            raise ValueError("anchor_interval must be positive")
        if top_k == 0 or top_k < -1:
            raise ValueError("top_k must be -1 (dense) or a positive integer")

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.routing_dim = routing_dim
        self.anchor_interval = anchor_interval
        self.top_k = top_k
        self.intermediate_size = intermediate_size
        self.anchor_token_id = anchor_token_id
        self.rms_norm_eps = rms_norm_eps
