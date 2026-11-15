"""Strict reuse of AKR's verified native Qwen3.5-4B backbone + released MTP.

Pass --akr-root pointing to a checkout of:
Yunbo-max/AKR, branch research/mtp-opsd-baseline.
No random fallback teacher / lost native MTP tensors / alternate tokenizer.
"""
from __future__ import annotations
import importlib
import json
import sys
from pathlib import Path

import torch
from .features import assert_native_layout, collect_mixer_readouts, native_gdn_matrix

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


class PinnedQwenBackend:
    def __init__(self, native):
        self.native = native
        self.base = native.model.base.eval()
        self.mtp = native.model.mtp.eval()
        self.tokenizer = native.tokenizer
        self.device = native.device
        assert_native_layout(self.base)
        if native.provenance.get("native_mtp_loaded") is not True:
            raise RuntimeError("pretrained native MTP is not loaded")
        if any(p.requires_grad for p in native.model.parameters()):
            raise RuntimeError("target and native MTP must be frozen")
        if len(self.tokenizer) != int(self.base.config.vocab_size):
            raise RuntimeError("vocabulary mismatch; cannot distil or verify")
        self.hidden = int(self.base.config.hidden_size)
        self.vocab = int(self.base.config.vocab_size)

    @classmethod
    def load(cls, *, akr_root: str, model_dir: str, device: str, dtype: str):
        if dtype not in {"float32", "bfloat16"}:
            raise ValueError("Only fp32/bf16 are qualified in AKR")
        root = Path(akr_root).expanduser().resolve()
        if not (root / "mtp_opsd" / "backend.py").is_file():
            raise FileNotFoundError("AKR research/mtp-opsd-baseline checkout required")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        module = importlib.import_module("mtp_opsd.backend")
        if (module.MODEL_ID, module.MODEL_REVISION) != (MODEL_ID, MODEL_REVISION):
            raise RuntimeError("AKR native source does not match pinned Qwen checkpoint")
        config = dict(id=MODEL_ID, revision=MODEL_REVISION,
                      path=str(Path(model_dir).expanduser().resolve()),
                      device=device, dtype=dtype, lora={"enabled": False},
                      gradient_checkpointing=False, attn_implementation="sdpa",
                      max_context_tokens=8192, enable_thinking=False)
        native = module.load_backend(config, trainable=False, device=device)
        return cls(native)

    def encode(self, prompt: str) -> torch.Tensor:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt required")
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        if not isinstance(ids, list) or not ids:
            raise RuntimeError("empty pinned chat template")
        if len(ids) > self.native.max_context_length - 64:
            raise ValueError("prompt exceeds max context minus safety margin")
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def boundary(self, prefix: torch.Tensor, *, collect_state: bool = False):
        if prefix.ndim != 2 or prefix.shape[0] != 1 or prefix.shape[1] < 1:
            raise ValueError("one unpadded context per call")
        with collect_mixer_readouts(self.base) as captured:
            result = self.base.model(input_ids=prefix, use_cache=collect_state)
        if set(captured) != {"gdn", "gqa"}:
            raise RuntimeError("GDN/GQA capture failed; missing branch feature")
        hidden = result.last_hidden_state[:, -1, :].detach()
        start_state = native_gdn_matrix(result.past_key_values) if collect_state else None
        return hidden, captured["gdn"], captured["gqa"], start_state

    @torch.no_grad()
    def teacher(self, prefix: torch.Tensor, draft_ids: torch.Tensor,
                *, collect_end_state: bool = False):
        """Full target p_j for x_j ~ candidate[j], plus p_bonus after K.

        hidden row P-1+j -> distribution over draft[j]; no target-future hidden
        is used for drawing the draft. Actual state end is supervision only.
        """
        if draft_ids.ndim != 2 or draft_ids.shape[0] != 1 or draft_ids.shape[1] < 1:
            raise ValueError("draft IDs must be nonempty [1,K]")
        full = torch.cat((prefix, draft_ids), dim=1)
        if full.shape[1] > self.native.max_context_length:
            raise ValueError("candidate extends beyond fixed context budget")
        output = self.base.model(input_ids=full, use_cache=collect_end_state)
        p = self.base.lm_head(
            output.last_hidden_state[:, prefix.shape[1]-1 : prefix.shape[1]+draft_ids.shape[1], :]
        ).float().detach()
        if p.shape != (1, draft_ids.shape[1]+1, self.vocab):
            raise RuntimeError("target distribution positions are misaligned")
        end_state = native_gdn_matrix(output.past_key_values) if collect_end_state else None
        return p, end_state
