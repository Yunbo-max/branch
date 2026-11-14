"""Frozen, pinned Qwen3.5 text-target adapter.

The native MTP side head is deliberately not used for target verification. It is
an explicit B1 baseline in the existing AKR branch, not a replacement for the
true next-token target distribution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import torch

TARGET_ID = "Qwen/Qwen3.5-4B"
TARGET_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


class FrozenQwenTarget:
    def __init__(self, model, tokenizer):
        self.model = model.eval()
        self.tokenizer = tokenizer
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        cfg = getattr(model.config, "text_config", model.config)
        self.hidden_size = int(cfg.hidden_size)
        self.vocab_size = int(cfg.vocab_size)
        self.device = next(model.parameters()).device
        if self.vocab_size != len(tokenizer):
            raise ValueError("target tokenizer/model vocabulary mismatch")

    @classmethod
    def from_local(cls, model_dir: str | Path, *, device: str = "cuda",
                   dtype: str = "bfloat16"):
        """Require pre-downloaded, AKR-style revision manifest: never silently download."""
        if dtype not in {"float32", "bfloat16"}:
            raise ValueError("use float32 or bfloat16")
        path = Path(model_dir).expanduser().resolve()
        manifest_file = path / "model_manifest.json"
        if not manifest_file.is_file():
            raise FileNotFoundError("Missing AKR pinned model_manifest.json; acquire/check exact snapshot first")
        manifest = json.loads(manifest_file.read_text())
        if (manifest.get("repo_id"), manifest.get("revision")) != (TARGET_ID, TARGET_REVISION):
            raise ValueError("model identity/revision differs from pinned teacher")
        for filename in ("config.json", "model.safetensors.index.json"):
            expected = manifest.get("files", {}).get(filename, {}).get("sha256")
            target = path / filename
            if not target.is_file() or not expected or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError(f"missing/mismatched pinned teacher metadata: {filename}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if device.startswith("cuda") and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU cannot run BF16")
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True,
                                                  trust_remote_code=False, use_fast=True)
        cfg = Qwen3_5TextConfig.from_pretrained(str(path), local_files_only=True)
        model, info = Qwen3_5ForCausalLM.from_pretrained(
            str(path), config=cfg, local_files_only=True, trust_remote_code=False,
            dtype=getattr(torch, dtype), device_map={"": device},
            attn_implementation="sdpa", output_loading_info=True,
        )
        unexpected = [x for x in info.get("unexpected_keys", [])
                      if not x.startswith(("mtp.", "model.visual.", "visual."))]
        if any(info.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")) or unexpected:
            raise RuntimeError(f"Frozen target checkpoint not loaded exactly: {info}")
        return cls(model, tokenizer)

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be nonempty")
        if not self.tokenizer.chat_template:
            raise ValueError("pinned Qwen chat template required")
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        )
        if not isinstance(ids, list) or not ids:
            raise ValueError("empty serialized prompt")
        return torch.tensor([ids], device=self.device, dtype=torch.long)

    @torch.no_grad()
    def boundary_and_logits(self, prefix: torch.Tensor,
                            draft: torch.Tensor | None = None):
        """Return boundary hidden [B,H] and p rows [B,K+1,V].

        p row j predicts draft[j] conditioned on prefix+draft[:j]; the last
        row predicts the bonus token after all K draft tokens. This is a
        full-prefix reference call, not a cached production verification path.
        """
        if prefix.ndim != 2 or prefix.shape[1] < 1 or prefix.dtype != torch.long:
            raise ValueError("prefix must be nonempty int64 [B,P]")
        if draft is None:
            draft = prefix[:, :0]
        if draft.ndim != 2 or draft.shape[0] != prefix.shape[0] or draft.dtype != torch.long:
            raise ValueError("draft must be int64 [B,K] aligned with prefix")
        combined = torch.cat((prefix, draft), dim=1)
        output = self.model(input_ids=combined, output_hidden_states=True,
                            use_cache=False, return_dict=True)
        plen, k = prefix.shape[1], draft.shape[1]
        # Causal LM logits at position plen-1+j predict token draft[j].
        teacher_logits = output.logits[:, plen - 1 : plen + k, :].detach()
        boundary = output.hidden_states[-1][:, plen - 1, :].detach()
        if teacher_logits.shape != (prefix.shape[0], k + 1, self.vocab_size):
            raise RuntimeError("teacher logits alignment error")
        return boundary, teacher_logits

    @torch.no_grad()
    def greedy(self, prefix: torch.Tensor, length: int) -> torch.Tensor:
        if length < 1:
            raise ValueError("length must be positive")
        result = self.model.generate(input_ids=prefix, max_new_tokens=length,
                                     do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        response = result[:, prefix.shape[1]:]
        if response.shape[1] < 1:
            raise RuntimeError("empty teacher continuation")
        return response
