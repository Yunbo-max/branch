"""Pinned real HF/FLA inference and an explicitly non-scientific CPU test backend."""
from __future__ import annotations

import hashlib
import inspect
import math
import time
from dataclasses import dataclass
from typing import Any

LABELS = "ABCD"


def single_token_choices(tokenizer: Any, question_suffix: str) -> dict[str, int]:
    base = list(tokenizer.encode(question_suffix, add_special_tokens=False))
    for template in (" {}", "{}", "\n{}"):
        result = {}
        for letter in LABELS:
            joined = list(tokenizer.encode(question_suffix + template.format(letter), add_special_tokens=False))
            if len(joined) != len(base) + 1 or joined[:-1] != base:
                break
            result[letter] = int(joined[-1])
        if len(result) == 4 and len(set(result.values())) == 4:
            return result
    raise ValueError("A/B/C/D must be four distinct single NEXT tokens after the actual query suffix")


def last_logit_kwargs(model: Any) -> dict:
    """Only pass explicitly advertised parameters, never guess through **kwargs."""
    params = inspect.signature(model.forward).parameters
    for key in ("logits_to_keep", "num_logits_to_keep"):
        if key in params:
            return {key: 1}
    return {}


def validate_loading_info(info: dict) -> None:
    """Missing weights must never turn a purported pretrained run into a proxy."""
    problems = {key: info.get(key) for key in
                ('missing_keys','unexpected_keys','mismatched_keys','error_msgs') if info.get(key)}
    if problems:
        raise RuntimeError(f"Checkpoint does not exactly match this model implementation: {problems}")


@dataclass
class ModelInfo:
    model_id: str
    revision: str | None
    tokenizer_id: str
    tokenizer_revision: str | None
    configured_context: int | None
    parameters: int
    backend: str
    device: str
    dtype: str
    last_logits_policy: dict
    model_type: str | None = None
    attention_config: Any = None
    allow_neg_eigval: Any = None


class PretrainedCausalLM:
    def __init__(self, model_id: str, *, tokenizer_id: str | None = None,
                 revision: str | None = None, tokenizer_revision: str | None = None,
                 device: str = "cuda", dtype: str = "bf16",
                 allow_extrapolation: bool = False, tokenizer: Any = None):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("dtype must be bf16, fp16 or fp32")
        # Registration depends on config.model_type, not a directory's name.
        # Importing all installed FLA model registrations supports local paths.
        try:
            import fla.models  # noqa: F401
        except ImportError:
            pass  # Standard HF models remain valid; AutoConfig fails on unknown GDN.
        model_kwargs = {"revision": revision, "trust_remote_code": False}
        self.tokenizer_id = tokenizer_id or model_id
        effective_tok_revision = (tokenizer_revision if tokenizer is not None or
                                  self.tokenizer_id != model_id else tokenizer_revision or revision)
        self.tok = tokenizer or AutoTokenizer.from_pretrained(
            self.tokenizer_id, revision=effective_tok_revision, trust_remote_code=False)
        config = AutoConfig.from_pretrained(model_id, **model_kwargs)
        self.context = getattr(config, "max_position_embeddings", None)
        self.allow_extrapolation = allow_extrapolation
        selected = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        self.model, loading_info = AutoModelForCausalLM.from_pretrained(
            model_id, config=config, torch_dtype=selected, low_cpu_mem_usage=True,
            output_loading_info=True, **model_kwargs)
        validate_loading_info(loading_info)
        self.model.eval().to(device)
        self.device, self.torch = torch.device(device), torch
        self.last_kwargs = last_logit_kwargs(self.model)
        self.info = ModelInfo(
            model_id, revision, self.tokenizer_id, effective_tok_revision,
            int(self.context) if self.context else None,
            sum(p.numel() for p in self.model.parameters()), type(self.model).__name__,
            str(self.device), dtype, self.last_kwargs,
            getattr(config, "model_type", None), getattr(config, "attn", None),
            getattr(config, "allow_neg_eigval", None))

    def score(self, token_ids: tuple[int, ...], choices: dict[str, int]) -> dict:
        torch = self.torch
        if self.context and len(token_ids) > self.context and not self.allow_extrapolation:
            raise ValueError(f"unsupported context: {len(token_ids)} > configured {self.context}; explicitly opt into extrapolation")
        ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        tic = time.perf_counter()
        with torch.inference_mode():
            out = self.model(input_ids=ids, use_cache=False, **self.last_kwargs)
        if getattr(out, "logits", None) is None:
            raise RuntimeError("Model returned no logits; check the FLA/HF compatibility")
        logits = out.logits[0, -1].float()
        picked = torch.stack([logits[choices[x]] for x in LABELS])
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Nonfinite model logits")
        conditional = (picked - torch.logsumexp(picked, 0)).detach().cpu().tolist()
        full = (picked - torch.logsumexp(logits, 0)).detach().cpu().tolist()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak = int(torch.cuda.max_memory_allocated(self.device))
            reserved = int(torch.cuda.max_memory_reserved(self.device))
        else:
            peak = reserved = None
        return {"prediction": LABELS[int(torch.argmax(picked).item())],
                "log_prob_choice": dict(zip(LABELS, conditional)),
                "log_prob_full": dict(zip(LABELS, full)),
                "latency_ms": 1000 * (time.perf_counter() - tic),
                "peak_allocated_bytes": peak, "peak_reserved_bytes": reserved}


class MockTokenizer:
    """Character units ONLY. Never use its lengths or scores as LM evidence."""
    name_or_path = "test-only-character-tokenizer-v1"
    def encode(self, text, add_special_tokens=False):
        return ([1] if add_special_tokens else []) + [ord(c) + 10 for c in text]
    def decode(self, ids, **kwargs):
        return "".join(chr(i - 10) for i in ids if i >= 10)


class MockEvaluator:
    """Deterministic plumbing response; receives no gold answer or model weights."""
    context = None
    def __init__(self, tokenizer=None, **kwargs):
        self.tok = tokenizer or MockTokenizer()
        self.info = ModelInfo("MOCK-NO-MODEL", "mock-v1", self.tok.name_or_path,
                              "mock-v1", None, 0, "MockEvaluator", "cpu", "none", {})
    def score(self, token_ids, choices):
        digest = hashlib.sha256(str(tuple(token_ids)).encode()).digest()
        winner = digest[0] % 4
        probs = [.1] * 4
        probs[winner] = .7
        logs = dict(zip(LABELS, map(math.log, probs)))
        return {"prediction": LABELS[winner], "log_prob_choice": logs,
                "log_prob_full": logs, "latency_ms": 0.,
                "peak_allocated_bytes": None, "peak_reserved_bytes": None}
