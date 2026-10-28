"""Real pretrained HF/FLA GDN inference; never substitutes a random-weight proxy."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

LABELS = "ABCD"


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def single_token_choices(tokenizer: Any, question_suffix: str) -> dict[str, int]:
    """Check next-token scoring exactly represents A/B/C/D for this tokenizer.

    Some BPE tokenizers merge across the prefix/answer boundary. Refuse to
    score an invalid one-token label, instead of silently using the wrong ID.
    """
    base = _encode(tokenizer, question_suffix)
    for template in (" {}", "{}", "\n{}"):
        result: dict[str, int] = {}
        for letter in LABELS:
            joined = _encode(tokenizer, question_suffix + template.format(letter))
            if len(joined) != len(base) + 1 or joined[:-1] != base:
                break
            result[letter] = int(joined[-1])
        if len(result) == 4 and len(set(result.values())) == 4:
            return result
    raise ValueError(
        "A/B/C/D are not uniquely representable as one next token for this "
        "tokenizer/question ending. Use an explicitly compatible tokenizer, "
        "or implement full multi-token option likelihood."
    )


@dataclass
class ModelInfo:
    model_id: str
    revision: str | None
    tokenizer_id: str
    train_context_advertised: int | None
    parameters: int
    backend: str
    device: str
    dtype: str


class PretrainedCausalLM:
    def __init__(
        self, model_id: str, *,
        tokenizer_id: str | None = None,
        revision: str | None = None,
        device: str = "cuda",
        dtype: str = "bf16",
        allow_extrapolation: bool = False,
    ) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA required but unavailable")
        if dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("dtype must be bf16, fp16 or fp32")

        # Registers GatedDeltaNetConfig/GatedDeltaNetForCausalLM with HF Auto*.
        # This is NOT a random-weight, hand-written or proxy DeltaNet.
        if "GatedDeltaNet" in model_id or "gated" in model_id.lower():
            try:
                import fla.models.gated_deltanet  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "Pretrained GDN requires flash-linear-attention (fla). "
                    "Install FLA with the matching PyTorch/Triton/CUDA stack."
                ) from exc

        model_kwargs = {"revision": revision, "trust_remote_code": False}
        self.tokenizer_id = tokenizer_id or model_id
        self.tok = AutoTokenizer.from_pretrained(
            self.tokenizer_id, **model_kwargs
        )
        config = AutoConfig.from_pretrained(model_id, **model_kwargs)
        self.context = getattr(config, "max_position_embeddings", None)
        self.allow_extrapolation = allow_extrapolation
        selected = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[dtype]
        # Some FLA checkpoints are FP32 on disk; cast for H100/H200 inference.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=selected, low_cpu_mem_usage=True, **model_kwargs
        )
        self.model.eval().to(device)
        self.device = torch.device(device)
        self.torch = torch
        self.info = ModelInfo(
            model_id=model_id,
            revision=revision,
            tokenizer_id=self.tokenizer_id,
            train_context_advertised=int(self.context) if self.context else None,
            parameters=sum(p.numel() for p in self.model.parameters()),
            backend=type(self.model).__name__,
            device=str(self.device),
            dtype=dtype,
        )

    def score(self, token_ids: tuple[int, ...], choices: dict[str, int]) -> dict:
        torch = self.torch
        if self.context and len(token_ids) > self.context and not self.allow_extrapolation:
            raise ValueError(
                f"{len(token_ids)} tokens exceed model config max_position_embeddings="
                f"{self.context}. Use --allow-extrapolation for explicit length stress tests."
            )
        ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        tic = time.perf_counter()
        with torch.inference_mode():
            out = self.model(input_ids=ids, use_cache=False)
        if not hasattr(out, "logits") or out.logits is None:
            raise RuntimeError("Model did not return logits; check FLA model compatibility.")
        logits = out.logits[0, -1].float()
        picked = torch.stack([logits[choices[x]] for x in LABELS])
        conditional = (picked - torch.logsumexp(picked, 0)).detach().cpu().tolist()
        full_log_den = torch.logsumexp(logits, 0)
        full_log_probs = (picked - full_log_den).detach().cpu().tolist()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak = int(torch.cuda.max_memory_allocated(self.device))
        else:
            peak = None
        ms = 1000.0 * (time.perf_counter() - tic)
        return {
            "prediction": LABELS[int(torch.argmax(picked).item())],
            "log_prob_choice": dict(zip(LABELS, conditional)),
            "log_prob_full": dict(zip(LABELS, full_log_probs)),
            "latency_ms": ms,
            "peak_allocated_bytes": peak,
        }
