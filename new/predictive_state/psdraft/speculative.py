"""Mathematically exact single-block speculative sampling, reference not fast path.

Input q[j] is the *actual proposal distribution* conditioned on draft[:j];
input p[j] is target distribution conditioned on the same draft[:j].
Input p[K] is the target's bonus-token distribution. Teacher logits must be
aligned using prompt_len - 1 : prompt_len + K.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class Verification:
    emitted: list[int]
    accepted: int
    rejected_at: int | None  # zero-based


def _normalize(p: torch.Tensor):
    if p.ndim != 1 or not bool(torch.isfinite(p).all()) or bool((p < 0).any()):
        raise ValueError("expected finite nonnegative 1D probabilities")
    total = p.sum()
    if not bool(total > 0):
        raise ValueError("zero probability mass")
    return p / total


@torch.no_grad()
def verify_block(draft_ids: torch.Tensor, q_probs: torch.Tensor,
                 target_probs: torch.Tensor, *, generator: torch.Generator | None = None) -> Verification:
    """One stream, full vocabulary, no top-k or argmax shortcuts; CPU/GPU tensors allowed."""
    if draft_ids.ndim != 1 or draft_ids.dtype != torch.long:
        raise ValueError("draft_ids must be int64 [K]")
    k = draft_ids.numel()
    if k < 1 or q_probs.ndim != 2 or target_probs.ndim != 2 or q_probs.shape[0] != k \
            or target_probs.shape != (k + 1, q_probs.shape[1]):
        raise ValueError("expected draft[K], q[K,V], p[K+1,V]")
    out = []
    for i, token in enumerate(draft_ids.tolist()):
        q = _normalize(q_probs[i].float())
        p = _normalize(target_probs[i].float())
        if not 0 <= token < p.numel() or not bool(q[token] > 0):
            raise ValueError("draft token has zero probability under q")
        accept = torch.minimum(torch.ones_like(p[token]), p[token] / q[token])
        if bool(torch.rand((), device=p.device, generator=generator) < accept):
            out.append(token)
            continue
        correction = (p - q).clamp_min(0)
        correction = _normalize(correction)
        out.append(int(torch.multinomial(correction, 1, generator=generator).item()))
        return Verification(out, i, i)
    bonus = _normalize(target_probs[k].float())
    out.append(int(torch.multinomial(bonus, 1, generator=generator).item()))
    return Verification(out, k, None)
