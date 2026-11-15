"""Causal K-token native MTP rollout; replay on the exact sampled token IDs.

Qwen's released MTP head is reused recursively with its own cache. Only
the small readout adapter is trained. No full backbone call between proposals;
future target logits/state are computed only AFTER a complete block is drafted.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .features import select_source
from .heads import NativeReadoutAdapter, StateJumpHead

CASES = ("c00_native", "c01_gdn", "c02_gqa", "c03_fusion", "c04_state")


@dataclass
class Draft:
    ids: torch.Tensor  # [1,K]
    q_probs: torch.Tensor  # [1,K,V], detached and exact distribution used for sampling
    logits: torch.Tensor  # [1,K,V], can require adapter gradients on replay
    last_hidden: torch.Tensor  # [1,D], adapter-conditioned after consuming K-th token


class PredictiveMTP(nn.Module):
    def __init__(self, *, case: str, hidden: int = 2560,
                 rank: int = 128, state_weight: float = 0.1):
        super().__init__()
        if case not in CASES:
            raise ValueError("invalid case")
        if state_weight < 0:
            raise ValueError("state_weight cannot be negative")
        self.case = case
        self.hidden = hidden
        self.state_weight = state_weight
        self.adapter = NativeReadoutAdapter(hidden=hidden, rank=rank) if case != "c00_native" else None
        self.state_head = StateJumpHead(hidden=hidden) if case == "c04_state" else None

    def feature(self, gdn: torch.Tensor, gqa: torch.Tensor) -> torch.Tensor:
        return select_source(self.case, gdn, gqa).detach()

    @staticmethod
    def _next_native_mtp(backend, native_hidden: torch.Tensor, token: torch.Tensor,
                         past, position: int):
        """Qwen released MTP forward, with K independent causal cache positions."""
        with torch.no_grad():
            next_emb = backend.base.model.embed_tokens(token[:, None])
            out, new_past = backend.mtp(
                native_hidden[:, None, :], next_emb, past_key_values=past,
                use_cache=True, start_position=position,
            )
        if out.shape != (1, 1, backend.hidden):
            raise RuntimeError("released MTP produced unexpected hidden")
        return out[:, 0, :].detach(), new_past

    def project(self, backend, native_hidden: torch.Tensor, feature: torch.Tensor):
        effective = native_hidden if self.adapter is None else self.adapter(native_hidden, feature)
        logits = backend.base.lm_head(effective.to(backend.base.lm_head.weight.dtype))
        return logits.float(), effective

    def _steps(self, backend, boundary: torch.Tensor, feature: torch.Tensor,
               length: int, *, ids: torch.Tensor | None, generator):
        if length < 1:
            raise ValueError("length must be positive")
        if boundary.shape != (1, backend.hidden) or feature.shape != boundary.shape:
            raise ValueError("boundary source is not aligned with the native model")
        if ids is not None and (ids.shape != (1, length) or ids.dtype != torch.long):
            raise ValueError("replay requires [1,K] int64 tokens")
        native = boundary.detach()
        cache = None
        sampled, probs, predictions = [], [], []
        first_effective = boundary
        for j in range(length):
            if j == 0:
                # First token ALWAYS comes from the original frozen target main head.
                # This also makes C00 native identical to all C01..C04 for token 1.
                current = backend.base.lm_head(boundary).float()
            else:
                current, first_effective = self.project(backend, native, feature)
            predictions.append(current)
            q = F.softmax(current.detach().float(), dim=-1)
            probs.append(q)
            if ids is None:
                tok = torch.multinomial(q, num_samples=1, generator=generator).reshape(1)
            else:
                tok = ids[:, j]
            sampled.append(tok)
            native, cache = self._next_native_mtp(backend, native, tok, cache, j)
        # Native hidden after consuming the complete K-token draft. This is
        # causally available before the target verifies the block.
        last_effective = native if self.adapter is None else self.adapter(native, feature)
        return Draft(ids=torch.stack(sampled, 1),
                     q_probs=torch.stack(probs, 1).detach(),
                     logits=torch.stack(predictions, 1),
                     last_hidden=last_effective)

    @torch.no_grad()
    def sample(self, backend, boundary: torch.Tensor, gdn: torch.Tensor,
               gqa: torch.Tensor, *, horizon: int, generator=None):
        return self._steps(backend, boundary, self.feature(gdn, gqa), horizon,
                           ids=None, generator=generator)

    def replay(self, backend, boundary: torch.Tensor, gdn: torch.Tensor,
               gqa: torch.Tensor, draft_ids: torch.Tensor):
        return self._steps(backend, boundary, self.feature(gdn, gqa),
                           draft_ids.shape[1], ids=draft_ids, generator=None)

    def state_loss(self, draft: Draft, start_state: torch.Tensor, end_state: torch.Tensor):
        if self.state_head is None:
            raise RuntimeError("only C04 has a true GDN state supervision head")
        return self.state_head.loss(draft.last_hidden, start_state, end_state)

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)
