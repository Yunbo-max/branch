"""Boundary-conditioned autoregressive GDN draft head with a fixed-size state.

Only the teacher's *accepted-prefix* hidden state is available at a draft boundary.
No teacher feature is fetched for an unverified draft token.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .gdn import GatedDeltaBlock


@dataclass(frozen=True)
class DrafterConfig:
    vocab_size: int = 248320
    teacher_hidden_size: int = 2560
    hidden_size: int = 256
    n_layers: int = 4
    n_heads: int = 4
    head_dim: int = 64
    mlp_ratio: int = 2

    def __post_init__(self):
        if min(self.vocab_size, self.teacher_hidden_size, self.hidden_size,
               self.n_layers, self.n_heads, self.head_dim, self.mlp_ratio) < 1:
            raise ValueError("all architecture dimensions must be positive")
        if self.n_heads * self.head_dim != self.hidden_size:
            raise ValueError("n_heads * head_dim must equal hidden_size")


class PredictiveStateDrafter(nn.Module):
    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.cfg = cfg
        self.condition = nn.Sequential(nn.LayerNorm(cfg.teacher_hidden_size),
                                       nn.Linear(cfg.teacher_hidden_size, cfg.hidden_size))
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(GatedDeltaBlock(cfg.hidden_size, cfg.n_heads,
                                                    cfg.head_dim, cfg.mlp_ratio)
                                    for _ in range(cfg.n_layers))
        self.out_norm = nn.LayerNorm(cfg.hidden_size)
        # Output and input share a vocabulary embedding; do not allocate two huge matrices.
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def step(self, hidden: torch.Tensor, states: list[torch.Tensor | None] | None = None):
        if states is None:
            states = [None] * len(self.blocks)
        if len(states) != len(self.blocks):
            raise ValueError("one state per GDN block is required")
        new_states = []
        for block, state in zip(self.blocks, states):
            hidden, state = block(hidden, state)
            new_states.append(state)
        return self.lm_head(self.out_norm(hidden)), new_states

    def start(self, teacher_boundary: torch.Tensor):
        if teacher_boundary.ndim != 2 or teacher_boundary.shape[-1] != self.cfg.teacher_hidden_size:
            raise ValueError("teacher_boundary must be [batch, teacher_hidden_size]")
        # The teacher may run BF16 while the drafter is FP32; this is not a grad path.
        hidden = self.condition(teacher_boundary.to(self.embedding.weight.dtype))
        return self.step(hidden)

    def advance(self, token_ids: torch.Tensor, states: list[torch.Tensor]):
        if token_ids.ndim != 1 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be int64 [batch]")
        return self.step(self.embedding(token_ids), states)

    def logits_on_draft(self, teacher_boundary: torch.Tensor, draft_ids: torch.Tensor,
                        *, include_bonus: bool = False):
        """At row j predict draft[j] given ONLY draft[:j]; bonus predicts draft[K]."""
        if draft_ids.ndim != 2 or draft_ids.shape[1] < 1 or draft_ids.dtype != torch.long:
            raise ValueError("draft_ids must be nonempty int64 [batch,K]")
        logits, states = self.start(teacher_boundary)
        if logits.shape[0] != draft_ids.shape[0]:
            raise ValueError("batch mismatch")
        rows = [logits]
        for j in range(draft_ids.shape[1]):
            logits, states = self.advance(draft_ids[:, j], states)
            if j < draft_ids.shape[1] - 1 or include_bonus:
                rows.append(logits)
        return torch.stack(rows, dim=1)

    @torch.no_grad()
    def sample(self, teacher_boundary: torch.Tensor, length: int,
               *, generator: torch.Generator | None = None):
        """Exact unwarped q sampling. Return IDs and the actual q used at each step."""
        if length < 1:
            raise ValueError("draft length must be positive")
        logits, states = self.start(teacher_boundary)
        ids, q_rows = [], []
        for j in range(length):
            q = F.softmax(logits.float(), dim=-1)
            token = torch.multinomial(q, 1, generator=generator).squeeze(-1)
            ids.append(token)
            q_rows.append(q)
            if j + 1 < length:
                logits, states = self.advance(token, states)
        return torch.stack(ids, dim=1), torch.stack(q_rows, dim=1)

    def recurrent_state_bytes(self, batch: int = 1, element_bytes: int = 2) -> int:
        """Matrix states only: excludes vocab embedding, activations and target cache."""
        c = self.cfg
        return batch * c.n_layers * c.n_heads * c.head_dim ** 2 * element_bytes

    def export_config(self) -> dict:
        return asdict(self.cfg)
