"""Full-vocabulary, causally aligned teacher-to-student objectives."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def forward_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
               temperature: float = 1.0) -> torch.Tensor:
    """Mean over [batch,positions] of KL(teacher || student); teacher detached."""
    if student_logits.ndim != 3 or student_logits.shape != teacher_logits.shape:
        raise ValueError("expected matching [B,K,V] logits")
    if not 0 < temperature < float("inf"):
        raise ValueError("temperature must be finite and positive")
    log_p = F.log_softmax(teacher_logits.detach().float() / temperature, dim=-1)
    log_q = F.log_softmax(student_logits.float() / temperature, dim=-1)
    loss = (log_p.exp() * (log_p - log_q)).sum(dim=-1).mean() * temperature ** 2
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError("non-finite KL")
    return loss


def teacher_forcing_loss(student_logits: torch.Tensor, target_ids: torch.Tensor):
    if student_logits.ndim != 3 or target_ids.shape != student_logits.shape[:2]:
        raise ValueError("student logits and target tokens must align")
    return F.cross_entropy(student_logits.float().reshape(-1, student_logits.size(-1)),
                           target_ids.reshape(-1))
