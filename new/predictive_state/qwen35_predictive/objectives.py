"""Causally aligned teacher-to-student full-vocabulary KL, plus C04 state loss."""
from __future__ import annotations
import torch
from torch.nn import functional as F


def forward_kl(student: torch.Tensor, teacher: torch.Tensor, *,
               temperature: float = 1.0) -> torch.Tensor:
    if student.ndim != 3 or teacher.shape != student.shape:
        raise ValueError("full-vocab tensors must be matching [B,K,V]")
    if temperature <= 0 or not float("-inf") < temperature < float("inf"):
        raise ValueError("temperature must be positive and finite")
    log_t = F.log_softmax(teacher.float().detach() / temperature, dim=-1)
    log_s = F.log_softmax(student.float() / temperature, dim=-1)
    value = (log_t.exp() * (log_t - log_s)).sum(-1).mean() * temperature**2
    if not bool(torch.isfinite(value.detach())):
        raise FloatingPointError("nonfinite distillation KL")
    return value


def score_candidate(student_logits: torch.Tensor, teacher_logits: torch.Tensor):
    """Mean per-token predictive KL and total-variation proxy (actual p and q)."""
    if student_logits.ndim != 3 or student_logits.shape != teacher_logits.shape:
        raise ValueError("candidate probabilities not aligned")
    p = torch.softmax(teacher_logits.detach().float(), dim=-1)
    q = torch.softmax(student_logits.detach().float(), dim=-1)
    tv = (p - q).abs().sum(-1).mul(0.5).mean()
    return {"teacher_to_student_kl": float(forward_kl(student_logits, teacher_logits)),
            "mean_tv": float(tv)}
