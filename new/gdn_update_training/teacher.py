"""Bounded direct-gradient native-key/beta teacher with actual replay acceptance.

This is a practical variant of the proposed state-gradient teacher, not an
implementation of its closed-form oracle. Teacher targets use training labels;
no future-labelled artifact is read by the exported causal model.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from native import Write

@dataclass
class SearchResult:
    target: Write
    accepted: bool
    before: float
    after: float
    protected_before: list[float]
    protected_after: list[float]
    replays: int


def search_write(base: Write, evaluate, *, steps=4, lr=.15, penalty=.05,
                 protect_tolerance=.02, min_gain=1e-4, max_angle=.35) -> SearchResult:
    if steps<1 or any(not math.isfinite(x) for x in (lr,penalty,protect_tolerance,min_gain,max_angle)) or lr<=0 or penalty<0 or protect_tolerance<0 or min_gain<0 or not 0<max_angle<math.pi/2:
        raise ValueError('invalid teacher search budget/coefficients')
    key0=F.normalize(base.key.detach().float(),dim=-1)
    beta0=base.beta.detach().float()
    if key0.ndim!=1 or beta0.numel()!=1 or not bool(torch.isfinite(key0).all()) or not bool(torch.isfinite(beta0).all()) or float(key0.norm())<.5 or not 0<=float(beta0)<=1:
        raise ValueError('invalid baseline write')
    key=key0.clone().requires_grad_()
    logit=torch.logit(beta0.clamp(1e-5,1-1e-5)).clone().requires_grad_()
    opt=torch.optim.Adam([key,logit],lr=lr)
    replays=0
    def checked(w):
        nonlocal replays
        loss,protected=evaluate(w); replays+=1
        if loss.numel()!=1 or protected.ndim!=1 or not bool(torch.isfinite(loss).all()) or not bool(torch.isfinite(protected).all()):
            raise FloatingPointError('nonfinite/malformed teacher replay')
        return loss,protected
    with torch.no_grad():
        start,guard=checked(Write(key0,beta0))
    before=float(start); guard=guard.detach()
    best_key,best_beta=key0.clone(),beta0.clone()
    best_loss=before; best_guard=guard.clone(); accepted=False
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        candidate=Write(F.normalize(key,dim=-1),logit.sigmoid())
        loss,protected=checked(candidate)
        trust=(candidate.key-key0).square().sum()+(candidate.beta-beta0).square()
        # Soft guard guides search; acceptance below independently enforces it.
        objective=loss+penalty*trust+torch.relu(protected-guard-protect_tolerance).sum()
        gradients=torch.autograd.grad(objective,(key,logit),allow_unused=True)
        if any(g is None for g in gradients):
            # An irrelevant direction is allowed; it receives zero update.
            gradients=tuple(torch.zeros_like(p) if g is None else g for g,p in zip(gradients,(key,logit)))
        if any(not bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError('nonfinite teacher gradient')
        key.grad,logit.grad=gradients
        opt.step()
        with torch.no_grad():
            unit=F.normalize(key,dim=-1)
            cosine=(unit*key0).sum().clamp(-1.,1.)
            if float(cosine)<math.cos(max_angle):
                tangent=unit-cosine*key0
                tangent=F.normalize(tangent,dim=-1)
                unit=math.cos(max_angle)*key0+math.sin(max_angle)*tangent
            key.copy_(unit)
            logit.clamp_(-12.,12.)
            trial=Write(key.clone(),logit.sigmoid().clone())
            replay,replay_guard=checked(trial)
            if float(replay)<best_loss-min_gain and bool((replay_guard<=guard+protect_tolerance).all()):
                best_loss=float(replay);best_key=trial.key.clone();best_beta=trial.beta.clone()
                best_guard=replay_guard.clone();accepted=True
    return SearchResult(Write(best_key.detach(),best_beta.detach()),accepted,before,best_loss,
                        guard.cpu().tolist(),best_guard.cpu().tolist(),replays)
