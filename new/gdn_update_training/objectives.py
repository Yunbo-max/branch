"""Shared full-vocabulary answer CE; consistency is restricted to current queries."""
import math
import torch
from torch.nn import functional as F


def js_logits(a, b):
    a, b = a.float().log_softmax(-1), b.float().log_softmax(-1)
    midpoint = torch.logaddexp(a,b) - math.log(2.)
    return .5*((a.exp()*(a-midpoint)).sum(-1)+(b.exp()*(b-midpoint)).sum(-1)).mean()


def group_objective(logits, gold, pairs, *, js_weight=0., choice_indices=None):
    if not math.isfinite(js_weight) or js_weight < 0:
        raise ValueError('JS weight must be finite and nonnegative')
    if logits.ndim != 2 or gold.shape != (logits.shape[0],):
        raise ValueError('expected [rows,vocab] logits and [rows] gold')
    ce = F.cross_entropy(logits.float(),gold)
    selected = logits if choice_indices is None else logits.gather(1,choice_indices)
    js = logits.new_zeros((),dtype=torch.float32)
    for a,b in pairs:
        if gold[a] != gold[b]:
            raise ValueError('consistency pairs must have the SAME gold, never historical/current pairs')
        js = js + js_logits(selected[a],selected[b])
    js = js/max(1,len(pairs))
    return ce + js_weight*js, {'ce':float(ce.detach()),'js':float(js.detach())}


def write_distillation(key, beta, target_key, target_beta):
    target_key,target_beta=target_key.detach(),target_beta.detach()
    return (1-(F.normalize(key.float(),dim=-1)*F.normalize(target_key.float(),dim=-1)).sum(-1)).mean() + F.mse_loss(beta.float(),target_beta.float())
