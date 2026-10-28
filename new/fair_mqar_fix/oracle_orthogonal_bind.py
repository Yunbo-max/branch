"""Oracle check: can a single-state delta rule solve online MQAR when the key
addresses are PERFECT (fixed, mutually orthogonal)?

Pre-check for BindDelta. If a delta-rule associative memory with orthogonal
addresses retrieves ~100% while the learned-key delta LM gets ~0%, the failure
is address learning/interference (BindDelta's thesis), not single-state
capacity. If it also fails, capacity is the real limit (DebtDelta).
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn.functional as F

from mqar import MQARConfig


def orthogonal_addresses(n_keys: int, d_key: int, seed: int) -> torch.Tensor:
    """B in R^{n_keys x d_key} with orthonormal rows (B B^T = I)."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(d_key, n_keys, generator=g)
    q, _ = torch.linalg.qr(r)
    return q.T.contiguous()


def delta_forward(tokens, B, Wv, beta, key_offset, value_offset,
                  num_pairs, gap):
    """Delta rule with fixed orthogonal addresses. With orthogonal addresses
    the removal term is exactly zero, so the whole write phase collapses to
    S = beta * sum_i b_i w_i^T (exact, not an approximation)."""
    Bt, seq = tokens.shape
    d_key = B.shape[1]
    prefix = seq - (2 * num_pairs + gap + 2)
    pos = torch.arange(num_pairs) * 2 + prefix
    keys = tokens[:, pos]                       # (Bt, num_pairs)
    vals = tokens[:, pos + 1]
    b = B[keys - key_offset]                    # (Bt, num_pairs, d_key)
    w = Wv[vals - value_offset]                 # (Bt, num_pairs, d_value)
    S = beta * torch.einsum("bnd,bnv->bdv", b, w)   # exact sum of outer products
    q = tokens[:, seq - 2]
    b_q = B[q - key_offset]
    content = torch.einsum("bdv,bd->bv", S, b_q)
    return content @ Wv.T                       # (Bt, n_values)


def make_codebook(n_values, d_value):
    W = torch.randn(n_values, d_value)
    return (W / W.norm(dim=1, keepdim=True)).float()


def analytic_acc(B, num_pairs, n_values, key_offset, value_offset, gap, beta,
                 trials=20):
    Wv = make_codebook(n_values, B.shape[1])
    cfg = MQARConfig(seq_len=128, num_pairs=num_pairs, n_keys=B.shape[0],
                     n_values=n_values, gap=gap)
    rng = np.random.default_rng(0)
    correct, total = 0, 0
    for _ in range(trials):
        tokens, answers = cfg.make_batch(64, rng)
        logits = delta_forward(torch.from_numpy(tokens), B, Wv, beta,
                               key_offset, value_offset, num_pairs, gap)
        correct += int((logits.argmax(-1) == torch.from_numpy(answers) - value_offset).sum())
        total += 64
    return correct / total


def trained_acc(B, num_pairs, n_values, key_offset, value_offset, gap, beta,
                steps, seed):
    cfg = MQARConfig(seq_len=128, num_pairs=num_pairs, n_keys=B.shape[0],
                     n_values=n_values, gap=gap)
    Wv = make_codebook(n_values, B.shape[1]).requires_grad_(True)
    opt = torch.optim.Adam([Wv], lr=1e-2)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        tokens, answers = cfg.make_batch(64, rng)
        logits = delta_forward(torch.from_numpy(tokens), B, Wv, beta,
                               key_offset, value_offset, num_pairs, gap)
        loss = F.cross_entropy(logits, torch.from_numpy(answers) - value_offset)
        opt.zero_grad(); loss.backward(); opt.step()
    vrng = np.random.default_rng(999)
    tokens, answers = cfg.make_batch(1024, vrng)
    with torch.no_grad():
        logits = delta_forward(torch.from_numpy(tokens), B, Wv, beta,
                               key_offset, value_offset, num_pairs, gap)
    return (logits.argmax(-1) == torch.from_numpy(answers) - value_offset).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-key", type=int, default=64)
    ap.add_argument("--n-keys", type=int, default=48)
    ap.add_argument("--n-values", type=int, default=48)
    ap.add_argument("--pairs", nargs="+", type=int, default=[2, 4, 8, 16])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--beta", type=float, default=0.5)
    a = ap.parse_args()

    B = orthogonal_addresses(a.n_keys, a.d_key, seed=0)
    key_offset, value_offset = 1, 1 + a.n_keys
    gap = 32
    print(f"orthogonal addresses: n_keys={a.n_keys} d_key={a.d_key} "
          f"(max orthonormal rows={a.d_key})")
    gram = B @ B.T
    print(f"  address Gram off-diagonal max abs = "
          f"{gram.fill_diagonal_(0).abs().max().item():.2e}")
    print(f"{'num_pairs':>9} {'analytic_acc':>13} {'trained_acc':>13}")
    for npairs in a.pairs:
        ana = analytic_acc(B, npairs, a.n_values, key_offset, value_offset,
                           gap, a.beta)
        tr = trained_acc(B, npairs, a.n_values, key_offset, value_offset,
                         gap, a.beta, a.steps, seed=1)
        print(f"{npairs:>9} {ana:>13.4f} {tr:>13.4f}")


if __name__ == "__main__":
    main()
