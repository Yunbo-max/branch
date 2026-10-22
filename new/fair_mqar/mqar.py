"""Synthetic multi-query associative recall (MQAR) for memory diagnostics.

Layout (fixed length, no padding mask needed):

    [filler] * prefix  k1 v1 ... kn vn  [filler] * gap  q  v_q

``q`` is one of the keys and ``v_q`` its value. The supervised token is the final
``v_q``; the model must predict it from the position of ``q``. Varying ``num_pairs`` and
``gap`` separates association capacity from retention interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class MQARConfig:
    seq_len: int = 256
    num_pairs: int = 8
    gap: int = 32
    n_keys: int = 48
    n_values: int = 48
    filler_token: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_pairs < 1:
            raise ValueError("num_pairs must be positive")
        if 2 * self.num_pairs + self.gap + 2 > self.seq_len:
            raise ValueError("num_pairs/gap do not fit in seq_len")
        if self.n_keys < 2 or self.n_values < 2:
            raise ValueError("need at least two keys and values")

    @property
    def vocab_size(self) -> int:
        return 1 + self.n_keys + self.n_values

    @property
    def key_offset(self) -> int:
        return 1

    @property
    def value_offset(self) -> int:
        return 1 + self.n_keys

    def make_batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        prefix = self.seq_len - (2 * self.num_pairs + self.gap + 2)
        tokens = np.full((batch_size, self.seq_len), self.filler_token, dtype=np.int64)
        answers = np.empty(batch_size, dtype=np.int64)
        for i in range(batch_size):
            keys = (
                rng.choice(self.n_keys, self.num_pairs, replace=False) + self.key_offset
            )
            vals = rng.integers(0, self.n_values, self.num_pairs) + self.value_offset
            order = rng.permutation(self.num_pairs)
            keys, vals = keys[order], vals[order]
            q = int(rng.integers(0, self.num_pairs))
            pos = prefix
            for k, v in zip(keys, vals):
                tokens[i, pos] = k
                tokens[i, pos + 1] = v
                pos += 2
            pos += self.gap
            tokens[i, pos] = keys[q]
            tokens[i, pos + 1] = vals[q]
            answers[i] = vals[q]
        return tokens, answers

    def make_fixed(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        return self.make_batch(n, rng)


def mqar_loss_acc(
    logits: torch.Tensor, tokens: torch.Tensor, answers: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Loss/accuracy at the final supervised position (after the query key)."""
    pred_logits = logits[:, -2, :]
    loss = torch.nn.functional.cross_entropy(pred_logits, tokens[:, -1])
    correct = int((pred_logits.argmax(-1) == tokens[:, -1]).sum())
    return loss, correct
