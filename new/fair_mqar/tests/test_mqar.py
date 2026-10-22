"""Checks for the synthetic MQAR diagnostic."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))

from mqar import MQARConfig, mqar_loss_acc  # noqa: E402


def test_layout_and_answer():
    cfg = MQARConfig(seq_len=32, num_pairs=4, gap=8, seed=1)
    tokens, answers = cfg.make_fixed(16, seed=1)
    prefix = cfg.seq_len - (2 * cfg.num_pairs + cfg.gap + 2)

    assert tokens.shape == (16, cfg.seq_len)
    assert (tokens[:, :prefix] == cfg.filler_token).all()
    assert torch.equal(torch.from_numpy(tokens[:, -1]), torch.from_numpy(answers))
    assert int(tokens.max()) < cfg.vocab_size

    for i in range(tokens.shape[0]):
        pair_block = tokens[i, prefix : prefix + 2 * cfg.num_pairs]
        assert tokens[i, -2] in pair_block[0::2]
        # value associated with the query key is the final token
        key_positions = pair_block[0::2]
        idx = int((key_positions == tokens[i, -2]).nonzero()[0])
        assert pair_block[2 * idx + 1] == tokens[i, -1]


def test_pairs_and_gap_grid():
    for pairs, gap in ((2, 0), (4, 16), (8, 32)):
        cfg = MQARConfig(seq_len=128, num_pairs=pairs, gap=gap)
        tokens, answers = cfg.make_fixed(4, seed=pairs + gap)
        assert tokens.shape == (4, 128)
        assert torch.equal(torch.from_numpy(tokens[:, -1]), torch.from_numpy(answers))


def test_loss_accuracy_with_perfect_logits():
    cfg = MQARConfig(seq_len=32, num_pairs=3, gap=4, seed=2)
    tokens, answers = cfg.make_fixed(2, seed=2)
    ids = torch.from_numpy(tokens)
    logits = torch.full((2, cfg.seq_len, cfg.vocab_size), -20.0)
    logits[torch.arange(2), -2, torch.from_numpy(answers)] = 20.0
    loss, correct = mqar_loss_acc(logits, ids, ids[:, -1])
    assert correct == 2
    assert float(loss) < 1e-3
