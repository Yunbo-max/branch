"""Label-alignment and determinism tests for the native MQAR adapter.

These are the checks the handoff requires before any MQAR number is trusted.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from branch_memory import next_token_loss
from mqar_adapter import (
    IGNORE_INDEX,
    MQARSpec,
    build_panels,
    build_segment,
    check_alignment,
    mqar_accuracy,
    mqar_loss,
    segment_sha256,
)


def small_spec(seed: int = 0, **kw) -> MQARSpec:
    base = dict(vocab_size=128, seq_len=32, num_kv_pairs=4, num_examples=8, seed=seed)
    base.update(kw)
    return MQARSpec(**base)


def test_labels_are_query_aligned():
    seg = build_segment(small_spec())
    count = 0
    for r in range(seg.inputs.shape[0]):
        for t in (seg.labels[r] != IGNORE_INDEX).nonzero(as_tuple=False).flatten().tolist():
            key, answer = int(seg.inputs[r, t]), int(seg.labels[r, t])
            # The answer is the value paired with `key` somewhere in the prefix.
            hits = [p for p in range(0, t - 1, 2)
                    if int(seg.inputs[r, p]) == key and int(seg.inputs[r, p + 1]) == answer]
            assert hits, (r, t, key, answer)
            assert key != answer
            count += 1
    assert count == seg.inputs.shape[0] * seg.spec.num_kv_pairs
    assert seg.alignment["labeled_positions"] == count


def test_scoring_denominator_is_labeled_positions_only():
    seg = build_segment(small_spec())
    torch.manual_seed(0)
    logits = torch.randn(seg.inputs.shape[0], seg.inputs.shape[1], seg.spec.vocab_size)
    mask = seg.labels.ne(IGNORE_INDEX)
    n = int(mask.sum())
    assert n == seg.inputs.shape[0] * seg.spec.num_kv_pairs

    loss = mqar_loss(logits, seg.labels)
    manual = F.cross_entropy(logits[mask], seg.labels[mask])
    torch.testing.assert_close(loss, manual)

    acc, denominator = mqar_accuracy(logits, seg.labels)
    assert denominator == n
    assert 0.0 <= acc <= 1.0


def test_do_not_double_shift_aligned_labels():
    """Applying next_token_loss to already-aligned labels is wrong by construction."""
    seg = build_segment(small_spec())
    torch.manual_seed(0)
    logits = torch.randn(seg.inputs.shape[0], seg.inputs.shape[1], seg.spec.vocab_size)
    aligned = mqar_loss(logits, seg.labels)
    # next_token_loss shifts the targets one more time. It does NOT raise: PyTorch's
    # cross_entropy already defaults to ignore_index=-100, so the mistake is silent.
    shifted = next_token_loss(logits, seg.labels)
    assert not torch.allclose(aligned, shifted)
    # And it drops the first query's target and re-pairs labels with the wrong inputs.
    assert shifted.ndim == 0 and torch.isfinite(shifted)


def test_same_seed_reproduces_data_and_hash():
    a = build_segment(small_spec(seed=11))
    b = build_segment(small_spec(seed=11))
    assert a.sha256 == b.sha256
    assert torch.equal(a.inputs, b.inputs) and torch.equal(a.labels, b.labels)


def test_different_seed_changes_data():
    a = build_segment(small_spec(seed=11))
    b = build_segment(small_spec(seed=12))
    assert a.sha256 != b.sha256


def test_torch_filler_rng_is_pinned():
    """random_non_queries draws from torch; a prior torch state must not leak in."""
    torch.manual_seed(999)
    _ = torch.randn(1000)
    a = build_segment(small_spec(seed=5))
    torch.manual_seed(1)
    _ = torch.randn(50)
    b = build_segment(small_spec(seed=5))
    assert a.sha256 == b.sha256


def test_panels_are_independent():
    panels = build_panels(vocab_size=128, seq_len=32, num_kv_pairs=4,
                          num_examples=16, seed=3)
    hashes = [p.sha256 for p in panels.values()]
    assert len(set(hashes)) == len(hashes)
    for p in panels.values():
        assert p.alignment["labels_per_row"] == 4


def test_misaligned_labels_are_rejected():
    seg = build_segment(small_spec())
    broken = seg.labels.clone()
    broken[0, 5] = 7
    with pytest.raises(ValueError):
        check_alignment(seg.inputs, broken, seg.spec)


def test_spec_rejects_impossible_protocol():
    with pytest.raises(ValueError):
        MQARSpec(vocab_size=32, seq_len=32, num_kv_pairs=4)   # vocab must exceed seq
    with pytest.raises(ValueError):
        MQARSpec(vocab_size=128, seq_len=32, num_kv_pairs=16)  # too many pairs
    with pytest.raises(ValueError):
        MQARSpec(vocab_size=128, seq_len=31, num_kv_pairs=4)   # odd length
