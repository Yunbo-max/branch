"""The pydantic-free generator fallback must match the pinned Zoology generator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))

from mqar_adapter import ZOOLOGY_DIR, _local_multiquery_ar  # noqa: E402

CASES = [
    (0, {}),
    (11, {}),
    (3, {"random_non_queries": False}),
]


def test_multi_pass_is_rejected_with_a_clear_error():
    """The pinned upstream generator crashes for num_passes > 1; surface it explicitly."""
    from mqar_adapter import MQARSpec

    with pytest.raises(ValueError, match="num_passes > 1"):
        MQARSpec(vocab_size=256, seq_len=64, num_kv_pairs=8, num_passes=2)

    if str(ZOOLOGY_DIR) not in sys.path:
        sys.path.insert(0, str(ZOOLOGY_DIR))
    try:
        from zoology.data.multiquery_ar import multiquery_ar as official
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"official generator unavailable: {exc}")

    np.random.seed(0)
    torch.manual_seed(0)
    with pytest.raises(ValueError, match="broadcast"):
        official(vocab_size=256, num_examples=4, input_seq_len=64, num_kv_pairs=8,
                 num_passes=2, seed=0)


@pytest.mark.parametrize("seed,extra", CASES)
def test_local_generator_matches_official(seed, extra):
    if str(ZOOLOGY_DIR) not in sys.path:
        sys.path.insert(0, str(ZOOLOGY_DIR))
    try:
        from zoology.data.multiquery_ar import multiquery_ar as official
    except Exception as exc:  # pragma: no cover - pydantic missing in the container
        pytest.skip(f"official generator unavailable: {exc}")

    kw = dict(vocab_size=256, num_examples=6, input_seq_len=64, num_kv_pairs=8,
              power_a=0.01, num_passes=1, random_non_queries=True)
    kw.update(extra)

    np.random.seed(seed)
    torch.manual_seed(seed)
    ref = official(seed=seed, **kw)
    np.random.seed(seed)
    torch.manual_seed(seed)
    got = _local_multiquery_ar(seed=seed, **kw)

    assert torch.equal(got.inputs, ref.inputs), f"inputs differ: seed={seed} {extra}"
    assert torch.equal(got.labels, ref.labels), f"labels differ: seed={seed} {extra}"


def test_fallback_alone_is_self_consistent():
    np.random.seed(1)
    torch.manual_seed(1)
    a = _local_multiquery_ar(vocab_size=128, num_examples=4, input_seq_len=32,
                             num_kv_pairs=4, seed=7)
    np.random.seed(1)
    torch.manual_seed(1)
    b = _local_multiquery_ar(vocab_size=128, num_examples=4, input_seq_len=32,
                             num_kv_pairs=4, seed=7)
    assert torch.equal(a.inputs, b.inputs) and torch.equal(a.labels, b.labels)
    assert int((a.labels != -100).sum()) == 4 * 4
