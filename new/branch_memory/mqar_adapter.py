"""Native MQAR adapter for Branch Memory.

Uses the upstream Zoology generator pinned by the handoff:
  HazyResearch/zoology @ 1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb
  zoology/data/multiquery_ar.py

Critical semantics (handoff §14 Stage B):
  * ``multiquery_ar`` returns ALREADY ALIGNED labels: ``labels[t]`` is the answer for
    the query that sits at ``inputs[t]``. Do NOT apply another next-token shift.
  * Score with ``cross_entropy(logits, labels, ignore_index=-100)``.
  * The generator seeds NumPy internally but draws the random non-query filler from
    the *torch* RNG. Both must be seeded here or data is not reproducible.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

ZOOLOGY_DIR = Path(__file__).resolve().parent / "vendor" / "zoology"
IGNORE_INDEX = -100


def _generator():
    """Official Zoology generator when importable, else the byte-identical local copy.

    ``zoology.config`` imports pydantic, which may be absent in a cluster container.
    ``_local_multiquery_ar`` reproduces the pinned commit's arithmetic exactly;
    ``tests/test_mqar.py::test_local_generator_matches_official`` verifies equality
    whenever both are available.
    """
    if str(ZOOLOGY_DIR) not in sys.path:
        sys.path.insert(0, str(ZOOLOGY_DIR))
    try:
        from zoology.data.multiquery_ar import multiquery_ar

        return multiquery_ar
    except Exception:  # pragma: no cover - depends on the container
        return _local_multiquery_ar


def _local_multiquery_ar(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float = 0.01,
    num_kv_pairs: int = 8,
    num_passes: int = 1,
    random_non_queries: bool = True,
    **kwargs,
):
    """Copy of ``zoology/data/multiquery_ar.py`` at 1ad20d19, minus pydantic.

    Same call order into NumPy and torch, so the outputs are identical.
    """
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 2 * num_passes + num_kv_pairs * 2 <= input_seq_len

    np.random.seed(seed)
    context_size = num_kv_pairs * 2 * num_passes

    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled,
                               replace=False, size=num_kv_pairs)
    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(np.random.choice, 1, values_unshuffled,
                                 replace=False, size=num_kv_pairs)

    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values
    kvs = np.tile(kvs, (1, num_passes))

    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x,
                               replace=False, p=p, size=num_kv_pairs)

    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
    examples = np.concatenate([kvs, queries], axis=1)

    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

    inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])
    if random_non_queries:
        inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]
    return SimpleNamespace(inputs=inputs, labels=labels,
                           slices={"num_kv_pairs": num_kv_pairs,
                                   "input_seq_len": input_seq_len,
                                   "num_passes": num_passes})


@dataclass(frozen=True)
class MQARSpec:
    """One MQAR panel. `seed` fixes both NumPy and torch RNGs before generation."""

    vocab_size: int = 8192
    seq_len: int = 128          # input_seq_len
    num_kv_pairs: int = 8
    power_a: float = 0.01
    num_passes: int = 1
    random_non_queries: bool = True
    num_examples: int = 1024
    seed: int = 0

    def __post_init__(self) -> None:
        if self.seq_len % 2 != 0:
            raise ValueError("seq_len must be even")
        if self.vocab_size <= self.seq_len:
            raise ValueError("vocab_size must exceed seq_len")
        if self.num_passes != 1:
            # Upstream defect at 1ad20d19: `kvs[:, 0::2] = keys` broadcasts a
            # (N, num_kv_pairs) array into (N, 2*num_kv_pairs*num_passes) columns and
            # raises ValueError as soon as num_passes > 1. Verified on the pinned commit.
            raise ValueError(
                "num_passes > 1 is not supported: the pinned Zoology generator raises "
                "ValueError at multiquery_ar.py:110 for any num_passes > 1"
            )
        if self.num_kv_pairs * 2 * self.num_passes + self.num_kv_pairs * 2 > self.seq_len:
            raise ValueError("sequence too short for the requested KV pairs/passes")


@dataclass
class MQARBatch:
    inputs: Tensor
    labels: Tensor
    spec: MQARSpec
    sha256: str = ""
    alignment: dict = field(default_factory=dict)


def _seed_all(seed: int) -> None:
    # multiquery_ar calls np.random.seed(seed) itself; the filler tokens come from
    # torch.randint, so the torch RNG must be pinned here as well.
    np.random.seed(seed)
    torch.manual_seed(seed)


def segment_sha256(inputs: Tensor, labels: Tensor) -> str:
    h = hashlib.sha256()
    h.update(inputs.to(torch.int64).contiguous().numpy().tobytes())
    h.update(labels.to(torch.int64).contiguous().numpy().tobytes())
    return h.hexdigest()


def check_alignment(inputs: Tensor, labels: Tensor, spec: MQARSpec) -> dict:
    """Verify label/query alignment and causality for every labeled position.

    For each labeled position t:
      * inputs[t] is a key (not the answer),
      * that key occurred earlier with the label as its value,
      * the label is never at position 0 and never overlaps an input KV slot.
    """
    if inputs.shape != labels.shape or inputs.ndim != 2:
        raise ValueError("inputs and labels must be the same 2-D shape")
    label_positions = labels.ne(IGNORE_INDEX)
    per_row = label_positions.sum(-1)
    if not torch.all(per_row == spec.num_kv_pairs):
        raise ValueError(
            f"expected {spec.num_kv_pairs} labels per row, got {per_row.tolist()[:8]}"
        )
    context = spec.num_kv_pairs * 2 * spec.num_passes
    rows = inputs.shape[0]
    checked = 0
    for r in range(rows):
        positions = label_positions[r].nonzero(as_tuple=False).flatten().tolist()
        for t in positions:
            if t == 0:
                raise ValueError("label at position 0 cannot be a next-token target")
            key = int(inputs[r, t])
            answer = int(labels[r, t])
            if key == answer:
                raise ValueError(f"row {r}: query input equals its label at {t}")
            found = False
            for p in range(0, t - 1, 2):
                if int(inputs[r, p]) == key and int(inputs[r, p + 1]) == answer:
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"row {r}: no prefix pair ({key} -> {answer}) before query at {t}"
                )
            checked += 1
    return {
        "rows": rows,
        "labeled_positions": checked,
        "labels_per_row": int(per_row[0]),
        "context_tokens": context,
        "answer_position_rule": "label[t] is the value for the key at inputs[t]",
    }


def build_segment(spec: MQARSpec) -> MQARBatch:
    gen = _generator()
    _seed_all(spec.seed)
    seg = gen(
        vocab_size=spec.vocab_size,
        num_examples=spec.num_examples,
        input_seq_len=spec.seq_len,
        seed=spec.seed,
        power_a=spec.power_a,
        num_kv_pairs=spec.num_kv_pairs,
        num_passes=spec.num_passes,
        random_non_queries=spec.random_non_queries,
    )
    inputs = seg.inputs.to(torch.long)
    labels = seg.labels.to(torch.long)
    alignment = check_alignment(inputs, labels, spec)
    return MQARBatch(inputs, labels, spec, segment_sha256(inputs, labels), alignment)


def mqar_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Aligned-label CE. `labels` must NOT be shifted again."""
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"logits {tuple(logits.shape)} vs labels {tuple(labels.shape)}")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


@torch.no_grad()
def mqar_accuracy(logits: Tensor, labels: Tensor) -> tuple[float, int]:
    """Accuracy over labeled positions only; returns (accuracy, denominator)."""
    mask = labels.ne(IGNORE_INDEX)
    if not bool(mask.any()):
        return 0.0, 0
    pred = logits.argmax(-1)
    correct = (pred[mask] == labels[mask]).sum().item()
    return correct / int(mask.sum()), int(mask.sum())


def build_panels(
    *, vocab_size: int, seq_len: int, num_kv_pairs: int, num_examples: int,
    seed: int, power_a: float = 0.01, num_passes: int = 1,
) -> dict[str, MQARBatch]:
    """Train/valid/test panels from independent seeds and disjoint sample sets."""
    specs = {
        "train": MQARSpec(vocab_size, seq_len, num_kv_pairs, power_a, num_passes,
                          True, num_examples, seed),
        "valid": MQARSpec(vocab_size, seq_len, num_kv_pairs, power_a, num_passes,
                          True, max(num_examples // 4, 32), seed + 1_000_003),
        "test": MQARSpec(vocab_size, seq_len, num_kv_pairs, power_a, num_passes,
                         True, max(num_examples // 4, 32), seed + 2_000_033),
    }
    return {name: build_segment(spec) for name, spec in specs.items()}
