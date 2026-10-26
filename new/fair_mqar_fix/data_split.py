"""Deterministic row-level split and a fixed evaluation panel for packed corpora.

The packed corpus has no document identifiers, so this module can only guarantee that the
*row indices* used for train, validation and test are disjoint. Sequences from the same
source document may still land in different splits; the manifest says so explicitly.

Every run that must be compared uses the same ``split_seed``, ``val_per_mille`` and
``test_per_mille``. The validation panel is drawn once and stored as tokens, so every
checkpoint of every arm is scored on exactly the same rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import torch

TRAIN, VAL, TEST = 0, 1, 2
_MASK64 = (1 << 64) - 1


def _bucket(indices: np.ndarray, seed: int, mod: int = 1000) -> np.ndarray:
    """Stable 64-bit hash of (index, seed) reduced to ``[0, mod)``."""
    z = np.asarray(indices, dtype=np.uint64) + np.uint64(seed & _MASK64)
    z = z ^ (z >> np.uint64(33))
    z = z * np.uint64(0xFF51AFD7ED558CCD)
    z = z ^ (z >> np.uint64(33))
    z = z * np.uint64(0xC4CEB9FE1A85EC53)
    z = z ^ (z >> np.uint64(33))
    return (z % np.uint64(mod)).astype(np.int64)


@dataclass
class RowSplit:
    seed: int = 12345
    val_per_mille: int = 20
    test_per_mille: int = 10

    def __post_init__(self) -> None:
        if not 0 <= self.val_per_mille < 1000:
            raise ValueError("val_per_mille must be in [0, 1000)")
        if not 0 <= self.test_per_mille < 1000:
            raise ValueError("test_per_mille must be in [0, 1000)")
        if self.val_per_mille + self.test_per_mille >= 1000:
            raise ValueError("val_per_mille + test_per_mille must be < 1000")

    def labels(self, indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(indices, dtype=np.int64)
        b = _bucket(idx, self.seed)
        out = np.full(idx.shape, TRAIN, dtype=np.int8)
        out[b < self.val_per_mille] = VAL
        hi = self.val_per_mille + self.test_per_mille
        out[(b >= self.val_per_mille) & (b < hi)] = TEST
        return out

    def train_indices(self, n: int, gen: torch.Generator, n_rows: int) -> np.ndarray:
        return _draw(n, gen, n_rows, partial=lambda idx: self.labels(idx) == TRAIN)

    def counts(self, n_rows: int) -> dict:
        labels = self.labels(np.arange(n_rows))
        return {
            "train": int((labels == TRAIN).sum()),
            "val": int((labels == VAL).sum()),
            "test": int((labels == TEST).sum()),
        }


def _draw(n: int, gen: torch.Generator, n_rows: int, partial) -> np.ndarray:
    """Rejection-sample ``n`` indices for which ``partial(indices)`` is true."""
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    out: list[np.ndarray] = []
    have = 0
    for _ in range(10000):
        draw = torch.randint(0, n_rows, (max(n, 64),), generator=gen).numpy()
        keep = draw[partial(draw)]
        if keep.size:
            out.append(keep)
            have += keep.size
        if have >= n:
            return np.concatenate(out)[:n]
    raise RuntimeError("could not draw enough rows for this split; check fractions")


@dataclass
class FixedPanel:
    indices: np.ndarray
    tokens: np.ndarray
    indices_sha256: str
    tokens_sha256: str

    def tensor(self, device: torch.device) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(self.tokens)).to(device)


def build_panel(corpus, n: int, split: RowSplit, gen: torch.Generator) -> FixedPanel:
    """Draw ``n`` validation rows once and freeze their tokenized content."""
    indices = _draw(n, gen, len(corpus), partial=lambda idx: split.labels(idx) == VAL)
    tokens = corpus.gather(indices)
    idx_bytes = np.ascontiguousarray(indices).tobytes()
    tok_bytes = np.ascontiguousarray(tokens).tobytes()
    return FixedPanel(
        indices=indices,
        tokens=tokens,
        indices_sha256=hashlib.sha256(idx_bytes).hexdigest(),
        tokens_sha256=hashlib.sha256(tok_bytes).hexdigest(),
    )


def manifest(split: RowSplit, corpus, panel: FixedPanel, seq_len: int) -> dict:
    counts = split.counts(len(corpus))
    return {
        "split": {
            "level": "row",
            "seed": split.seed,
            "val_per_mille": split.val_per_mille,
            "test_per_mille": split.test_per_mille,
            "counts": counts,
            "note": (
                "Packed corpus has no document ids. Rows are disjoint across splits, "
                "but sequences from the same source document may overlap."
            ),
        },
        "corpus": {
            "shards": len(corpus.files),
            "sequences": len(corpus),
            "seq_len": seq_len,
        },
        "panel": {
            "n": int(len(panel.indices)),
            "indices_sha256": panel.indices_sha256,
            "tokens_sha256": panel.tokens_sha256,
        },
    }


def write_manifest(path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
