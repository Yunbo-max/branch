"""Anchor placement and visibility (paper Eq. 8-9)."""

from __future__ import annotations

import torch
from torch import Tensor


def anchor_layout(lengths: Tensor, interval: int, max_anchors: int | None = None):
    """Compute anchor boundaries for a batch of unpadded sequences.

    Args:
        lengths:  (B,) number of text tokens per sequence.
        interval: C, text tokens between anchors.
        max_anchors: optional pad width; defaults to the batch maximum.
    Returns:
        boundaries: (B, M) int64, number of text tokens before each anchor (b_m);
                    padded entries are 0 and must be masked by ``counts``.
        counts:     (B,) int64, number of real anchors per sequence.
    """
    b = lengths.shape[0]
    counts = torch.div(lengths, interval, rounding_mode="floor")
    m = max_anchors if max_anchors is not None else int(counts.max().item())
    m = max(m, 1)
    ar = torch.arange(1, m + 1, device=lengths.device).unsqueeze(0)  # (1, M)
    boundaries = ar * interval
    valid = ar <= counts.unsqueeze(1)
    boundaries = torch.where(valid, boundaries, torch.zeros_like(boundaries))
    return boundaries, counts


def expand_with_anchors(
    input_ids: Tensor, anchor_token_id: int, interval: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Insert one shared anchor token after every ``interval`` text tokens.

    Args:
        input_ids: (B, L) right-padded token ids.
    Returns:
        expanded_ids: (B, L + M) ids with anchors inserted.
        text_mask:    (B, L + M) bool, True at text positions.
        anchor_pos:   (B, M) long, expanded positions of anchors (-1 padded).
        boundaries:   (B, M) long, b_m = number of text tokens before anchor m.
    """
    b, seq = input_ids.shape
    counts = torch.div(torch.full((b,), seq, device=input_ids.device), interval, rounding_mode="floor")
    m = max(int(counts.max().item()), 1)
    boundaries, counts = anchor_layout(torch.full((b,), seq, device=input_ids.device), interval, m)

    pieces: list[Tensor] = []
    anchor_pos = torch.full((b, m), -1, dtype=torch.long, device=input_ids.device)
    text_mask_parts: list[Tensor] = []

    # Build the expanded layout segment by segment: [chunk 1][anchor 1][chunk 2][anchor 2]...
    # Positions are identical across the batch because sequences are the same length.
    out_len = seq + m
    text_mask = torch.zeros((b, out_len), dtype=torch.bool, device=input_ids.device)
    expanded = torch.full((b, out_len), anchor_token_id, dtype=input_ids.dtype, device=input_ids.device)

    cursor = 0
    prev = 0
    for j in range(m):
        nxt = int(boundaries[0, j].item())
        if nxt == 0:
            continue
        expanded[:, cursor:cursor + (nxt - prev)] = input_ids[:, prev:nxt]
        text_mask[:, cursor:cursor + (nxt - prev)] = True
        cursor += nxt - prev
        anchor_pos[:, j] = cursor
        cursor += 1
        prev = nxt
    if prev < seq:
        expanded[:, cursor:cursor + (seq - prev)] = input_ids[:, prev:]
        text_mask[:, cursor:cursor + (seq - prev)] = True
        cursor += seq - prev
    assert cursor == out_len, (cursor, out_len)
    return expanded, text_mask, anchor_pos, boundaries
