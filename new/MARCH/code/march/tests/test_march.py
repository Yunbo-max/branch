"""CPU smoke tests for the MARCH reference implementation.

Run:  python -m pytest code/march/tests -q
"""

from __future__ import annotations

import torch

from march import MARCHConfig, MARCHModel, anchor_layout, expand_with_anchors, routing_weights


def tiny_config(**kw) -> MARCHConfig:
    base = dict(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        routing_dim=8,
        anchor_interval=4,
        anchor_token_id=127,
    )
    base.update(kw)
    return MARCHConfig(**base)


def test_anchor_layout_counts():
    lengths = torch.tensor([10, 7])
    boundaries, counts = anchor_layout(lengths, interval=4)
    assert counts.tolist() == [2, 1]
    assert boundaries[0].tolist() == [4, 8]
    assert boundaries[1].tolist() == [4, 0]  # padded slot masked by counts


def test_expand_layout():
    ids = torch.arange(10).view(1, 10)
    expanded, text_mask, anchor_pos, boundaries = expand_with_anchors(ids, 127, 4)
    # 10 text + 2 anchors = 12
    assert expanded.shape == (1, 12)
    assert text_mask.sum().item() == 10
    assert anchor_pos.tolist() == [[4, 9]]
    assert boundaries.tolist() == [[4, 8]]
    assert expanded[0, 4].item() == 127 and expanded[0, 9].item() == 127
    # text order is preserved
    assert expanded[0, text_mask[0]].tolist() == list(range(10))


def test_null_routing_when_no_anchor_visible():
    b, t, m, dr = 1, 3, 2, 8
    rho = torch.randn(b, t, dr)
    keys = torch.randn(b, m, dr)
    null = torch.zeros(b, t)
    visible = torch.zeros(b, t, m, dtype=torch.bool)  # nothing visible yet
    pi = routing_weights(rho, keys, null, visible)
    assert pi.shape == (b, t, m + 1)
    assert torch.allclose(pi[..., -1], torch.ones(b, t), atol=1e-6)


def test_top_k_masks_extra_anchors():
    b, t, m, dr = 1, 1, 5, 8
    rho = torch.randn(b, t, dr)
    keys = torch.randn(b, m, dr)
    null = torch.zeros(b, t)
    visible = torch.ones(b, t, m, dtype=torch.bool)
    pi = routing_weights(rho, keys, null, visible, top_k=2)
    nonzero = (pi[..., :-1] > 0).sum().item()
    assert nonzero == 2


def test_forward_shape():
    cfg = tiny_config()
    model = MARCHModel(cfg)
    ids = torch.randint(0, 100, (2, 9))
    logits = model(ids)
    assert logits.shape == (2, 9, cfg.vocab_size)


def test_causality():
    """Changing a future token must not change earlier text logits."""
    torch.manual_seed(0)
    cfg = tiny_config()
    model = MARCHModel(cfg).eval()
    ids = torch.randint(0, 100, (1, 12))
    with torch.no_grad():
        a = model(ids)
        ids2 = ids.clone()
        ids2[0, 9] = (ids2[0, 9] + 1) % 100
        b = model(ids2)
    # Positions 0..8 are before the modified token; the anchor at text boundary 8
    # sits after position 7, so nothing at or before index 8 may see position 9.
    assert torch.allclose(a[0, :9], b[0, :9], atol=1e-6), (
        (a[0, :9] - b[0, :9]).abs().max().item()
    )
    # And the change must actually propagate forward.
    assert not torch.allclose(a[0, 9:], b[0, 9:], atol=1e-6)
