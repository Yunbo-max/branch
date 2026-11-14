import torch
import pytest
from psdraft import verify_block


def test_full_accept_bonus_and_rejection():
    q = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
    p = torch.tensor([[0.2, 0.8], [0.7, 0.3], [0., 1.]])
    out = verify_block(torch.tensor([1, 0]), q, p, generator=torch.Generator().manual_seed(7))
    assert out.emitted == [1, 0, 1] and out.accepted == 2 and out.rejected_at is None
    q = torch.tensor([[1.0, 0.0]])
    p = torch.tensor([[0.0, 1.0], [0.5, 0.5]])
    out = verify_block(torch.tensor([0]), q, p)
    assert out.emitted == [1] and out.accepted == 0 and out.rejected_at == 0


def test_one_step_marginal_matches_target():
    q = torch.tensor([[0.72, 0.12, 0.16]])
    p = torch.tensor([[0.11, 0.34, 0.55], [0.2, 0.3, 0.5]])
    rng = torch.Generator().manual_seed(123)
    totals = torch.zeros(3)
    for _ in range(6000):
        draft = torch.multinomial(q[0], 1, generator=rng)
        got = verify_block(draft, q, p, generator=rng)
        totals[got.emitted[0]] += 1
    assert torch.allclose(totals / 6000, p[0], atol=0.024)


def test_reject_bad_shapes():
    with pytest.raises(ValueError):
        verify_block(torch.tensor([1]), torch.ones(1, 2), torch.ones(1, 2))
