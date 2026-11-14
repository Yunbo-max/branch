import torch
import pytest
from psdraft import DrafterConfig, PredictiveStateDrafter, forward_kl, teacher_forcing_loss
from psdraft.gdn import GatedDeltaCell


def build():
    return PredictiveStateDrafter(DrafterConfig(vocab_size=19, teacher_hidden_size=12,
                                               hidden_size=16, n_layers=2,
                                               n_heads=2, head_dim=8))


def test_state_math_exact_and_gradient():
    torch.manual_seed(0)
    cell = GatedDeltaCell(16, 2, 8)
    x = torch.randn(2, 16)
    s = torch.randn(2, 2, 8, 8)
    out, nxt = cell(x, s)
    h, d = 2, 8
    hidden = x
    k = torch.nn.functional.normalize(cell.k_proj(hidden).view(2, h, d), dim=-1)
    a = torch.sigmoid(cell.a_proj(hidden)).view(2, h, 1, 1)
    beta = torch.sigmoid(cell.b_proj(hidden)).view(2, h, 1, 1)
    v = cell.v_proj(hidden).view(2, h, d)
    sa = s * a
    manual = sa + beta * (v - torch.einsum('bhvk,bhk->bhv', sa, k)).unsqueeze(-1) * k.unsqueeze(-2)
    assert torch.allclose(nxt, manual, atol=1e-6)
    (out.square().mean() + nxt.square().mean()).backward()
    assert cell.k_proj.weight.grad.abs().sum() > 0
    assert cell.b_proj.weight.grad.abs().sum() > 0


def test_draft_condition_and_teacher_forcing_alignment():
    torch.manual_seed(1)
    model = build()
    cond = torch.randn(2, 12)
    draft = torch.randint(0, 19, (2, 4), dtype=torch.long)
    all_logits = model.logits_on_draft(cond, draft, include_bonus=True)
    assert all_logits.shape == (2, 5, 19)
    logits, state = model.start(cond)
    assert torch.allclose(all_logits[:, 0], logits)
    for j in range(4):
        logits, state = model.advance(draft[:, j], state)
        assert torch.allclose(all_logits[:, j + 1], logits, atol=2e-5)
    train_logits = model.logits_on_draft(cond, draft)
    tf_loss = teacher_forcing_loss(train_logits, draft)
    kl = forward_kl(train_logits, torch.randn_like(train_logits))
    (tf_loss + kl).backward()
    assert model.condition[1].weight.grad.abs().sum() > 0
    assert model.embedding.weight.grad.abs().sum() > 0


def test_sample_replay_logprob_consistency():
    model = build().eval()
    cond = torch.randn(1, 12)
    tokens, proposal = model.sample(cond, 5, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        logits = model.logits_on_draft(cond, tokens)
        replay = torch.softmax(logits.float(), -1)
    assert torch.allclose(replay, proposal, atol=1e-6)
    assert model.recurrent_state_bytes() == 2 * 2 * 8 * 8 * 2


def test_shape_fail_fast():
    with pytest.raises(ValueError):
        DrafterConfig(hidden_size=15)
    with pytest.raises(ValueError):
        build().start(torch.randn(1, 13))


def test_kl_zero_and_teacher_stopgrad():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 7)
    assert abs(float(forward_kl(logits, logits))) < 1e-6
    teacher = torch.randn(2, 3, 7, requires_grad=True)
    student = torch.randn(2, 3, 7, requires_grad=True)
    forward_kl(student, teacher).backward()
    assert teacher.grad is None
    assert student.grad.abs().sum() > 0
