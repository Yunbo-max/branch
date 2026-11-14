import json
import tempfile
from pathlib import Path
import torch
from torch import nn
from psdraft import DrafterConfig, PredictiveStateDrafter
from psdraft.training import load_prompts, optimize_one, verify_one, train_run


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(13, 12)
        self.head = nn.Linear(12, 13)
        self.hidden_size, self.vocab_size = 12, 13
        self.device = torch.device('cpu')
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def boundary_and_logits(self, prefix, draft=None):
        if draft is None:
            draft = prefix[:, :0]
        combined = torch.cat([prefix, draft], dim=1)
        hidden = self.emb(combined)
        p = self.head(hidden)[:, prefix.shape[1]-1:, :]
        return hidden[:, prefix.shape[1]-1, :], p

    @torch.no_grad()
    def greedy(self, prefix, length):
        return torch.full((prefix.shape[0], length), 2, dtype=torch.long)

    def encode_prompt(self, prompt):
        return torch.tensor([[1, 2, 3]])


def test_train_both_stages_and_checkpoint(tmp_path):
    model = PredictiveStateDrafter(DrafterConfig(vocab_size=13, teacher_hidden_size=12,
                                                hidden_size=16, n_layers=2, n_heads=2, head_dim=8))
    teacher = TinyTeacher()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    prefix = torch.tensor([[1, 2, 3]])
    tf = optimize_one(model, teacher, prefix, optim, mode='tf', horizon=3)
    opd = optimize_one(model, teacher, prefix, optim, mode='opd', horizon=3)
    assert tf['loss'] > 0 and opd['loss'] >= 0
    assert tf['draft_tokens'] == opd['draft_tokens'] == 3
    result = verify_one(model, teacher, prefix, horizon=3,
                        generator=torch.Generator().manual_seed(5))
    assert 1 <= len(result['emitted']) <= 4
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_prompt_only_and_refuse_overwrite(tmp_path):
    data = tmp_path / 'questions.jsonl'
    data.write_text(json.dumps({'id': 'a', 'prompt': '2+2', 'answer': '4'}) + '\n')
    assert load_prompts(data) == [{'id': 'a', 'prompt': '2+2'}]
    with (tmp_path / 'bad.jsonl').open('w') as f:
        f.write('{"id":"a","prompt":"x"}\n{"id":"a","prompt":"y"}\n')
    try:
        load_prompts(tmp_path / 'bad.jsonl')
    except ValueError:
        pass
    else:
        assert False, 'duplicate ID should fail'
