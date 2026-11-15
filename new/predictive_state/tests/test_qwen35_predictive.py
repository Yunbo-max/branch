"""CPU-only mock contract suite. Not equivalent to real Qwen3.5/CUDA qualification."""
from types import SimpleNamespace
import torch
from torch import nn
import pytest
from qwen35_predictive.features import FixedStateSketch, native_gdn_matrix, select_source
from qwen35_predictive.heads import StateJumpHead
from qwen35_predictive.rollout import PredictiveMTP
from qwen35_predictive.experiments import train_case, evaluate_case
from qwen35_predictive.checkpoint import load_small


class FakeMTP(nn.Module):
    def forward(self, hidden, next_embedding, *, past_key_values, use_cache, start_position):
        assert use_cache
        assert hidden.shape == next_embedding.shape and hidden.shape[0] == 1
        assert hidden.shape[-1] == 8
        assert (past_key_values is None and start_position == 0) or past_key_values == start_position
        if past_key_values is None:
            assert hidden.shape[1] == 3, 'native MTP must prefill the full accepted prefix'
        return (hidden + 0.3 * next_embedding).tanh(), start_position + hidden.shape[1]


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(19, 8)


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeBackbone()
        self.lm_head = nn.Linear(8, 19, bias=False)


class FakeBackend:
    """The same public interface used by the native bridge, without any pretrained weights."""
    def __init__(self):
        torch.manual_seed(9)
        self.base = FakeBase().eval()
        self.mtp = FakeMTP()
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.hidden, self.vocab, self.device = 8, 19, torch.device("cpu")
        self.native = SimpleNamespace(provenance={"mock_backend_not_real_model": True})

    def encode(self, prompt):
        return torch.tensor([[1, 2, 3 if len(prompt) % 2 else 4]], dtype=torch.long)

    @staticmethod
    def _state(tokens):
        state = torch.zeros(1, 32, 128, 128)
        value = float(tokens.float().sum()) / 20
        state[0, 0, 0, 0] = value
        state[0, 8, 3, 4] = value * 2
        return state

    @torch.no_grad()
    def boundary(self, prefix, *, collect_state=False):
        self._mtp_prefill_hidden = self.base.model.embed_tokens(prefix).detach()
        self._mtp_prefill_tokens = prefix.detach().clone()
        hidden = self._mtp_prefill_hidden[:, -1, :]
        state = self._state(prefix) if collect_state else None
        return hidden, hidden * 0.8, hidden * 0.2, state

    @torch.no_grad()
    def teacher(self, prefix, draft_ids, *, collect_end_state=False):
        full = torch.cat((prefix, draft_ids), dim=1)
        h = self.base.model.embed_tokens(full)
        logits = self.base.lm_head(h[:, prefix.shape[1]-1:, :]).float()
        assert logits.shape == (1, draft_ids.shape[1]+1, 19)
        return logits, self._state(full) if collect_end_state else None


def test_c00_c04_sample_replay_full_vocab_and_no_future_leak():
    native = FakeBackend()
    prefix = native.encode("hello")
    hidden, gdn, gqa, _ = native.boundary(prefix)
    for case in ("c00_native", "c01_gdn", "c02_gqa", "c03_fusion", "c04_state"):
        model = PredictiveMTP(case=case, hidden=8, rank=4).eval()
        proposal = model.sample(native, hidden, gdn, gqa, horizon=4,
                                generator=torch.Generator().manual_seed(41))
        replay = model.replay(native, hidden, gdn, gqa, proposal.ids)
        assert proposal.ids.shape == (1, 4)
        assert proposal.logits.shape == (1, 4, 19)
        assert torch.allclose(proposal.logits, replay.logits, atol=1e-6)
        assert torch.allclose(proposal.q_probs, replay.logits.float().softmax(-1), atol=1e-6)
        target, _ = native.teacher(prefix, proposal.ids)
        assert torch.allclose(proposal.logits[:, 0], target[:, 0], atol=1e-6)


def test_zero_init_and_distinct_readouts():
    torch.manual_seed(12)
    g, a = torch.randn(1, 8), torch.randn(1, 8)
    assert not torch.allclose(select_source("c01_gdn", g, a),
                              select_source("c02_gqa", g, a))
    assert torch.isfinite(select_source("c03_fusion", g, a)).all()
    model = PredictiveMTP(case="c03_fusion", hidden=8, rank=4)
    current = torch.randn(1, 8)
    assert torch.allclose(model.adapter(current, g), current, atol=1e-7)
    with pytest.raises(ValueError):
        select_source("not_a_case", g, a)


def test_actual_matrix_sketch_and_future_head_grad():
    torch.manual_seed(4)
    first = torch.randn(1, 32, 128, 128)
    final = torch.randn(1, 32, 128, 128)
    sketch = FixedStateSketch(rank=4, seed=11)
    assert sketch(first).shape == (1, 64)
    assert torch.equal(sketch(first), sketch(first))
    head = StateJumpHead(hidden=8, sketch_rank=8, width=32)
    z = torch.randn(1, 8, requires_grad=True)
    loss, stats = head.loss(z, first, final)
    assert bool(torch.isfinite(loss)) and stats["sketch_delta_rms"] > 0
    loss.backward()
    assert head.net[-1].weight.grad.abs().sum() > 0
    matrix = torch.ones(1, 32, 128, 128)
    layer = SimpleNamespace(recurrent_states=[matrix])
    cache = SimpleNamespace(layers=[None] * 26 + [layer])
    copied = native_gdn_matrix(cache)
    copied.zero_()
    assert matrix.sum() > 0
    with pytest.raises(RuntimeError):
        native_gdn_matrix(SimpleNamespace(layers=[]))


@pytest.mark.parametrize("case", ("c01_gdn", "c02_gqa", "c03_fusion", "c04_state"))
def test_mock_train_then_evaluate_and_checkpoint(tmp_path, case):
    native = FakeBackend()
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"id":"a","prompt":"Calculate 2+2","answer":"4"}\n'
                       '{"id":"b","prompt":"Write Python","answer":"secret"}\n')
    run = tmp_path / f"train_{case}"
    training = train_case(native, case=case, prompts=str(prompts), output=str(run),
                          horizon=3, updates=2, seed=11, learning_rate=1e-3,
                          rank=4, state_weight=0.1, max_prompts=2)
    assert training["updates"] == 2
    checkpoint = run / "adapter.pt"
    assert checkpoint.exists()
    model = load_small(checkpoint, case=case, hidden=8, rank=4,
                       state_weight=0.1, device="cpu")
    assert all(torch.isfinite(p).all() for p in model.parameters())
    evaluated = evaluate_case(native, case=case, prompts=str(prompts),
                              output=str(tmp_path / f"eval_{case}"),
                              checkpoint=str(checkpoint),
                              horizon=3, seed=43, samples=2, rank=4,
                              state_weight=0.1, max_prompts=2)
    assert evaluated["num_prompts"] == 2
    assert 0 <= evaluated["mean_accepted"] <= 3
    assert evaluated["true_cache_aware_tokens_per_second"] is None
    assert "secret" not in (tmp_path / f"eval_{case}" / "rows.jsonl").read_text()


def test_c00_evaluation_without_train(tmp_path):
    native = FakeBackend()
    prompts = tmp_path / "p.jsonl"
    prompts.write_text('{"id":"a","prompt":"Hello world"}\n')
    output = evaluate_case(native, case="c00_native", prompts=str(prompts),
                           output=str(tmp_path / "e00"), checkpoint=None, horizon=2,
                           seed=10, samples=2, rank=4, state_weight=0.1, max_prompts=1)
    assert output["case"] == "c00_native"
    with pytest.raises(ValueError):
        train_case(native, case="c00_native", prompts=str(prompts),
                   output=str(tmp_path / "not_allowed"), horizon=2, updates=1,
                   seed=11, learning_rate=1e-3, rank=4, state_weight=0.1,
                   max_prompts=1)
