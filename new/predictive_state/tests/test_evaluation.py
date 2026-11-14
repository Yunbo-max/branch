import json
import torch
from psdraft import DrafterConfig, PredictiveStateDrafter
from psdraft.evaluation import evaluate_acceptance
from test_training import TinyTeacher


def test_reference_acceptance_protocol(tmp_path):
    data = tmp_path / 'prompts.jsonl'
    data.write_text('{"id":"x","prompt":"Compute 2+2","answer":"4"}\n')
    drafter = PredictiveStateDrafter(DrafterConfig(vocab_size=13, teacher_hidden_size=12,
                                                  hidden_size=16, n_layers=2,
                                                  n_heads=2, head_dim=8))
    teacher = TinyTeacher()
    summary = evaluate_acceptance(drafter, teacher, data_path=str(data),
                                  report_dir=str(tmp_path/'eval'), horizon=3,
                                  samples=3, seed=42)
    assert summary['prompt_count'] == 1
    assert 0 <= summary['mean_accepted_draft_tokens'] <= 3
    assert summary['tokens_per_second'] is None
    lines = (tmp_path/'eval'/'one_block_rows.jsonl').read_text().splitlines()
    assert len(lines) == 3
    assert all('answer' not in line for line in lines)
