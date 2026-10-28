"""CPU-only regression tests; do not download pretrained checkpoint in pytest."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from backend import single_token_choices  # noqa: E402
from spec import CONDITIONS, make_probe  # noqa: E402
from summarize import summarize_records  # noqa: E402
import run as census_run  # noqa: E402


class CharTokenizer:
    """Deterministic test tokenizer; one character per token."""
    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 10 for c in text]


@pytest.mark.parametrize("condition", CONDITIONS)
def test_exact_length_and_reproducibility(condition):
    tok = CharTokenizer()
    p = make_probe(tok, condition=condition, length=4096, seed=17,
                   needle_fraction=.3, num_events=8)
    q = make_probe(tok, condition=condition, length=4096, seed=17,
                   needle_fraction=.3, num_events=8)
    assert p == q
    assert p.actual_length == 4096
    assert len(p.token_ids) == 4096
    assert p.needle_token < 4096
    assert p.answer in "ABCD"
    assert p.question_suffix.endswith("Answer:")
    assert p.to_meta()["prompt_hash"] == p.prompt_hash
    assert "token_ids" not in p.to_meta()


def test_paired_variants_share_target_position_and_value():
    tok = CharTokenizer()
    variants = [
        make_probe(tok, condition=c, length=4096, seed=42,
                   needle_fraction=.17, num_events=16)
        for c in ("plain", "unrelated", "collision")
    ]
    assert len({p.pair_id for p in variants}) == 1
    assert len({p.answer for p in variants}) == 1
    assert len({p.needle_token for p in variants}) == 1
    assert len({p.key for p in variants}) == 1
    assert len({p.prompt_hash for p in variants}) == 3


def test_event_position_control():
    tok = CharTokenizer()
    early = make_probe(tok, condition="collision", length=4096,
                       seed=123, num_events=8, event_placement="early")
    late = make_probe(tok, condition="collision", length=4096,
                      seed=123, num_events=8, event_placement="late")
    assert early.needle_token == late.needle_token
    assert early.event_token < late.event_token
    assert len(early.token_ids) == len(late.token_ids)


def test_revision_recency_change_and_same_history():
    tok = CharTokenizer()
    latest = make_probe(tok, condition="revision_latest", length=4096,
                        seed=91, num_events=8)
    earliest = make_probe(tok, condition="revision_earliest", length=4096,
                          seed=91, num_events=8)
    assert latest.pair_id == earliest.pair_id
    assert latest.answer == earliest.latest_answer
    assert earliest.answer == earliest.initial_answer
    assert latest.answer != earliest.answer
    assert latest.token_ids != earliest.token_ids


def test_too_short_refuses_instead_of_truncating():
    tok = CharTokenizer()
    with pytest.raises(ValueError, match="required text"):
        make_probe(tok, condition="collision", length=128, seed=17,
                   num_events=100)


def test_choice_token_validation():
    ids = single_token_choices(CharTokenizer(), "\nQuestion: example?\nAnswer:")
    assert len(ids) == 4 and len(set(ids.values())) == 4
    assert ids["A"] == ord("A") + 10


def _rec(*, pair, condition, length=512, correct=1, events=8):
    return {
        "status": "ok", "probe_id": f"{pair}-{condition}",
        "pair_id": pair, "condition": condition,
        "actual_length": length, "needle_fraction": .15,
        "num_events": events, "correct": correct,
        "margin": 0.2, "gold_conditional_nll": 0.3,
        "latency_ms": 10., "is_context_extrapolated": False,
    }


def test_paired_summary():
    records = [
        _rec(pair="x", condition="plain", correct=1),
        _rec(pair="x", condition="unrelated", correct=1),
        _rec(pair="x", condition="collision", correct=0),
        _rec(pair="y", condition="plain", correct=1),
        _rec(pair="y", condition="unrelated", correct=1),
        _rec(pair="y", condition="collision", correct=1),
    ]
    s = summarize_records(records)
    assert s["short_control"]["competence_gate"] == "PASS"
    p = s["paired"][0]
    assert p["n_triplets"] == 2
    assert p["n_unrelated_correct"] == 2
    assert p["conditional_collision_failure_mean"] == pytest.approx(.5)
    assert p["unrelated_vs_collision_mean"] == pytest.approx(-.5)


def test_short_control_gate_blocks_bad_interpretation():
    rows = [_rec(pair="one", condition="plain", correct=0),
            _rec(pair="one", condition="unrelated", correct=0),
            _rec(pair="one", condition="collision", correct=0)]
    x = summarize_records(rows)
    assert x["short_control"]["competence_gate"] == "INSUFFICIENT_SHORT_CONTROL"


def test_dry_run_does_not_load_weights(tmp_path, monkeypatch):
    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return CharTokenizer()
    monkeypatch.setitem(
        sys.modules, "transformers",
        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )
    rc = census_run.main([
        "--dry-run", "--out", str(tmp_path),
        "--lengths", "2048", "--seeds", "1",
        "--events", "8", "--positions", "0.2",
        "--conditions", "plain,unrelated,collision",
    ])
    assert rc == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["planned_cases"] == 3
    assert manifest["dry_run"]
    assert not (tmp_path / "results.jsonl").exists()
