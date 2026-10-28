"""Online MQAR: deterministic fresh-sample stream, fixed validation panels, CPU smoke."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_fair import add_common_args, run  # noqa: E402
from mqar_adapter import MQARSpec, build_segment, check_alignment  # noqa: E402


def test_online_examples_deterministic_and_fresh():
    spec = MQARSpec(vocab_size=256, seq_len=64,
                    num_kv_pairs=2, num_examples=8, seed=3010000)
    a = build_segment(spec)
    b = build_segment(spec)
    c = build_segment(replace(spec, seed=3010001))
    assert torch.equal(a.inputs, b.inputs)
    assert torch.equal(a.labels, b.labels)
    assert a.sha256 == b.sha256
    assert a.sha256 != c.sha256
    assert not torch.equal(a.inputs, c.inputs)
    assert check_alignment(a.inputs, a.labels, a.spec)["labels_per_row"] == 2


def test_cpu_online_training_writes_report(tmp_path):
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args([
        "--mixer", "delta", "--device", "cpu",
        "--online-train", "--vocab-size", "256",
        "--seq-len", "64", "--num-kv-pairs", "2",
        "--num-examples", "32", "--eval-examples", "16",
        "--d-model", "16", "--layers", "1", "--heads", "1",
        "--d-key", "8", "--d-value", "8", "--d-route", "8",
        "--steps", "2", "--batch-size", "2",
        "--eval-batch-size", "8",
        "--eval-every", "1", "--log-every", "2",
        "--threads", "1", "--output", str(tmp_path),
    ])
    report = run(args)
    assert report["training"]["online_train"] is True
    assert report["training"]["online_examples_generated"] == 4
    assert report["training"]["steps"] == 2
    assert (tmp_path / "checkpoint.pt").exists()
    assert (tmp_path / "report.json").exists()
    j = json.loads((tmp_path / "report.json").read_text())
    assert j["panels"]["valid"]["sha256"] == report["panels"]["valid"]["sha256"]
    assert 0 <= report["final"]["valid"]["accuracy"] <= 1
