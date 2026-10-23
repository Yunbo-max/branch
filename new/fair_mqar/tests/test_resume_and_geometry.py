"""Resume-equivalence and single-state geometry-ablation tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))

from chunked_branch import ChunkedBranchConfig, ChunkedBranchMixer  # noqa: E402
from fair_models import FairLMConfig, build_mixer  # noqa: E402


def _tiny_corpus(root: Path) -> None:
    rng = np.random.default_rng(0)
    for shard in range(2):
        arr = rng.integers(0, 64, size=(200, 64), dtype=np.uint16)
        np.save(root / f"shard_{shard}.npy", arr)


def _run_pretrain(
    root: Path, out: Path, steps: int, resume: Path | None = None
) -> None:
    cmd = [
        sys.executable,
        str(_HERE / "pretrain_branch.py"),
        "--packed-dir",
        str(root),
        "--seq-len",
        "16",
        "--batch-size",
        "2",
        "--steps",
        str(steps),
        "--eval-every",
        "2",
        "--eval-batches",
        "1",
        "--mixer",
        "chunked",
        "--d-model",
        "16",
        "--layers",
        "1",
        "--heads",
        "2",
        "--d-key",
        "4",
        "--d-value",
        "4",
        "--d-route",
        "4",
        "--max-states",
        "4",
        "--chunk-len",
        "4",
        "--vocab-size",
        "64",
        "--val-per-mille",
        "300",
        "--test-per-mille",
        "100",
        "--log-every",
        "1",
        "--checkpoint-every",
        "2",
        "--threads",
        "1",
        "--output",
        str(out),
        "--device",
        "cpu",
    ]
    if resume is not None:
        cmd += ["--resume", str(resume)]
    subprocess.run(cmd, check=True, cwd=_HERE, capture_output=True, text=True)


def test_resume_matches_uninterrupted_run(tmp_path: Path):
    root = tmp_path / "packed"
    root.mkdir()
    _tiny_corpus(root)
    out_a = tmp_path / "full"
    out_b = tmp_path / "split"
    _run_pretrain(root, out_a, steps=4)
    _run_pretrain(root, out_b, steps=2)
    _run_pretrain(root, out_b, steps=4, resume=out_b / "checkpoint.pt")

    a = [json.loads(x) for x in (out_a / "metrics.jsonl").read_text().splitlines()]
    b = [json.loads(x) for x in (out_b / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in a] == [1, 2, 3, 4]
    assert [r["step"] for r in b] == [1, 2, 3, 4]
    for ra, rb in zip(a, b):
        assert ra["train_loss"] == rb["train_loss"]
        assert ra.get("eval_loss") == rb.get("eval_loss")

    ckpt_a = torch.load(out_a / "checkpoint.pt", map_location="cpu", weights_only=False)
    ckpt_b = torch.load(out_b / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ckpt_a["step"] == ckpt_b["step"] == 4
    for key in ckpt_a["model"]:
        assert torch.equal(ckpt_a["model"][key], ckpt_b["model"][key])
    for key in (
        "optimizer",
        "gen_state",
        "torch_rng_state",
        "numpy_rng_state",
        "python_rng_state",
    ):
        assert key in ckpt_b

    assert (out_a / "checkpoint.pt").exists()
    assert not list(out_a.glob("checkpoint_*.pt"))
    assert (out_b / "checkpoint.pt").exists()
    assert not list(out_b.glob("checkpoint_*.pt"))

    report = json.loads((out_b / "report.json").read_text())
    assert report["resume"]["start_step"] == 2
    assert report["resume"]["data_stream_continuous"] is True
    assert report["training"]["steps"] == 4


def test_isotropic_geometry_ignores_covariance():
    cfg = ChunkedBranchConfig(
        d_model=16,
        n_heads=2,
        d_key=4,
        d_value=4,
        d_route=4,
        max_states=4,
        chunk_len=4,
        protect=0.0,
        covariance="full",
    )
    m = ChunkedBranchMixer(cfg)
    B, H, M, K = 1, 2, 4, 4
    a = torch.randn(B, H, M, K, K)
    C = a @ a.transpose(-1, -2)
    got = m._geometry_factor(C)
    ref = torch.linalg.cholesky(torch.eye(K)).expand(B, M, H, K, K)
    assert torch.allclose(got, ref, atol=1e-6)

    m_full = ChunkedBranchMixer(
        ChunkedBranchConfig(
            d_model=16,
            n_heads=2,
            d_key=4,
            d_value=4,
            d_route=4,
            max_states=4,
            chunk_len=4,
            protect=8.0,
            covariance="full",
        )
    )
    assert not torch.allclose(m_full._geometry_factor(C), ref, atol=1e-6)


def test_geometry_ablation_changes_only_protect():
    base = dict(
        mixer="chunked",
        vocab_size=64,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_key=4,
        d_value=4,
        d_route=4,
        max_states=1,
        chunk_len=4,
        covariance="full",
    )
    iso = build_mixer(FairLMConfig(**base, protect=0.0))
    full = build_mixer(FairLMConfig(**base, protect=8.0))
    assert iso.config.protect == 0.0
    assert full.config.protect == 8.0
    assert iso.config.max_states == full.config.max_states == 1
    keys = sorted(iso.state_dict())
    assert keys == sorted(full.state_dict())
    for key in keys:
        assert iso.state_dict()[key].shape == full.state_dict()[key].shape
