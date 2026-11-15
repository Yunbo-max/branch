"""Atomic and strict small-module checkpoint; never modifies frozen 4B weights."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import torch
from .rollout import PredictiveMTP

MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def save_small(path: str | Path, model: PredictiveMTP, *, metadata: dict) -> None:
    dest = Path(path)
    if dest.exists():
        raise FileExistsError("checkpoint file already exists; use a new run ID")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".tmp")
    if temp.exists():
        raise FileExistsError("stale temporary checkpoint requires inspection")
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"state_dict": state, "meta": {"revision": MODEL_REVISION,
                  "case": model.case, "hidden": model.hidden,
                  "state_weight": model.state_weight, **metadata}}, temp)
    os.replace(temp, dest)


def load_small(path: str | Path, *, case: str, hidden: int = 2560, rank: int = 128,
               state_weight: float = 0.1, device: str = "cpu") -> PredictiveMTP:
    # weights_only forbids arbitrary pickle object execution.
    raw = torch.load(path, map_location="cpu", weights_only=True)
    meta = raw["meta"]
    if meta.get("revision") != MODEL_REVISION or meta.get("case") != case \
            or meta.get("hidden") != hidden or meta.get("rank") != rank \
            or meta.get("state_weight") != state_weight:
        raise ValueError("checkpoint identity/config mismatch")
    model = PredictiveMTP(case=case, hidden=hidden, rank=rank, state_weight=state_weight)
    model.load_state_dict(raw["state_dict"], strict=True)
    return model.to(device)
