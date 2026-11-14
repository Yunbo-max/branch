"""Reference two-stage teacher-forcing / student-rollout OPD training.

One sample/step to expose alignment and budget. No RL, no verifier-state adapter,
no MTP future head, and no GPU throughput claim. A student rollout uses exactly
the same frozen parameters as the later differentiable teacher-forced replay.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from .losses import forward_kl, teacher_forcing_loss
from .model import DrafterConfig, PredictiveStateDrafter
from .speculative import verify_block


def load_prompts(path: str | Path) -> list[dict]:
    """Gold answers never enter prompt rendering, reference selection or training."""
    items = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                raise ValueError("JSONL rows require a nonempty 'prompt'")
            identity = str(row.get("id", len(items)))
            if identity in seen:
                raise ValueError("duplicate prompt ID")
            seen.add(identity)
            items.append({"id": identity, "prompt": row["prompt"]})
    if not items:
        raise ValueError("no prompts found")
    return items


def optimize_one(drafter: PredictiveStateDrafter, teacher, prefix: torch.Tensor,
                 optimizer: torch.optim.Optimizer, *, mode: str, horizon: int):
    if mode not in {"tf", "opd"} or horizon < 1:
        raise ValueError("mode must be tf or opd, horizon >= 1")
    drafter.train()
    with torch.no_grad():
        boundary, _ = teacher.boundary_and_logits(prefix)
        if mode == "tf":
            drafted = teacher.greedy(prefix, horizon)
        else:
            drafted, _ = drafter.sample(boundary, horizon)
        boundary, teacher_logits = teacher.boundary_and_logits(prefix, drafted)
    student_logits = drafter.logits_on_draft(boundary, drafted)
    if mode == "tf":
        loss = teacher_forcing_loss(student_logits, drafted)
    else:
        loss = forward_kl(student_logits, teacher_logits[:, :drafted.shape[1], :])
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError("nonfinite training objective")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(drafter.parameters(), 1.0, error_if_nonfinite=True)
    if not bool(torch.isfinite(grad.detach())):
        raise FloatingPointError("nonfinite student gradient")
    optimizer.step()
    return {"loss": float(loss.detach()), "grad_norm": float(grad.detach()),
            "draft_tokens": int(drafted.shape[1]), "mode": mode}


@torch.no_grad()
def verify_one(drafter: PredictiveStateDrafter, teacher,
               prefix: torch.Tensor, horizon: int, *, generator=None):
    drafter.eval()
    boundary, _ = teacher.boundary_and_logits(prefix)
    drafted, q = drafter.sample(boundary, horizon, generator=generator)
    _, target_logits = teacher.boundary_and_logits(prefix, drafted)
    p = torch.softmax(target_logits[0].float(), dim=-1)
    result = verify_block(drafted[0], q[0], p, generator=generator)
    return {"draft": drafted[0].tolist(), "emitted": result.emitted,
            "accepted": result.accepted, "rejected_at": result.rejected_at}


def save_student(path: Path, drafter: PredictiveStateDrafter, *, metadata: dict):
    path.mkdir(parents=True, exist_ok=True)
    tmp = path / ".model.pt.tmp"
    torch.save(drafter.state_dict(), tmp)
    os.replace(tmp, path / "model.pt")
    (path / "metadata.json").write_text(json.dumps({"config": drafter.export_config(),
                                                      **metadata}, indent=2) + "\n")


def load_student(path: Path, *, target_vocab: int, teacher_hidden: int, device: str):
    spec = json.loads((path / "metadata.json").read_text())["config"]
    cfg = DrafterConfig(**spec)
    if cfg.vocab_size != target_vocab or cfg.teacher_hidden_size != teacher_hidden:
        raise ValueError("checkpoint is incompatible with teacher/tokenizer")
    model = PredictiveStateDrafter(cfg).to(device)
    model.load_state_dict(torch.load(path / "model.pt", map_location=device,
                                     weights_only=True), strict=True)
    return model


def train_run(teacher, *, data_path: str, output_dir: str, mode: str,
              horizon: int, updates: int, lr: float, seed: int, hidden: int,
              n_layers: int, n_heads: int, initial: str | None = None):
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("refusing to overwrite a run directory")
    out.mkdir(parents=True, exist_ok=True)
    if updates < 1 or lr <= 0 or horizon < 1:
        raise ValueError("updates, lr and horizon must be positive")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    examples = load_prompts(data_path)
    # Seeded independent order; do not always train on the first dataset rows.
    sampler = random.Random(seed)
    sampler.shuffle(examples)
    if initial is None:
        if hidden % n_heads != 0:
            raise ValueError("hidden must be divisible by n_heads")
        c = DrafterConfig(vocab_size=teacher.vocab_size,
                          teacher_hidden_size=teacher.hidden_size,
                          hidden_size=hidden, n_layers=n_layers,
                          n_heads=n_heads, head_dim=hidden // n_heads)
        student = PredictiveStateDrafter(c).to(teacher.device)
    else:
        student = load_student(Path(initial), target_vocab=teacher.vocab_size,
                               teacher_hidden=teacher.hidden_size, device=str(teacher.device))
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    source_bytes = Path(data_path).read_bytes()
    events = []
    started = time.perf_counter()
    for index in range(updates):
        if index > 0 and index % len(examples) == 0:
            sampler.shuffle(examples)
        item = examples[index % len(examples)]
        prefix = teacher.encode_prompt(item["prompt"])
        stats = optimize_one(student, teacher, prefix, optimizer, mode=mode, horizon=horizon)
        row = {"step": index + 1, "prompt_id": item["id"], **stats}
        events.append(row)
        with (out / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    receipt = {"status": "completed_reference_training", "mode": mode,
               "updates": updates, "seed": seed, "horizon": horizon,
               "teacher_id": "Qwen/Qwen3.5-4B",
               "data_sha256": hashlib.sha256(source_bytes).hexdigest(),
               "final_loss": events[-1]["loss"],
               "elapsed_seconds": time.perf_counter() - started,
               "recurrent_matrix_bytes_per_sequence_bf16": student.recurrent_state_bytes(),
               "limitations": ["pure PyTorch reference GDN, not FLA kernel",
                               "no GPU inference speed or quality result from this receipt",
                               "no learned verifier-feedback state in v0"]}
    save_student(out, student, metadata=receipt)
    return receipt
