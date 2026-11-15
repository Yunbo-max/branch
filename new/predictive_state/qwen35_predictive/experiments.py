"""Isolated C00–C04 native-Qwen experiments, honest status and provenance."""
from __future__ import annotations
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from .backend import MODEL_ID, MODEL_REVISION
from .data import load_questions, shuffled_questions, run_seed, source_sha256
from .rollout import CASES, PredictiveMTP
from .checkpoint import save_small, load_small
from .objectives import forward_kl, score_candidate


def _mkdir_new(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _write_json(path: Path, payload: dict):
    with path.open("x", encoding="utf-8") as writer:
        json.dump(payload, writer, indent=2, ensure_ascii=False, allow_nan=False)
        writer.write("\n")


def _generator(device, seed):
    return torch.Generator(device=device).manual_seed(int(seed))


def _seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_case(case: str, *, hidden: int = 2560, rank: int = 128,
                   state_weight: float = 0.1) -> PredictiveMTP:
    return PredictiveMTP(case=case, hidden=hidden, rank=rank,
                         state_weight=state_weight)


def qualify(backend, *, prompt: str, horizon: int, seed: int):
    if horizon < 2:
        raise ValueError("MTP qualification needs at least two proposed tokens")
    prefix = backend.encode(prompt)
    boundary, gdn, gqa, state = backend.boundary(prefix, collect_state=True)
    model = configure_case("c00_native", hidden=backend.hidden).to(backend.device)
    with torch.no_grad():
        draft = model.sample(backend, boundary, gdn, gqa,
                             horizon=horizon, generator=_generator(backend.device, seed))
        replay = model.replay(backend, boundary, gdn, gqa, draft.ids)
        target, future_state = backend.teacher(prefix, draft.ids, collect_end_state=True)
        first_student = draft.logits[:, 0, :]
        first_target = target[:, 0, :]
        mse = (first_student.float() - first_target.float()).abs().max()
        state_change = future_state.float().sub(state.float()).square().mean().sqrt()
        sampling_error = (draft.q_probs - torch.softmax(replay.logits.float(), -1)).abs().max()
        if not torch.allclose(draft.logits, replay.logits, rtol=2e-3, atol=2e-3):
            raise RuntimeError("native MTP sample/replay logits mismatch")
        if not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(state_change)):
            raise FloatingPointError("native target/state produced nonfinite tensors")
        # A concrete threshold on this one GPU is deliberately not a training metric.
        if float(sampling_error) > 1e-5:
            raise RuntimeError("native MTP rollout/replay proposal probabilities differ")
        # The main head should predict the same first-token distribution.
        if float(mse) > 0.08:
            raise RuntimeError(f"cached/native first token logits not aligned: {float(mse)}")
    return {"status": "software_qualification_only", "prompt_length": prefix.shape[1],
            "draft_horizon": horizon, "state_shape": list(state.shape),
            "first_position_max_logit_error": float(mse),
            "sample_replay_max_prob_error": float(sampling_error),
            "true_gdn_state_delta_rms": float(state_change),
            "gdn_layer": 26, "gqa_layer": 27,
            "native_mtp_pretrained": True, "training_updates": 0,
            "measured_tokens_per_second": None}


def train_case(backend, *, case: str, prompts: str, output: str, horizon: int,
               updates: int, seed: int, learning_rate: float, rank: int,
               state_weight: float, max_prompts: int):
    if case not in CASES or case == "c00_native":
        raise ValueError("train only c01_gdn .. c04_state; c00 is frozen control")
    if horizon < 2 or updates < 1 or not 0 < learning_rate < 1 or max_prompts < 1:
        raise ValueError("invalid bounded pilot settings")
    _seed_everything(seed)
    questions = shuffled_questions(load_questions(prompts), seed)[:max_prompts]
    model = configure_case(case, hidden=backend.hidden, rank=rank, state_weight=state_weight).to(backend.device).train()
    optim = torch.optim.AdamW(model.trainable_parameters(), lr=learning_rate, weight_decay=0.0)
    root = _mkdir_new(output)
    identity = {"status": "running", "case": case, "seed": seed,
                "teacher": MODEL_ID, "revision": MODEL_REVISION,
                "horizon": horizon, "max_prompts": max_prompts,
                "updates_requested": updates, "learning_rate": learning_rate,
                "rank": rank, "state_weight": state_weight,
                "questions_sha256": source_sha256(prompts),
                "model_provenance": backend.native.provenance,
                "gpu_qualified": "requires local qualification before conclusions"}
    _write_json(root / "run.json", identity)
    started = time.perf_counter()
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backend.device)
    events = []
    for step in range(updates):
        item = questions[step % len(questions)]
        prefix = backend.encode(item["prompt"])
        need_state = (case == "c04_state")
        with torch.no_grad():
            boundary, gdn, gqa, state = backend.boundary(prefix, collect_state=need_state)
            draft = model.sample(backend, boundary, gdn, gqa,
                                 horizon=horizon, generator=_generator(backend.device,
                                     run_seed(seed, item["id"], step)))
            target, end_state = backend.teacher(prefix, draft.ids,
                                                collect_end_state=need_state)
        replay = model.replay(backend, boundary, gdn, gqa, draft.ids)
        loss_kl = forward_kl(replay.logits, target[:, :horizon, :])
        state_loss = None
        diagnostics = {}
        if need_state:
            state_loss, diagnostics = model.state_loss(replay, state, end_state)
        loss = loss_kl if state_loss is None else loss_kl + state_weight * state_loss
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError("nonfinite loss")
        optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 1.0,
                                                   error_if_nonfinite=True)
        if not bool(torch.isfinite(grad_norm.detach())):
            raise FloatingPointError("nonfinite adapter/state gradients")
        optim.step()
        row = {"step": step + 1, "id": item["id"], "prefix_tokens": prefix.shape[1],
               "kl": float(loss_kl.detach()), "loss": float(loss.detach()),
               "grad_norm": float(grad_norm.detach()),
               "state_loss": float(state_loss.detach()) if state_loss is not None else None,
               **diagnostics}
        events.append(row)
        with (root / "events.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    save_small(root / "adapter.pt", model, metadata={
        "rank": rank, "seed": seed, "updates": updates,
        "horizon": horizon, "source_sha256": source_sha256(prompts)})
    summary = {"status": "completed_pilot_training_not_benchmark",
               "case": case, "updates": len(events),
               "final_kl": events[-1]["kl"], "final_loss": events[-1]["loss"],
               "trainable_parameters": sum(p.numel() for p in model.trainable_parameters()),
               "elapsed_seconds": time.perf_counter() - started,
               "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(backend.device))
                                    if backend.device.type == "cuda" else None,
               "target": MODEL_ID, "native_mtp_frozen": True,
               "no_speedup_claim": True}
    _write_json(root / "summary.json", summary)
    return summary


@torch.no_grad()
def evaluate_case(backend, *, case: str, prompts: str, output: str, checkpoint: str | None,
                  horizon: int, seed: int, samples: int, rank: int,
                  state_weight: float, max_prompts: int):
    if case not in CASES or horizon < 2 or samples < 1 or max_prompts < 1:
        raise ValueError("invalid case/eval protocol")
    if case == "c00_native":
        if checkpoint:
            raise ValueError("C00 native baseline must not load adapter checkpoint")
        model = configure_case(case, hidden=backend.hidden).to(backend.device)
    else:
        if not checkpoint:
            raise ValueError("trained case requires explicit completed adapter.pt")
        model = load_small(checkpoint, case=case, hidden=backend.hidden, rank=rank,
                           state_weight=state_weight, device=str(backend.device))
    model.eval()
    questions = shuffled_questions(load_questions(prompts), seed)[:max_prompts]
    root = _mkdir_new(output)
    spec = {"case": case, "status": "reference_acceptance_only",
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "horizon": horizon, "seed": seed, "samples": samples, "max_prompts": max_prompts,
            "question_sha256": source_sha256(prompts),
            "checkpoint": checkpoint,
            "teacher_verification": "full_prefix_recompute_not_cached"}
    _write_json(root / "config.json", spec)
    from psdraft.speculative import verify_block
    rows = []
    started = time.perf_counter()
    for item in questions:
        prefix = backend.encode(item["prompt"])
        need_state = case == "c04_state"
        boundary, gdn, gqa, state = backend.boundary(prefix, collect_state=need_state)
        for j in range(samples):
            rng = _generator(backend.device, run_seed(seed, item["id"], j))
            draft = model.sample(backend, boundary, gdn, gqa, horizon=horizon,
                                 generator=rng)
            target, end_state = backend.teacher(prefix, draft.ids,
                                                collect_end_state=need_state)
            p = torch.softmax(target[0].float(), dim=-1)
            result = verify_block(draft.ids[0], draft.q_probs[0], p, generator=rng)
            diag = score_candidate(draft.logits, target[:, :horizon, :])
            row = {"prompt_id": item["id"], "sample": j, "accepted": result.accepted,
                   "emitted": len(result.emitted), "rejected_at": result.rejected_at,
                   **diag}
            if need_state:
                value, aux = model.state_loss(draft, state, end_state)
                row["state_jump_loss"] = float(value)
                row.update(aux)
            rows.append(row)
            with (root / "rows.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    def mean(name):
        return sum(row[name] for row in rows) / len(rows)
    result = {"status": "completed_reference_acceptance_not_speed_or_accuracy",
              "case": case, "num_prompts": len(questions), "samples": samples,
              "horizon": horizon, "mean_accepted": mean("accepted"),
              "mean_emitted": mean("emitted"),
              "full_accept_fraction": sum(x["accepted"] == horizon for x in rows) / len(rows),
              "mean_teacher_to_student_kl": mean("teacher_to_student_kl"),
              "mean_tv": mean("mean_tv"), "mean_state_jump_loss": mean("state_jump_loss")
                   if need_state else None,
              "elapsed_reference_seconds": time.perf_counter() - started,
              "true_cache_aware_tokens_per_second": None, "target_task_accuracy": None}
    _write_json(root / "summary.json", result)
    return result
