"""Run native training only after the pinned 4B target is locally available."""
from __future__ import annotations
import argparse
import json
import torch
from .teacher import FrozenQwenTarget
from .training import load_student, train_run, verify_one


def main():
    p = argparse.ArgumentParser(description="Predictive-State Drafter reference pilot")
    p.add_argument("command", choices=["train", "verify"])
    p.add_argument("--model-dir", required=True, help="locally checked AKR Qwen3.5-4B snapshot")
    p.add_argument("--data", required=True, help="prompt-only JSONL, not a benchmark result")
    p.add_argument("--out", required=True, help="new output for train; trained checkpoint for verify")
    p.add_argument("--mode", choices=["tf", "opd"], default="opd")
    p.add_argument("--init", help="teacher-forcing student checkpoint to initialize OPD")
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--updates", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    args = p.parse_args()
    teacher = FrozenQwenTarget.from_local(args.model_dir, device=args.device,
                                           dtype=args.dtype)
    if args.command == "train":
        result = train_run(teacher, data_path=args.data, output_dir=args.out,
                           mode=args.mode, horizon=args.horizon, updates=args.updates,
                           lr=args.lr, seed=args.seed, hidden=args.hidden,
                           n_layers=args.layers, n_heads=args.heads, initial=args.init)
    else:
        from .training import load_prompts
        prompt = load_prompts(args.data)[0]["prompt"]
        student = load_student(__import__("pathlib").Path(args.out),
                               target_vocab=teacher.vocab_size,
                               teacher_hidden=teacher.hidden_size, device=args.device)
        rng = torch.Generator(device=args.device).manual_seed(args.seed)
        result = verify_one(student, teacher, teacher.encode_prompt(prompt),
                             args.horizon, generator=rng)
        result["status"] = "reference_verification_only_not_speed_benchmark"
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
