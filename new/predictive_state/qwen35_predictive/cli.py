"""Native Qwen3.5-4B predictive-state pilot with independent case outputs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from .rollout import CASES
from .experiments import qualify, train_case, evaluate_case


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen Qwen3.5 native MTP + GDN/GQA predictive state")
    parser.add_argument("command", choices=["plan", "qualify", "train", "evaluate"])
    parser.add_argument("--case", choices=CASES, default="c00_native")
    parser.add_argument("--akr-root", default=os.environ.get("AKR_ROOT"))
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR"))
    parser.add_argument("--prompts", default="data/smoke_prompts.jsonl")
    parser.add_argument("--out", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--updates", type=int, default=32)
    parser.add_argument("--max-prompts", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--state-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.command == "plan":
        print(json.dumps({
            "model": "Qwen/Qwen3.5-4B",
            "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "trainable": "small MTP readout adapter (plus C04 state head), never backbone or MTP",
            "cases": {
                "c00_native": "released recursive native MTP, no new parameters",
                "c01_gdn": "native MTP + original GDN mixer readout",
                "c02_gqa": "native MTP + original GQA mixer readout",
                "c03_fusion": "native MTP + mean-normalized GDN/GQA readouts",
                "c04_state": "C03 + TRUE GDN recurrent matrix future-sketch auxiliary",
            },
            "horizon": args.horizon,
            "qualification": "required on actual CUDA host; no results yet",
        }, indent=2))
        return
    if not args.akr_root or not args.model_dir or not args.out:
        parser.error("--akr-root (AKR_ROOT), --model-dir (MODEL_DIR), --out are required")
    from .backend import PinnedQwenBackend
    backend = PinnedQwenBackend.load(akr_root=args.akr_root,
                                    model_dir=args.model_dir,
                                    device=args.device, dtype=args.dtype)
    from .data import load_questions
    if args.command == "qualify":
        questions = load_questions(args.prompts)
        result = qualify(backend, prompt=questions[0]["prompt"],
                         horizon=args.horizon, seed=args.seed)
        output = Path(args.out)
        if output.exists():
            raise FileExistsError("refusing to overwrite qualification receipt")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    elif args.command == "train":
        result = train_case(backend, case=args.case, prompts=args.prompts,
                            output=args.out, horizon=args.horizon,
                            updates=args.updates, seed=args.seed,
                            learning_rate=args.learning_rate, rank=args.rank,
                            state_weight=args.state_weight,
                            max_prompts=args.max_prompts)
    else:
        result = evaluate_case(backend, case=args.case, prompts=args.prompts,
                               output=args.out, checkpoint=args.checkpoint,
                               horizon=args.horizon, seed=args.seed,
                               samples=args.num_samples, rank=args.rank,
                               state_weight=args.state_weight,
                               max_prompts=args.max_prompts)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
