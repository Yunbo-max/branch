#!/usr/bin/env python3
"""Build every model in the paper's matrix on the meta device and report parameter counts.

No GPU or real memory is needed: ``torch.device("meta")`` constructs modules without
allocating storage. Caveat: FLA's Triton kernels initialise a driver at import time, so on a
CPU-only node the four FLA/H-GDN baselines cannot be imported at all. In that case this
script still verifies MARCH and prints the baseline configs for a GPU-node run.

    python code/march/check_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
MARCH_ROOT = CODE.parent
sys.path.insert(0, str(CODE))                          # `march`
sys.path.insert(0, str(CODE / "log-linear-attention"))  # `hattention`

import march  # noqa: F401,E402  (registers model_type "march")

FLA_OK = True
FLA_ERR = None
try:
    import fla.models  # noqa: F401
except Exception as exc:  # noqa: BLE001
    FLA_OK = False
    FLA_ERR = f"{type(exc).__name__}: {exc}"

HATTENTION_OK = True
try:
    import hattention  # noqa: F401
except Exception as exc:  # noqa: BLE001
    HATTENTION_OK = False
    if FLA_ERR is None:
        FLA_ERR = f"{type(exc).__name__}: {exc}"

CONFIGS = [
    ("Transformer", "transformer.json"),
    ("Gated DeltaNet", "gated_deltanet.json"),
    ("GDN + Log-Linear", "h_gated_deltanet.json"),
    ("MARCH", "march.json"),
    ("Transformer 24L (optional)", "paper/transformer_mid.json"),
]


def count(cfg_path: Path) -> int:
    cfg = AutoConfig.from_pretrained(str(cfg_path))
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    return sum(p.numel() for p in model.parameters())


def main() -> int:
    print(f"{'model':26} {'params':>13}  config")
    print("-" * 74)
    failed = 0
    for label, name in CONFIGS:
        path = MARCH_ROOT / "configs" / name
        mt = json.loads(path.read_text()).get("model_type", "?")
        is_march = mt == "march"
        if not is_march and not FLA_OK:
            print(f"{label:26} {'(needs GPU)':>13}  {name}  [{mt}]")
            continue
        try:
            n = count(path)
            print(f"{label:26} {n/1e6:9.1f} M  {name}  [{mt}]")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"{label:26} {'FAILED':>13}  {name}: {type(exc).__name__}: {exc}")
    print()
    if not FLA_OK:
        print("note: FLA/H-GDN baselines could not be imported on this CPU-only node;")
        print(f"      Triton needs an active GPU driver ({FLA_ERR}).")
        print("      Re-run this script on the GPU node to verify all five models.")
    print("Shared backbone: hidden 1536 / 21 layers / FFN 4096 / 16K / vocab 32000.")
    print("Only the mixer differs; see configs/common.md for the fairness contract.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
