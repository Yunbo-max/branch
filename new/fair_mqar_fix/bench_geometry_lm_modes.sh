#!/usr/bin/env bash
# Honest single-state LM efficiency: isotropic (no C, no Cholesky) vs full geometry.
# One process per arm so peak CUDA memory is measured cleanly.
set -euo pipefail

REPO=/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet
cd "$REPO/new/fair_mqar_fix"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

OUT="$REPO/results/sg_delta/lm_efficiency_2k"
rm -rf "$OUT"

COMMON=(
  --packed-dir "$REPO/new/MARCH/data/packed_16k"
  --seq-len 16384 --batch-size 4 --steps 2000 --eval-batches 4
  --d-model 256 --layers 2 --heads 4 --d-key 64 --d-value 64 --d-route 32
  --chunk-len 256 --lr 3e-4 --weight-decay 0.1
  --split-seed 12345 --val-per-mille 20 --test-per-mille 10
  --seed 17 --device cuda --vocab-size 32000 --output "$OUT"
)

python3 bench_geometry_lm_modes.py "${COMMON[@]}" --mode isotropic
python3 bench_geometry_lm_modes.py "${COMMON[@]}" --mode full

python3 - <<PY
import json
from pathlib import Path

out = Path("$OUT")
iso = json.loads((out / "isotropic.json").read_text())
full = json.loads((out / "full.json").read_text())
bench = {"panel_sha256": iso["panel_sha256"], "isotropic": iso, "full": full}
(out / "bench.json").write_text(json.dumps(bench, indent=2))
summary = {
    "isotropic": {k: iso[k] for k in ("seconds_per_step", "peak_allocated_gb", "train_loss_last100")},
    "full": {k: full[k] for k in ("seconds_per_step", "peak_allocated_gb", "train_loss_last100")},
    "eval_isotropic": iso["eval"],
    "eval_full": full["eval"],
}
print(json.dumps(summary, indent=2))
PY
