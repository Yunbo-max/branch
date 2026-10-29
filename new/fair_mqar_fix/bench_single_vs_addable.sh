#!/usr/bin/env bash
# Run each benchmark arm in its own process so CUDA memory is measured cleanly.
set -euo pipefail

REPO=/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet
cd "$REPO/new/fair_mqar_fix"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

OUT="$REPO/results/bench_single_vs_addable"
rm -rf "$OUT"

COMMON=(
  --packed-dir "$REPO/new/MARCH/data/packed_16k"
  --seq-len 16384 --batch-size 4 --steps 50 --eval-batches 4
  --d-model 256 --layers 2 --heads 4 --d-key 64 --d-value 64 --d-route 32
  --chunk-len 256 --single-states 1 --addable-states 64
  --fork-cost 0.02 --protect 8.0 --lr 3e-4 --seed 17
  --split-seed 12345 --val-per-mille 20 --test-per-mille 10
  --device cuda --output "$OUT"
)

python3 bench_single_vs_addable.py "${COMMON[@]}" --arm single
python3 bench_single_vs_addable.py "${COMMON[@]}" --arm addable

python3 - <<PY
import json
from pathlib import Path
out = Path("$OUT")
s = json.loads((out / "single.json").read_text())
a = json.loads((out / "addable.json").read_text())
bench = {
    "config": s["config"],
    "panel_sha256": s["panel_sha256"],
    "single": s["single"],
    "addable": a["addable"],
}
(out / "bench.json").write_text(json.dumps(bench, indent=2))
print(json.dumps(bench, indent=2))
PY
