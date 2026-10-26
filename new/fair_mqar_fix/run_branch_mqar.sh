#!/usr/bin/env bash
# Branch Memory on native MQAR — single arm, single seed, 1xH100.
#
# Phase 1 (warmup) validates the GPU environment and measures seconds/step.
# Phase 2 sizes the real run from the remaining wall-clock budget.
set -euo pipefail

REPO=/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv
cd "$REPO/new/fair_mqar"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

python3 - <<'PY'
import sys, torch
print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("no CUDA device visible; refusing to run on CPU")
print("device:", torch.cuda.get_device_name(0), "| memory GB:",
      round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
PY

python3 run_branch_only.py \
  --vocab-size 8192 \
  --seq-len 128 \
  --num-kv-pairs 8 \
  --num-examples 8192 \
  --seed 17 \
  --d-model 256 \
  --layers 2 \
  --heads 4 \
  --d-key 64 \
  --d-value 64 \
  --d-route 32 \
  --max-states 8 \
  --covariance full \
  --differentiate-covariance \
  --protect 8.0 \
  --fork-cost 0.02 \
  --batch-size 32 \
  --lr 1e-3 \
  --eval-every 500 \
  --log-every 500 \
  --threads 8 \
  --device cuda \
  --budget-seconds 16200 \
  --warmup-steps 25 \
  --target-steps 40000 \
  --output-root "$REPO/results/branch_mqar"
