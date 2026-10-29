#!/usr/bin/env bash
# NEW METHOD (`branch_cached`, cached SPD factorization) on native MQAR, 1xH100.
#
# Two sequential phases against the SAME generated panels and the same step count:
#   1. branch_cached --solver-refresh 8   (the new method)
#   2. branch         --solver-refresh 1  (the reference, as a quality control)
# Fixes from the first run: the training pool is raised from 8,192 to 100,000 unique
# examples, which is what the step-500+ overfitting signature called for.
set -euo pipefail

REPO=/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet
cd "$REPO/new/fair_mqar"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

COMMON="--vocab-size 8192 --seq-len 128 --num-kv-pairs 8 --num-examples 100000 --seed 17 \
  --d-model 256 --layers 2 --heads 4 --d-key 64 --d-value 64 --d-route 32 --max-states 8 \
  --covariance full --differentiate-covariance --protect 8.0 --fork-cost 0.02 \
  --batch-size 32 --lr 1e-3 --eval-every 250 --log-every 50 --eval-examples 1024 \
  --threads 8 --device cuda"

python3 - <<'PY'
import sys, torch
print(f"torch {torch.__version__} | cuda: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("no CUDA device visible; refusing to run on CPU")
print("device:", torch.cuda.get_device_name(0))
PY

echo "########## PHASE 1: new method (cached factor, refresh=8) ##########"
python3 run_branch_only.py --mixer branch_cached --solver-refresh 8 $COMMON \
  --budget-seconds 12600 --warmup-steps 25 --target-steps 6000 \
  --output-root "$REPO/results/newmethod_mqar"

echo "########## PHASE 2: reference (refresh=1) ##########"
python3 run_branch_only.py --mixer branch --solver-refresh 1 $COMMON \
  --budget-seconds 4200 --warmup-steps 5 --target-steps 2000 \
  --output-root "$REPO/results/newmethod_mqar_reference"

echo "########## DONE ##########"
