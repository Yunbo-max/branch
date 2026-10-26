#!/usr/bin/env bash
# GPU screening run: four arms x three seeds on native MQAR, one H100.
#
# Identical backbone, data panels, optimiser, budget and matched parameters across arms.
# Incremental output: results/fair_gpu_screen/sweep_summary.json is rewritten after every
# (arm, seed) so a killed job keeps completed runs.
set -euo pipefail

REPO=/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv
cd "$REPO/new/fair_mqar"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

python3 - <<'PY'
import torch, sys
print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("no CUDA device visible; refusing to run the GPU screen on CPU")
print("device:", torch.cuda.get_device_name(0))
PY

python3 sweep.py \
  --arms march branch delta transformer \
  --seeds 17 23 29 \
  --steps 1500 \
  --batch-size 32 \
  --num-examples 8192 \
  --seq-len 128 \
  --num-kv-pairs 8 \
  --vocab-size 8192 \
  --d-model 256 \
  --layers 2 \
  --heads 4 \
  --d-key 64 \
  --d-value 64 \
  --d-route 32 \
  --anchor-interval 16 \
  --max-anchors 8 \
  --read-topk 4 \
  --max-states 8 \
  --differentiate-covariance \
  --lr 1e-3 \
  --eval-every 250 \
  --log-every 250 \
  --threads 8 \
  --device cuda \
  --time-budget-seconds 5400 \
  --output-root "$REPO/results/fair_gpu_screen"
