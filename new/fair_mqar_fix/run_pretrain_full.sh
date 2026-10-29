#!/usr/bin/env bash
# Full-corpus chunked Branch Memory run, 1xH100.
#
# Probes chunk_len {1024,512,256} at the real model size, keeps the fastest that fits,
# then trains for up to 4 days with checkpoints every 10k steps.
set -euo pipefail

REPO=/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet
cd "$REPO/new/fair_mqar"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

python3 - <<'PY'
import sys, torch
print(f"torch {torch.__version__} | cuda: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("no CUDA device visible; refusing to run on CPU")
p = torch.cuda.get_device_properties(0)
print("device:", p.name, "| memory GB:", round(p.total_memory / 1e9, 1))
PY

python3 pretrain_branch.py \
  --packed-dir "$REPO/new/MARCH/data/packed_16k" \
  --seq-len 16384 \
  --batch-size 4 \
  --steps 1200000 \
  --mixer chunked \
  --chunk-len 256 \
  --grad-checkpoint \
    --d-model 256 \
  --layers 2 \
  --heads 4 \
  --d-key 64 \
  --d-value 64 \
  --d-route 32 \
  --ff-mult 4 \
  --vocab-size 32000 \
  --max-states 64 \
  --protect 8.0 \
  --covariance full \
  --fork-cost 0.02 \
  --differentiate-covariance \
  --split-seed 12345 \
  --val-per-mille 20 \
  --test-per-mille 10 \
  --lr 3e-4 \
  --weight-decay 0.1 \
  --eval-every 500 \
  --eval-batches 4 \
  --checkpoint-every 10000 \
  --log-every 50 \
  --threads 8 \
  --device cuda \
  --time-budget-seconds 345600 \
  --output "$REPO/results/pretrain_full_m64"
