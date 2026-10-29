#!/usr/bin/env bash
# Pretraining pilot on the packed 16K corpus, chunked Branch Memory, 1xH100.
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
print("device:", torch.cuda.get_device_name(0))
PY

python3 pretrain_branch.py \
  --packed-dir "$REPO/new/MARCH/data/packed_16k" \
  --seq-len 16384 \
  --batch-size 4 \
  --steps 5000 \
  --mixer chunked \
  --chunk-len 128 \
  --d-model 256 \
  --layers 2 \
  --heads 4 \
  --d-key 64 \
  --d-value 64 \
  --d-route 32 \
  --ff-mult 4 \
  --vocab-size 32000 \
  --max-states 8 \
  --protect 8.0 \
  --differentiate-covariance \
  --lr 3e-4 \
  --weight-decay 0.1 \
  --eval-every 100 \
  --eval-batches 2 \
  --log-every 10 \
  --threads 8 \
  --device cuda \
  --time-budget-seconds 14400 \
  --output "$REPO/results/pretrain_chunked_16k"
