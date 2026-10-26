#!/usr/bin/env bash
set -euo pipefail
REPO=/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv
cd "$REPO/new/fair_mqar"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
python3 probe_batch.py --packed-dir "$REPO/new/MARCH/data/packed_16k" \
  --seq-len 16384 --batches 4 8 16 --mixer chunked --chunk-len 256 \
  --d-model 256 --layers 2 --heads 4 --d-key 64 --d-value 64 --d-route 32 \
  --ff-mult 4 --max-states 8
