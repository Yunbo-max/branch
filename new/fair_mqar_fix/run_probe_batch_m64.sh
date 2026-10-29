#!/usr/bin/env bash
# Memory probe for M=64 Branch Memory on 1xH100.
#
# M=64 was never actually measured: earlier m64 jobs died in argparse before
# touching CUDA. This probe uses the same checkpointed path as training and
# reports peak allocated memory for batch 1/2/4 at chunk 256.
set -euo pipefail

REPO=/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet
cd "$REPO/new/fair_mqar"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

python3 probe_batch.py --packed-dir "$REPO/new/MARCH/data/packed_16k" \
  --seq-len 16384 --batches 1 2 4 --mixer chunked --chunk-len 256 \
  --grad-checkpoint \
  --d-model 256 --layers 2 --heads 4 --d-key 64 --d-value 64 --d-route 32 \
  --ff-mult 4 --max-states 64
