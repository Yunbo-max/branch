#!/usr/bin/env bash
# 0.79B-scale arm runner (21 layers, hidden 1536, 6 heads, seq 16K, 8xH200).
# Usage: bash run_0p79b.sh <arm> [steps]
set -euo pipefail

ARM=${1:?arm: march | single_full | fork | isotropic | rayleigh | delta_chunked | transformer25}
STEPS=${2:-11921}
REPO=/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv
cd "$REPO/new/fair_mqar_fix"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
unset CUDA_VISIBLE_DEVICES

COMMON=(
  --packed-dir "$REPO/new/MARCH/data/packed_16k"
  --seq-len 16384 --batch-size 1 --grad-accum 32 --steps "$STEPS"
  --d-model 1536 --layers 21 --heads 6 --d-key 256 --d-value 256
  --d-route 64 --ff-mult 4 --vocab-size 32000
  --chunk-len 256 --grad-checkpoint
  --lr 4e-4 --weight-decay 0.1
  --lr-schedule wsd --warmup-frac 0.01 --decay-frac 0.2 --min-lr-ratio 0.1
  --split-seed 12345 --val-per-mille 20 --test-per-mille 10
  --eval-every 500 --eval-batches 4 --checkpoint-every 500
  --keep-checkpoints last --log-every 10 --threads 8 --device cuda
  --amp none --time-budget-seconds 345600
)

case "$ARM" in
  march)
    EXTRA=(--mixer march_chunked --anchor-interval 512 --read-topk 4 --max-states 64 --ff-extra 1090)
    ;;
  single_full)
    EXTRA=(--mixer single_geo --geometry-mode full --max-states 1 --ff-extra 1285)
    ;;
  fork)
    EXTRA=(--mixer chunked --max-states 64 --ff-extra 1285)
    ;;
  isotropic)
    EXTRA=(--mixer single_geo --geometry-mode isotropic --max-states 1 --ff-extra 1285)
    ;;
  rayleigh)
    EXTRA=(--mixer single_geo --geometry-mode rayleigh --max-states 1 --ff-extra 1285)
    ;;
  delta_chunked)
    EXTRA=(--mixer single_geo --geometry-mode isotropic --read-mode current_only --max-states 1 --ff-extra 1285)
    ;;
  transformer25)
    EXTRA=(--mixer transformer --layers 25 --ff-extra -177)
    ;;
  *)
    echo "unknown arm: $ARM" >&2
    exit 2
    ;;
esac

torchrun --standalone --nproc_per_node=8 pretrain_branch.py \
  "${COMMON[@]}" "${EXTRA[@]}" \
  --output "$REPO/results/scale_0p79b/$ARM"
