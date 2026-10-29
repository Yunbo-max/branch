#!/usr/bin/env bash
# Single-state LM geometry ablation: H=lambda I vs H=lambda I+mu C.
# Both arms: M=1, same backbone, chunk length, data order, split, panel and budget.
set -euo pipefail

REPO=/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet
cd "$REPO/new/fair_mqar_fix"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

COMMON=(
  --packed-dir "$REPO/new/MARCH/data/packed_16k"
  --seq-len 16384 --batch-size 4 --steps 10000
  --mixer chunked --chunk-len 256 --grad-checkpoint
  --d-model 256 --layers 2 --heads 4 --d-key 64 --d-value 64 --d-route 32
  --ff-mult 4 --vocab-size 32000 --max-states 1 --fork-cost 0.0
  --covariance full --differentiate-covariance
  --split-seed 12345 --val-per-mille 20 --test-per-mille 10
  --lr 3e-4 --weight-decay 0.1
  --eval-every 500 --eval-batches 4 --checkpoint-every 5000
  --log-every 100 --threads 8 --device cuda
)

python3 pretrain_branch.py "${COMMON[@]}" --protect 0.0 \
  --output "$REPO/results/bench_geometry_lm/isotropic"
python3 pretrain_branch.py "${COMMON[@]}" --protect 8.0 \
  --output "$REPO/results/bench_geometry_lm/full"
