#!/usr/bin/env bash
# 0.79B GeoDelta training protocol: 4K first, 16K continuation/controlled comparison.
# H200 default: 8 GPUs. Works for other counts by overriding GPUS/GLOBAL_TOKENS.
# Usage: bash run_0p79b.sh <single_full|isotropic|rayleigh|delta_proxy|march_proxy|fork|transformer25> [total_steps]
# Dry setup check: GPUS=8 SEQ_LEN=4096 GLOBAL_TOKENS=32768 bash run_0p79b.sh single_full 3
set -euo pipefail

ARM=${1:?usage: $0 <single_full|isotropic|rayleigh|delta_proxy|march_proxy|fork|transformer25> [total_steps]}
STEPS=${2:-11921}
GPUS=${GPUS:-8}
SEQ_LEN=${SEQ_LEN:-4096}
MICRO_BATCH=${MICRO_BATCH:-1}
GLOBAL_TOKENS=${GLOBAL_TOKENS:-4194304}
CHUNK_LEN=${CHUNK_LEN:-256}
DELTA_TILE_LEN=${DELTA_TILE_LEN:-0}
DRY_RUN=${DRY_RUN:-0}
PROTECTION_DECAY=${PROTECTION_DECAY:-0.95}
AMP=${AMP:-bf16}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-500}
EVAL_EVERY=${EVAL_EVERY:-100}
TIME_BUDGET_SECONDS=${TIME_BUDGET_SECONDS:-345600}
RANDOM_CROP=${RANDOM_CROP:-1}
FUSED_ADAMW=${FUSED_ADAMW:-1}
LR=${LR:-4e-4}
REPO=${REPO:-/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv}
PACKED_DIR=${PACKED_DIR:-$REPO/new/MARCH/data/packed_16k}
RUN_TAG=${RUN_TAG:-4k_fullC}
OUT_ROOT=${OUT_ROOT:-$REPO/results/scale_0p79b/$RUN_TAG}
RESUME=${RESUME:-}

case "$SEQ_LEN" in 4096|16384) ;; *) echo "ERROR: only tested plan lengths 4096/16384" >&2; exit 2 ;; esac
case "$AMP" in bf16|none) ;; *) echo "ERROR: AMP must be bf16 or none" >&2; exit 2 ;; esac
case "$GPUS" in ''|*[!0-9]*) echo "ERROR: GPUS must be integer" >&2; exit 2 ;; esac
if (( GPUS < 1 || MICRO_BATCH < 1 || GLOBAL_TOKENS < 1 )); then
  echo "ERROR: invalid GPU/batch/token configuration" >&2; exit 2
fi
PER_MICRO=$((GPUS * MICRO_BATCH * SEQ_LEN))
if (( GLOBAL_TOKENS % PER_MICRO != 0 )); then
  echo "ERROR: GLOBAL_TOKENS must be divisible by GPUS*MICRO_BATCH*SEQ_LEN=$PER_MICRO" >&2
  exit 2
fi
GRAD_ACCUM=$((GLOBAL_TOKENS / PER_MICRO))
if (( GRAD_ACCUM < 1 )); then echo "ERROR: grad accumulation below one" >&2; exit 2; fi
if (( CHUNK_LEN < 1 || SEQ_LEN % CHUNK_LEN != 0 )); then
  echo "ERROR: CHUNK_LEN must divide SEQ_LEN" >&2; exit 2
fi
if (( DELTA_TILE_LEN < 0 || (DELTA_TILE_LEN > 0 && CHUNK_LEN % DELTA_TILE_LEN != 0) )); then
  echo "ERROR: DELTA_TILE_LEN must divide CHUNK_LEN (or be 0)" >&2; exit 2
fi
if [[ "$DRY_RUN" != 1 && ! -d "$PACKED_DIR" ]]; then
  echo "ERROR: packed dataset does not exist: $PACKED_DIR" >&2; exit 2
fi

cd "$REPO/new/fair_mqar_fix"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HUB_OFFLINE=1

COMMON=(
  --packed-dir "$PACKED_DIR"
  --seq-len "$SEQ_LEN" --batch-size "$MICRO_BATCH" --grad-accum "$GRAD_ACCUM" --steps "$STEPS"
  --d-model 1536 --layers 21 --heads 6 --d-key 256 --d-value 256
  --d-route 64 --ff-mult 4 --vocab-size 32000
  --chunk-len "$CHUNK_LEN" --delta-tile-len "$DELTA_TILE_LEN" --grad-checkpoint
  --protection-decay "$PROTECTION_DECAY"
  --lr "$LR" --weight-decay 0.1
  --lr-schedule wsd --warmup-frac 0.01 --decay-frac 0.2 --min-lr-ratio 0.1
  --split-seed 12345 --val-per-mille 20 --test-per-mille 10
  --eval-every "$EVAL_EVERY" --eval-batches 4
  --checkpoint-every "$CHECKPOINT_EVERY" --keep-checkpoints last
  --log-every 10 --threads 8 --device cuda
  --amp "$AMP" --time-budget-seconds "$TIME_BUDGET_SECONDS"
)
if [[ "$RANDOM_CROP" == 1 ]]; then COMMON+=(--random-crop); fi
if [[ "$FUSED_ADAMW" == 1 ]]; then COMMON+=(--fused-adamw); fi
if [[ -n "$RESUME" ]]; then COMMON+=(--resume "$RESUME"); fi

case "$ARM" in
  single_full)
    EXTRA=(--mixer single_geo --geometry-mode full --max-states 1 --ff-extra 1280)
    ;;
  isotropic)
    EXTRA=(--mixer single_geo --geometry-mode isotropic --max-states 1 --ff-extra 1280)
    ;;
  rayleigh)
    EXTRA=(--mixer single_geo --geometry-mode rayleigh --max-states 1 --ff-extra 1280)
    ;;
  delta_proxy|delta_chunked)
    # NOT genuine Gated DeltaNet: fixed isotropic Delta recurrence with no anchor read.
    EXTRA=(--mixer single_geo --geometry-mode isotropic --read-mode current_only --max-states 1 --ff-extra 1280)
    ;;
  march_proxy|march)
    # Proxy uses one anchor per CHUNK_LEN, independent of --anchor-interval.
    # Do not call this an official GDN-based MARCH reproduction.
    EXTRA=(--mixer march_chunked --anchor-interval "$CHUNK_LEN" --read-topk 4 --max-states 64 --ff-extra 1088)
    ;;
  fork)
    EXTRA=(--mixer chunked --max-states 64 --ff-extra 1280)
    ;;
  transformer25)
    EXTRA=(--mixer transformer --layers 25 --ff-extra -192)
    ;;
  *)
    echo "ERROR: unknown arm $ARM" >&2; exit 2
    ;;
esac

mkdir -p "$OUT_ROOT"
OUT_DIR="$OUT_ROOT/$ARM"
echo "0.79B TRAIN: arm=$ARM gpus=$GPUS seq_len=$SEQ_LEN global_tokens/step=$GLOBAL_TOKENS grad_accum=$GRAD_ACCUM amp=$AMP chunk=$CHUNK_LEN tile=$DELTA_TILE_LEN gamma=$PROTECTION_DECAY random_crop=$RANDOM_CROP output=$OUT_DIR"
echo "NOTE: this launcher uses proxy MARCH/Delta rather than official fused GDN/MARCH."
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'torchrun --standalone --nproc_per_node=%q pretrain_branch.py ' "$GPUS"
  printf '%q ' "${COMMON[@]}" "${EXTRA[@]}" --output "$OUT_DIR"
  echo
  exit 0
fi
torchrun --standalone --nproc_per_node="$GPUS" pretrain_branch.py \
  "${COMMON[@]}" "${EXTRA[@]}" \
  --output "$OUT_DIR"
