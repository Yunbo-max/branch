#!/usr/bin/env bash
# Real H100/H200 pre-trained GDN diagnostic: never trains new weights.
# STAGE=smoke|screen|full   MODEL_ID=...   CUDA_VISIBLE_DEVICES=0
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
STAGE=${STAGE:-smoke}
MODEL_ID=${MODEL_ID:-m-a-p/340M-20B-GatedDeltaNet-pure-baseline}
OUT_ROOT=${OUT_ROOT:-"$HERE/../../results/gdn_failure_census"}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-bf16}
SEED_START=${SEED_START:-1000}
DRY_RUN=${DRY_RUN:-0}
RESUME=${RESUME:-0}
PREEDIT=${PREEDIT:-1}

case "$STAGE" in
  smoke)
    LENGTHS=${LENGTHS:-512,1024,2048}
    SEEDS=${SEEDS:-3}
    POSITIONS=${POSITIONS:-0.12,0.88}
    EVENTS=${EVENTS:-8}
    CONDITIONS=${CONDITIONS:-plain,unrelated,collision,revision_latest,revision_earliest}
    ;;
  screen)
    LENGTHS=${LENGTHS:-512,1024,2048,4096,8192}
    SEEDS=${SEEDS:-20}
    POSITIONS=${POSITIONS:-0.12,0.50,0.88}
    EVENTS=${EVENTS:-8,32}
    CONDITIONS=${CONDITIONS:-plain,unrelated,collision,revision_latest,revision_earliest,alias,two_hop}
    ;;
  full)
    LENGTHS=${LENGTHS:-512,1024,2048,4096,8192,16384,32768}
    SEEDS=${SEEDS:-48}
    POSITIONS=${POSITIONS:-0.12,0.50,0.88}
    EVENTS=${EVENTS:-8,32,64}
    CONDITIONS=${CONDITIONS:-plain,unrelated,collision,revision_latest,revision_earliest,alias,two_hop}
    ;;
  *)
    echo "Unknown STAGE=$STAGE. Use smoke, screen or full." >&2
    exit 2
    ;;
esac

if [[ -z "${OUT_DIR:-}" ]]; then
  SAFE_ID="${MODEL_ID//\//__}"
  OUT_DIR="$OUT_ROOT/${SAFE_ID}_${STAGE}_s${SEED_START}_n${SEEDS}"
fi
mkdir -p "$OUT_ROOT"
echo "GDN CENSUS: model=$MODEL_ID stage=$STAGE device=$DEVICE dtype=$DTYPE output=$OUT_DIR"
echo "Length conditions=$LENGTHS; positions=$POSITIONS; events=$EVENTS; seeds=$SEEDS start=$SEED_START"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

ARGS=(
  --model "$MODEL_ID" --device "$DEVICE" --dtype "$DTYPE"
  --lengths "$LENGTHS" --positions "$POSITIONS"
  --events "$EVENTS" --conditions "$CONDITIONS"
  --seed-start "$SEED_START" --seeds "$SEEDS" --out "$OUT_DIR"
  --allow-extrapolation
)
if [[ "$PREEDIT" == "1" ]]; then ARGS+=(--preedit-probes); fi
if [[ "$DRY_RUN" == "1" ]]; then ARGS+=(--dry-run); fi
if [[ "$RESUME" == "1" ]]; then ARGS+=(--resume); fi
python3 run.py "${ARGS[@]}" 2>&1 | tee "$OUT_ROOT/$(basename "$OUT_DIR").log"
if [[ "$DRY_RUN" != "1" ]]; then
  python3 summarize.py --run "$OUT_DIR"
fi
echo "Done: $OUT_DIR"
