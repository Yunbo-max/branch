#!/usr/bin/env bash
# Run ONE arm. Never queues paid jobs; invoke inside an allocated GPU environment.
set -euo pipefail
cd "$(dirname "$0")"
MODE=${MODE:-consistency}
ACTION=${ACTION:-train}
MODEL=${MODEL:-puigde/gated-deltanet-360M-15B-slimpajama}
SEED=${SEED:-42}
RUN_TAG=${RUN_TAG:-pilot}
OUT_ROOT=${OUT_ROOT:-../../results/gdn_update_learning}
python train.py "$ACTION" --mode "$MODE" --model "$MODEL" \
  --out "$OUT_ROOT/$RUN_TAG/${MODE}_seed${SEED}" --seed "$SEED" \
  --steps "${STEPS:-120}" --train-groups "${TRAIN_GROUPS:-120}" \
  --eval-groups "${EVAL_GROUPS:-24}" --length "${LENGTH:-1024}" \
  --gap "${GAP:-256}" --layer "${LAYER:-10}" --head "${HEAD:-0}" \
  --teacher-steps "${TEACHER_STEPS:-4}" "$@"
