#!/usr/bin/env bash
# Fresh online MQAR: every optimizer step gets new KV associations.
# Tests whether recurrent models learn the algorithm rather than memorize a fixed dataset.
# Usage: bash new/fair_mqar_fix/run_mqar_online.sh [steps]
set -euo pipefail

STEPS=${1:-2000}
ARMS=(${ARMS:-delta single_geo transformer})
DEVICE=${DEVICE:-cuda}
EVAL_EXAMPLES=${EVAL_EXAMPLES:-512}
EVAL_EVERY=${EVAL_EVERY:-250}
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
SEED=${SEED:-17}
OUTPUT_DIR=${OUTPUT_DIR:-}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$HERE/../../results/mqar_online_$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "$OUTPUT_DIR"
echo "online MQAR: steps=$STEPS seed=$SEED eval_examples=$EVAL_EXAMPLES batch=$BATCH_SIZE output=$OUTPUT_DIR"
for ARM in "${ARMS[@]}"; do
  echo "===== $ARM ====="
  python3 train_fair.py \
    --online-train --online-seed-offset 3000001 \
    --mixer "$ARM" --device "$DEVICE" \
    --vocab-size 8192 --seq-len 128 --num-kv-pairs 8 \
    --num-examples 8192 --seed "$SEED" \
    --d-model 256 --layers 2 --heads 4 --d-key 64 --d-value 64 \
    --steps "$STEPS" --batch-size "$BATCH_SIZE" --lr "$LEARNING_RATE" \
    --weight-decay 0.01 --eval-every "$EVAL_EVERY" \
    --eval-examples "$EVAL_EXAMPLES" --log-every "$EVAL_EVERY" \
    --threads 8 --output "$OUTPUT_DIR/$ARM" \
    2>&1 | tee "$OUTPUT_DIR/$ARM.log"
done

python3 - "$OUTPUT_DIR" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
print("\nOnline MQAR final summary (validation/test are held-out)")
print(f"{'model':15} {'valid_acc':>11} {'test_acc':>10} {'valid_ce':>11} {'train_last':>11} {'wall_s':>9}")
for d in sorted(root.iterdir()):
    report = d / "report.json"
    if not report.exists():
        continue
    j = json.loads(report.read_text())
    f = j["final"]
    last = j["history_tail"][-1]
    print(f"{j['mixer']:15} {f['valid']['accuracy']:11.4f} {f['test']['accuracy']:10.4f} "
          f"{f['valid']['loss']:11.3f} {last['train_accuracy']:11.4f} "
          f"{j['training']['wall_seconds']:9.1f}")
PY
echo "Saved to $OUTPUT_DIR"
