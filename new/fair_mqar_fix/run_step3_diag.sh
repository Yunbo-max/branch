#!/usr/bin/env bash
# Step 3 diagnostics: (a) 2-pair easy MQAR, (b) single_geo with gradient through C.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1

echo "===== env ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "===== (a) easy: 2 pairs, seq 64 ====="
NUM_KV_PAIRS=2 SEQ_LEN=64 ARMS="delta single_geo transformer" \
  EVAL_EXAMPLES=512 EVAL_EVERY=250 \
  OUTPUT_DIR="../../results/mqar_online_easy" \
  bash run_mqar_online.sh 2000 2>&1 | tee ../../results/mqar_online_easy_run.log

echo "===== (b) grad-through-C: single_geo, 8 pairs ====="
ARMS="single_geo" DIFFERENTIATE_COVARIANCE=1 \
  EVAL_EXAMPLES=512 EVAL_EVERY=250 \
  OUTPUT_DIR="../../results/mqar_online_gradC" \
  bash run_mqar_online.sh 2000 2>&1 | tee ../../results/mqar_online_gradC_run.log

echo "===== DONE ====="
