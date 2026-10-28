#!/usr/bin/env bash
# Fair online-MQAR comparison including the real Gated DeltaNet baseline.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1

echo "===== env ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

ARMS="delta gated_delta single_geo transformer" \
EVAL_EXAMPLES=512 \
EVAL_EVERY=250 \
OUTPUT_DIR="../../results/mqar_online_gdn_2k" \
bash run_mqar_online.sh 2000 2>&1 | tee ../../results/mqar_online_gdn_2k_run.log

echo "===== DONE ====="
