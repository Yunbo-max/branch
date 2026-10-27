#!/usr/bin/env bash
# Geometry isolation: full-C vs isotropic, matched architecture & seed, with
# chunk sizes that make C inactive (one chunk) vs active (shorter chunks).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1

echo "===== env ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "===== 2-pair: 64 vs 16 vs 2-token geometry chunks ====="
CHUNKS="64 16 2" OUT_ROOT="../../results/mqar_geo_isolation_2pair" \
  bash run_mqar_geometry_isolation.sh 1000 2>&1 | tee ../../results/mqar_geo_isolation_2pair_run.log

echo "===== 8-pair: 64 vs 8-token geometry chunks ====="
SEQ_LEN=128 NUM_KV_PAIRS=8 CHUNKS="64 8" OUT_ROOT="../../results/mqar_geo_isolation_8pair" \
  bash run_mqar_geometry_isolation.sh 1000 2>&1 | tee ../../results/mqar_geo_isolation_8pair_run.log

echo "===== DONE ====="
