#!/usr/bin/env bash
# Pilot: three arms at default budget (120 steps, 120 train groups).
set -euo pipefail
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet/new/gdn_update_training
export MODEL=/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet/models/puigde_360M

echo "===== [1/3] sft (baseline) ====="
MODE=sft RUN_TAG=pilot bash run.sh

echo "===== [2/3] consistency (method A) ====="
MODE=consistency RUN_TAG=pilot bash run.sh

echo "===== [3/3] write_distill (method B) ====="
MODE=write_distill RUN_TAG=pilot bash run.sh

echo "===== ALL THREE ARMS DONE ====="
