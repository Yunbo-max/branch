#!/usr/bin/env bash
# First-round native GPU smoke: 2 steps per arm, matched data, one layer/head for B.
set -euo pipefail
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet/new/gdn_update_training
export MODEL=/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet/models/puigde_360M

echo "===== [1/3] sft ====="
MODE=sft STEPS=2 TRAIN_GROUPS=12 EVAL_GROUPS=12 RUN_TAG=smoke bash run.sh

echo "===== [2/3] consistency (method A) ====="
MODE=consistency STEPS=2 TRAIN_GROUPS=12 EVAL_GROUPS=12 RUN_TAG=smoke bash run.sh

echo "===== [3/3] write_distill (method B) ====="
MODE=write_distill STEPS=2 TRAIN_GROUPS=12 EVAL_GROUPS=12 TEACHER_STEPS=1 RUN_TAG=smoke bash run.sh

echo "===== ALL THREE ARMS DONE ====="
