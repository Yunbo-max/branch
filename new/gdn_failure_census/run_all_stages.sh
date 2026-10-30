#!/usr/bin/env bash
# Run all GDN census stages sequentially on one card: smoke -> competence -> screen.
set -euo pipefail
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet/new/gdn_failure_census
MODEL=/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet/models/puigde_360M

echo "===== [1/3] smoke: weight load + real forward + paired small test ====="
STAGE=smoke MODEL_ID="$MODEL" bash run_h100.sh

echo "===== [2/3] competence: <=2K capability check ====="
STAGE=competence MODEL_ID="$MODEL" bash run_h100.sh

echo "===== [3/3] screen: 512-8192 context scan ====="
STAGE=screen MODEL_ID="$MODEL" bash run_h100.sh

echo "===== ALL STAGES DONE ====="
