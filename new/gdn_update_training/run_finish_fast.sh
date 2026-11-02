#!/usr/bin/env bash
# Re-fit the reader with current code (code_hash changed after --route-only), then
# finish_rwdiag in fast mode. One card, under 3h.
set -euo pipefail
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet/new/gdn_update_training
export MODEL=/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet/models/puigde_360M
export OUT_ROOT="$PWD/runs/binding_v2"

echo "===== [1/2] pilot (re-fit reader with current code) ====="
bash run_rwdiag.sh pilot

echo "===== [2/2] finish_rwdiag --fast ====="
pilot=$(ls -dt "$OUT_ROOT"/pilot_*/ 2>/dev/null | head -1)
pilot="${pilot%/}"
echo "pilot: $pilot"
python finish_rwdiag.py --fast --pilot "$pilot" --candidate q 2>&1 | tee "$pilot/finish.log"
echo "===== route.json ====="
python -m json.tool "$pilot/design_confirmation/route.json"
echo "===== DONE ====="
