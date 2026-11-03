#!/usr/bin/env bash
# Final rwdiag round: v2 binding pilot, then the independent Q-only test -> route.json.
set -euo pipefail
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet/new/gdn_update_training
export MODEL=/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet/models/puigde_360M
export OUT_ROOT="$PWD/runs/binding_v2"

echo "===== [1/2] v2 pilot (audit + check + fit + eval + report) ====="
bash run_rwdiag.sh pilot

echo "===== [2/2] finish_rwdiag (independent test, candidate q) ====="
pilot=""
for p in "$OUT_ROOT"/pilot_*/; do
  [ -f "$p/dev/decision.json" ] && pilot="$p"
done
if [[ -z "$pilot" ]]; then echo "no pilot with dev/decision.json found under $OUT_ROOT" >&2; exit 1; fi
pilot="${pilot%/}"
echo "pilot dir: $pilot"
python finish_rwdiag.py --pilot "$pilot" --candidate q

echo "===== route.json ====="
python -m json.tool "$pilot/design_confirmation/route.json"

echo "===== ALL DONE ====="
