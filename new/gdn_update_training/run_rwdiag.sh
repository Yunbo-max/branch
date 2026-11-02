#!/usr/bin/env bash
# Frozen-GDN read/write/retention diagnostic (rwdiag) — first round.
set -euo pipefail
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet/new/gdn_update_training
export MODEL=/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet/models/puigde_360M

echo "===== [1/4] check (tap/intervention == native baseline) ====="
python -m rwdiag.run check --model "$MODEL" --out runs/rwd_check

echo "===== [2/4] fit (train diagnostic reader, frozen GDN) ====="
python -m rwdiag.run fit --model "$MODEL" --scenes 24 --steps 192 --out runs/rwd_fit

echo "===== [3/4] eval (dev panels) ====="
python -m rwdiag.run eval --model "$MODEL" --reader runs/rwd_fit \
  --split dev --scenes 24 --panels swap cross \
  --stages write other neutral records --distances 128 \
  --questions x_now --out runs/rwd_dev

echo "===== [4/4] report (decision.json) ====="
python -m rwdiag.report runs/rwd_dev --out runs/rwd_dev/decision.json

echo "===== ALL DONE ====="
