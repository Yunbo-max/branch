#!/usr/bin/env bash
# Discriminating controls: is the revision_latest failure due to
# (a) query wording, (b) update mechanism, or (c) repetition/conflict?
# Short context (1024 tok), middle position, 40 seeds, chance=0.25.
set -euo pipefail
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet/new/gdn_failure_census
MODEL=/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet/models/puigde_360M
ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet/results/gdn_discriminate

echo "===== [A] wording / simplest-update / repetition (single event) ====="
python run.py --model "$MODEL" \
  --conditions latest_alone,earliest_alone,explicit_update,repeat_same \
  --events 1 --lengths 1024 --positions 0.5 --seeds 40 \
  --out "$ROOT/a" --device cuda --dtype bf16
python summarize.py --run "$ROOT/a"

echo "===== [B] repetition sweep for repeat_same ====="
python run.py --model "$MODEL" \
  --conditions repeat_same \
  --events 2,4,8,16 --lengths 1024 --positions 0.5 --seeds 40 \
  --out "$ROOT/b" --device cuda --dtype bf16
python summarize.py --run "$ROOT/b"

echo "===== DONE ====="
