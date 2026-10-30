#!/usr/bin/env bash
# Targeted probe: GDN memory-update failure.
# Sweeps the number of same-key revisions (1..32) for revision_latest vs
# revision_earliest, with plain as the no-update baseline. Fixed 1024 tokens,
# middle position, 40 seeds. Answers: does reading the LATEST value degrade
# with more updates, while reading the ORIGINAL stays stable?
set -euo pipefail
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet/new/gdn_failure_census
MODEL=/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet/models/puigde_360M
OUT=/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet/results/gdn_update_probe

python run.py \
  --model "$MODEL" \
  --conditions revision_latest,revision_earliest,plain \
  --events 1,2,4,8,16,32 \
  --lengths 1024 \
  --positions 0.5 \
  --seeds 40 \
  --out "$OUT" \
  --device cuda --dtype bf16

python summarize.py --run "$OUT"
