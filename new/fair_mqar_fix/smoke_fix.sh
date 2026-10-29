#!/usr/bin/env bash
# Short 1xH100 smoke test for the corrected chunked Branch / MARCH code.
set -euo pipefail

REPO=/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/Deltanet
cd "$REPO/new/fair_mqar_fix"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

OUT="$REPO/results/smoke_fix_correctness"
rm -rf "$OUT"

python3 pretrain_branch.py \
  --packed-dir "$REPO/new/MARCH/data/packed_16k" \
  --seq-len 4096 \
  --batch-size 1 \
  --steps 20 \
  --eval-every 10 \
  --eval-batches 1 \
  --mixer chunked \
  --d-model 64 \
  --layers 1 \
  --heads 2 \
  --d-key 32 \
  --d-value 32 \
  --d-route 16 \
  --max-states 8 \
  --chunk-len 256 \
  --vocab-size 32000 \
  --val-per-mille 20 \
  --test-per-mille 10 \
  --log-every 5 \
  --checkpoint-every 0 \
  --output "$OUT" \
  --device cuda

python3 - <<'PY'
import torch
from chunked_branch import ChunkedBranchConfig, ChunkedBranchMixer

cfg = ChunkedBranchConfig(
    d_model=64, n_heads=2, d_key=32, d_value=32, d_route=16, n_probes=2,
    max_states=8, conv_kernel=3, chunk_len=64,
)
m = ChunkedBranchMixer(cfg).cuda().eval()


def forced(states, factor, active, key, value):
    B, T, H, _ = key.shape
    M = states.shape[2]
    return (
        torch.zeros(B, T, H, dtype=torch.long, device=key.device),
        torch.ones(B, T, H, dtype=torch.bool, device=key.device),
        torch.zeros(B, T, H, 2 * M, device=key.device),
    )


m._route = forced
with torch.no_grad():
    _, cache, stats = m(torch.randn(2, 128, 64, device="cuda"))
active = cache.active.sum(-1)
assert active.min() > 1, active
assert stats["fork"].any()
print("SMOKE_FORK_OK", active.tolist(), "forks", int(stats["fork"].sum()))
PY

echo SMOKE_DONE
