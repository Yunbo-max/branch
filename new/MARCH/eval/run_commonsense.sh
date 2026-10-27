#!/usr/bin/env bash
# Zero-shot commonsense evaluation (paper Table 1).
# Usage: bash eval/run_commonsense.sh <checkpoint-or-hf-dir> [output.json]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
HARNESS="$ROOT/code/log-linear-attention/flame/3rdparty/lm-evaluation-harness"
MODEL="${1:?usage: run_commonsense.sh <checkpoint-or-hf-dir> [out.json]}"
OUT="${2:-$ROOT/logs/commonsense.json}"

TASKS="lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,commonsense_qa"

mkdir -p "$(dirname "$OUT")"
cd "$HARNESS"
python -m lm_eval \
  --model hf \
  --model_args "pretrained=${MODEL},trust_remote_code=True" \
  --tasks "$TASKS" \
  --num_fewshot 0 \
  --batch_size auto \
  --output_path "$OUT"

echo "wrote $OUT"
