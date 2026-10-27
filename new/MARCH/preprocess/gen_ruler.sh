#!/usr/bin/env bash
# Generate the RULER NIAH panels used in the MARCH paper (6 tasks x 4K/8K/16K).
#
# Usage: bash preprocess/gen_ruler.sh [num_samples] [lengths...]
#   defaults: 100 samples, 4096 8192 16384
#
# Output layout: data/ruler/<length>/<task>/validation.jsonl
# (RULER's prepare.py writes <save_dir>/<task>/validation.jsonl, so each length
#  needs its own save_dir or the panels overwrite each other.)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RULER="$ROOT/code/RULER"
TOK="$ROOT/data/tokenizer"
OUT="$ROOT/data/ruler"
N="${1:-100}"
shift || true
if [ "$#" -gt 0 ]; then LENGTHS=("$@"); else LENGTHS=(4096 8192 16384); fi

TASKS=(niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multiquery niah_multivalue)

cd "$RULER"
for len in "${LENGTHS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo ">>> $task @ $len (n=$N)"
    python3 scripts/data/prepare.py \
      --save_dir "$OUT/$len" \
      --task "$task" \
      --tokenizer_path "$TOK" \
      --tokenizer_type hf \
      --max_seq_length "$len" \
      --num_samples "$N" \
      --subset validation
  done
done

echo "done -> $OUT"
