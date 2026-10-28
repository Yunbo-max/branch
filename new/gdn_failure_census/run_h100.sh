#!/usr/bin/env bash
# One GPU/process. Stage names describe planned evaluation, never completed results.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
STAGE=${STAGE:-smoke}
MODEL_ID=${MODEL_ID:-m-a-p/340M-20B-GatedDeltaNet-pure-baseline}
OUT_ROOT=${OUT_ROOT:-"$HERE/../../results/gdn_failure_census"}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-bf16}
BACKEND=${BACKEND:-hf}
SEED_START=${SEED_START:-1000}
DRY_RUN=${DRY_RUN:-0}
RESUME=${RESUME:-0}
PREEDIT=${PREEDIT:-1}
EVENT_PLACEMENT=${EVENT_PLACEMENT:-early}
ERROR_POLICY=${ERROR_POLICY:-stop}
MAX_SAMPLES=${MAX_SAMPLES:-0}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-0}
MAX_RUNTIME_SECONDS=${MAX_RUNTIME_SECONDS:-0}
case "$STAGE" in
  smoke)
    LENGTHS=${LENGTHS:-512,1024,2048}; SEEDS=${SEEDS:-3}
    POSITIONS=${POSITIONS:-0.12,0.88}; EVENTS=${EVENTS:-8}
    CONDITIONS=${CONDITIONS:-plain,unrelated,collision,revision_latest,revision_earliest,no_target}
    ALLOW_EXTRAPOLATION=${ALLOW_EXTRAPOLATION:-0}
    ;;
  competence)
    LENGTHS=${LENGTHS:-512,1024}; SEEDS=${SEEDS:-40}
    POSITIONS=${POSITIONS:-0.50}; EVENTS=${EVENTS:-8}
    CONDITIONS=${CONDITIONS:-plain,unrelated,collision,revision_latest,revision_earliest,alias,two_hop,no_target}
    ALLOW_EXTRAPOLATION=${ALLOW_EXTRAPOLATION:-0}
    ;;
  screen)
    LENGTHS=${LENGTHS:-512,1024,2048,4096,8192}; SEEDS=${SEEDS:-20}
    POSITIONS=${POSITIONS:-0.12,0.50,0.88}; EVENTS=${EVENTS:-8,32}
    CONDITIONS=${CONDITIONS:-plain,unrelated,collision,revision_latest,revision_earliest,alias,two_hop,no_target}
    ALLOW_EXTRAPOLATION=${ALLOW_EXTRAPOLATION:-1}
    ;;
  full)
    LENGTHS=${LENGTHS:-512,1024,2048,4096,8192,16384,32768}; SEEDS=${SEEDS:-48}
    POSITIONS=${POSITIONS:-0.12,0.50,0.88}; EVENTS=${EVENTS:-8,32,64}
    CONDITIONS=${CONDITIONS:-plain,unrelated,collision,revision_latest,revision_earliest,alias,two_hop,no_target}
    ALLOW_EXTRAPOLATION=${ALLOW_EXTRAPOLATION:-1}
    ;;
  *) echo 'STAGE must be smoke, competence, screen, or full' >&2; exit 2 ;;
esac
if [[ -z "${OUT_DIR:-}" ]]; then
  SAFE_ID="${MODEL_ID//\//__}"
  SUFFIX=""
  if [[ "$DRY_RUN" == 1 ]]; then SUFFIX="_dryrun"; fi
  OUT_DIR="$OUT_ROOT/${SAFE_ID}_${BACKEND}_${STAGE}_${EVENT_PLACEMENT}_s${SEED_START}_n${SEEDS}${SUFFIX}"
fi
mkdir -p "$OUT_ROOT"
echo "GDN census: model=$MODEL_ID stage=$STAGE output=$OUT_DIR"
ARGS=(--model "$MODEL_ID" --backend "$BACKEND" --device "$DEVICE" --dtype "$DTYPE"
      --lengths "$LENGTHS" --positions "$POSITIONS" --events "$EVENTS" --conditions "$CONDITIONS"
      --seed-start "$SEED_START" --seeds "$SEEDS" --out "$OUT_DIR"
      --event-placement "$EVENT_PLACEMENT" --error-policy "$ERROR_POLICY"
      --max-samples "$MAX_SAMPLES" --max-input-tokens "$MAX_INPUT_TOKENS"
      --max-runtime-seconds "$MAX_RUNTIME_SECONDS")
if [[ "$ALLOW_EXTRAPOLATION" == 1 ]]; then ARGS+=(--allow-extrapolation); fi
if [[ "$PREEDIT" == 1 ]]; then ARGS+=(--preedit-probes); fi
if [[ "$DRY_RUN" == 1 ]]; then ARGS+=(--dry-run); fi
if [[ "$RESUME" == 1 ]]; then ARGS+=(--resume); fi
if [[ -n "${MODEL_REVISION:-}" ]]; then ARGS+=(--revision "$MODEL_REVISION"); fi
if [[ -n "${TOKENIZER_ID:-}" ]]; then ARGS+=(--tokenizer "$TOKENIZER_ID"); fi
if [[ -n "${TOKENIZER_REVISION:-}" ]]; then ARGS+=(--tokenizer-revision "$TOKENIZER_REVISION"); fi
# Partial-budget exit75 remains visible; still summarize the attempted rows.
set +e
python3 run.py "${ARGS[@]}" 2>&1 | tee -a "$OUT_ROOT/$(basename "$OUT_DIR").log"
STATUS=${PIPESTATUS[0]}
set -e
if [[ "$DRY_RUN" != 1 && -f "$OUT_DIR/results.jsonl" ]]; then
  python3 summarize.py --run "$OUT_DIR"
fi
if [[ "$STATUS" != 0 ]]; then
  echo "Run exited $STATUS; inspect manifest execution status. No completion claimed." >&2
  exit "$STATUS"
fi
echo "Completed frozen plan (check any deferred grid cases): $OUT_DIR"
