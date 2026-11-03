#!/usr/bin/env bash
# From repo root: bash new/gdn_update_training/run_rwdiag.sh [audit|pilot|retention]
# No paid job submission. Set MODEL to the original (not fine-tuned) checkpoint.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mode=${1:-pilot}; py=${PYTHON:-python}
[[ $# -le 1 ]] || { echo 'expected only audit, pilot or retention' >&2; exit 2; }
case "$mode" in audit|pilot|retention) ;; *) echo 'use audit, pilot or retention' >&2; exit 2;; esac
train_n=${TRAIN_SCENES:-24}; dev_n=${DEV_SCENES:-24}; seed=${DATA_SEED:-9001}
root=${OUT_ROOT:-$PWD/runs/binding_v2}; mkdir -p -- "$root"
root=$(cd -- "$root" && pwd -P); dest=$(mktemp -d "$root/${mode}_XXXXXXXX")
echo "RUN_DIR=$dest"
exec > >(tee -a "$dest/console.log") 2>&1
printf 'RUN_DIR=%s\n' "$dest"
trap 'echo "FAILED at line $LINENO. Retain $dest for debugging; see console.log and any failure.json." >&2' ERR
panels=(bind_x_xy bind_x_yx bind_y_xy bind_y_yx bind_swap_xy bind_swap_yx)
for split in train dev; do
  n=$dev_n; [[ $split != train ]] || n=$train_n
  "$py" -m rwdiag.binding_score audit --split "$split" --scenes "$n" --seed "$seed" --out "$dest/audit_$split.json"
done
[[ $mode != audit ]] || exit 0
model=${MODEL:-$PWD/../../models/puigde_360M}
if [[ ! -d $model && $model != puigde/gated-deltanet-360M-15B-slimpajama ]]; then
  echo "Original checkpoint not found: $model. Set MODEL to its absolute directory (or the pinned Hub ID)." >&2; exit 2
fi
common=(--model "$model" --layer "${LAYER:-10}" --head "${HEAD:-0}" --rank "${RANK_DIM:-16}" --data-seed "$seed")
"$py" -m rwdiag.run check "${common[@]}" --out "$dest/check"
reader=${READER_DIR:-$dest/fit}
if [[ -z ${READER_DIR:-} ]]; then
  "$py" -m rwdiag.run fit "${common[@]}" --out "$reader" --panels "${panels[@]}" \
    --questions x_now y_now x_first z_now --stages write other neutral records \
    --distances "${DISTANCE:-128}" --scenes "$train_n" --steps "${FIT_STEPS:-192}"
fi
stages=(other); questions=(x_now y_now)
if [[ $mode == retention ]]; then stages=(write other neutral records); questions=(x_now y_now x_first z_now); fi
# The full v1 binary/provenance checks are retained; do not accept v1-only fit configs.
"$py" -m rwdiag.binding_score check-reader --reader "$reader"
"$py" -m rwdiag.run eval "${common[@]}" --reader "$reader" --split dev --scenes "$dev_n" \
  --panels "${panels[@]}" --stages "${stages[@]}" --distances "${DISTANCE:-128}" \
  --questions "${questions[@]}" --out "$dest/dev"
"$py" -m rwdiag.report "$dest/dev" --out "$dest/dev/base_report.json"
"$py" -m rwdiag.binding_score report "$dest/dev" --out "$dest/dev/decision.json"
printf 'SUCCESS\nreader: %s\nmetrics: %s/dev/summary.json\nbinding: %s/dev/decision.json\n' "$reader" "$dest" "$dest"
