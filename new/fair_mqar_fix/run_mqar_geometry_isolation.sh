#!/usr/bin/env bash
# Matched-architecture GeoDelta Full-C vs Isotropic causality diagnostic.
# 2-pair seq64 chunk64: C is zero for the ONLY chunk, so output and training
# MUST match at identical initialization (modulo numerical implementation).
# 2-pair seq64 chunk16: 2nd+ chunks can use C, so any real geometry effect
# can appear. Compare like-for-like with an identical SingleGeometryMixer.
set -euo pipefail

STEPS=${1:-1000}
DEVICE=${DEVICE:-cuda}
SEED=${SEED:-17}
OUT_ROOT=${OUT_ROOT:-}
EVAL_EXAMPLES=${EVAL_EXAMPLES:-512}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
if [[ -z "$OUT_ROOT" ]]; then
  OUT_ROOT="$HERE/../../results/mqar_geo_isolation_$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "$OUT_ROOT"

python3 -m pytest -q tests/test_mqar_geometry_activation.py tests/test_online_mqar.py

for CHUNK in 64 16; do
  for GEOMETRY in isotropic full; do
    OUTPUT="$OUT_ROOT/chunk_${CHUNK}_${GEOMETRY}"
    echo "=== geometry=$GEOMETRY chunk=$CHUNK steps=$STEPS ==="
    python3 train_fair.py \
      --mixer single_geo --geometry-mode "$GEOMETRY" --read-mode current_only \
      --chunk-len "$CHUNK" --online-train --online-seed-offset 3000001 \
      --vocab-size 8192 --seq-len 64 --num-kv-pairs 2 \
      --num-examples 8192 --seed "$SEED" \
      --d-model 256 --layers 2 --heads 4 --d-key 64 --d-value 64 \
      --steps "$STEPS" --batch-size 32 --lr 1e-3 --weight-decay 0.01 \
      --eval-every 250 --eval-examples "$EVAL_EXAMPLES" \
      --log-every 250 --threads 8 --device "$DEVICE" \
      --output "$OUTPUT" 2>&1 | tee "$OUTPUT.log"
  done
done

python3 - "$OUT_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
res = {}
for name in ("chunk_64_isotropic", "chunk_64_full", "chunk_16_isotropic", "chunk_16_full"):
    r = json.loads((root/name/"report.json").read_text())
    res[name] = (r["final"]["valid"]["accuracy"], r["final"]["valid"]["loss"])
    print(f"{name:22s} valid acc={res[name][0]:.5f}  CE={res[name][1]:.5f}")
print("within-model Full minus Iso:")
for chunk in (64,16):
    iso, full = res[f"chunk_{chunk}_isotropic"], res[f"chunk_{chunk}_full"]
    print(f"chunk {chunk}: Δvalid_acc={full[0]-iso[0]:+.5f}, ΔCE={full[1]-iso[1]:+.5f}")
PY
echo "Results: $OUT_ROOT"
