#!/usr/bin/env bash
# MQAR quality comparison: plain DeltaNet (no geometry) vs Full-C GeoDelta
# (= whitened GeoDelta) vs Transformer. Same backbone/data/budget; only the
# memory mechanism differs. 1xGPU.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1
OUT="$HERE/../../results/mqar_quality_$(date +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== env ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

COMMON=(
  --device cuda
  --vocab-size 8192 --seq-len 128 --num-kv-pairs 8
  --num-examples 8192 --seed 17
  --d-model 256 --layers 2 --heads 4 --d-key 64 --d-value 64
  --steps 8000 --batch-size 32 --lr 1e-3 --weight-decay 0.01
  --eval-every 500 --log-every 500 --threads 8
)

run_arm() {
  local mixer="$1"
  echo "===== ARM: $mixer ====="
  python3 train_fair.py --mixer "$mixer" "${COMMON[@]}" \
    --output "$OUT/$mixer" 2>&1 | tee "$OUT/$mixer.log"
  echo "----- done $mixer -----"
}

run_arm delta
run_arm single_geo
run_arm transformer

echo "===== SUMMARY ====="
python3 - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
print(f"{'arm':12} {'valid_acc':>10} {'test_acc':>10} {'valid_loss':>12} {'mixer_params':>12} {'total_params':>13} {'wall_s':>8}")
for d in sorted(out.iterdir()):
    if not d.is_dir():
        continue
    j = d / "report.json"
    if not j.exists():
        continue
    m = json.loads(j.read_text())
    fin = m.get("final", {})
    v, t = fin.get("valid", {}), fin.get("test", {})
    p = m.get("parameters", {})
    tr = m.get("training", {})
    print(f"{d.name:12} {v.get('accuracy',0):>10.4f} {t.get('accuracy',0):>10.4f} "
          f"{v.get('loss',0):>12.4f} {p.get('mixer',0):>12} {p.get('total',0):>13} {tr.get('wall_seconds',0):>8.1f}")
PY
