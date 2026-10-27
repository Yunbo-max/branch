#!/usr/bin/env bash
# H200 gate runner: exact Fast-GeoDelta operator benchmarks (forward/backward)
# plus the GPU-gated FLA parity tests. Saves JSON results under results/.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1
RES="$HERE/../../results"
mkdir -p "$RES"

echo "===== env ====="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "no nvidia-smi"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "===== fla check ====="
python - <<'PY'
try:
    import fla
    print("fla", getattr(fla, "__version__", "?"), "OK")
    import fla.ops.delta_rule, fla.ops.gated_delta_rule
    print("delta_rule / gated_delta_rule import OK")
except Exception as e:
    print("FLA-IMPORT-FAIL:", repr(e))
    raise SystemExit(1)
PY

echo "===== GPU-gated tests ====="
python -m pytest -q tests/test_experimental_whitened_fla.py -p no:cacheprovider 2>&1 | tail -20

echo "===== woodbury fwd ====="
python bench_fast_woodbury.py --device cuda --d-key 256 --history 256 --chunk 256 \
  --heads 6 --ranks 32 64 96 128 160 --warmup 8 --repeats 30 \
  --output "$RES/woodbury_h200_fwd.json" 2>&1 | tail -20

echo "===== woodbury bwd ====="
python bench_fast_woodbury.py --device cuda --d-key 256 --history 256 --chunk 256 \
  --heads 6 --ranks 32 64 96 128 160 --train --warmup 8 --repeats 30 \
  --output "$RES/woodbury_h200_bwd.json" 2>&1 | tail -20

echo "===== whitened fla fwd ====="
python bench_whitened_fla.py --d-key 256 --d-value 256 --heads 6 --chunk 256 \
  --output "$RES/whitened_fla_fwd.json" 2>&1 | tail -30

echo "===== whitened fla bwd ====="
python bench_whitened_fla.py --d-key 256 --d-value 256 --heads 6 --chunk 256 \
  --include-backward --output "$RES/whitened_fla_bwd.json" 2>&1 | tail -30

echo "===== DONE ====="
ls -la "$RES"
echo "----- summary of json files -----"
for f in "$RES"/woodbury_h200_*.json "$RES"/whitened_fla_*.json; do
  [ -f "$f" ] && echo "### $f" && python -c "import json,sys; d=json.load(open('$f')); print(json.dumps(d, indent=1)[:1500])"
done
