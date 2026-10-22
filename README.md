# Branch Memory vs MARCH — fair long-context comparison

> **Status (2026-10-22): legacy pilot, not a valid method comparison.**
> Two correctness issues reproduced on CPU invalidate the capacity/routing claims below:
> (1) the chunked Branch fork allocator never allocates a child, so the "64-state" run is
> a single-active-state replace-only model; (2) the chunked MARCH port uses the same
> constant anchor key for every anchor, so routing is not content-conditioned. A third
> issue is that evaluation samples from the training corpus rather than a fixed held-out
> panel. See [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) for reproductions and required fixes.
> Do not use the curves in `results/` as evidence of branch capacity or MARCH fidelity.

One backbone, one corpus, one budget, matched parameters. Only the memory mechanism
differs. This repository contains the comparison harness, the four memory arms, the
official MARCH reference code used for parity, and the training curves from the
68.77B-token packed-16K pretraining runs.

## Arms

| arm | file | mechanism |
|---|---|---|
| `march` / `march_chunked` | `new/fair_mqar/march_mixer.py` | evolving delta state + periodic anchors + content-routed Top-4 read (faithful to the released MARCH implementation) |
| `chunked` (Branch v2) | `new/fair_mqar/chunked_branch.py`, `new/branch_memory/` | per-write parent choice, replace/fork, full-covariance protection, content reader |
| `delta` | `new/fair_mqar/baseline_mixers.py` | single delta-rule state, no bank, no history read (Gated DeltaNet baseline) |
| `transformer` | `new/fair_mqar/baseline_mixers.py` | causal MHA + RoPE with incremental KV cache |

Shared backbone: embedding -> N x (pre-LN mixer, pre-LN MLP) -> norm -> head.
The same short depthwise convolution is applied to all arms, including the
Transformer, so the arms differ only in the memory mechanism. See
`new/fair_mqar/README.md` for the fairness caveat.

## Repository layout

```
new/fair_mqar/                  comparison harness, pretraining script, tests, qz job configs
new/branch_memory/              Branch Memory package + MQAR adapter/tests
new/MARCH/code/march_official/  archived official MARCH ops used for routing parity
results/pretrain_full/          Branch v1 (legacy fork implementation) curve
results/pretrain_full_m64/      Branch v2 (replace-only in practice) curve
results/pretrain_march/         MARCH port curve
evidence/logs/                  stdout logs of the three pretraining jobs
```

Data, checkpoints and `__pycache__` are intentionally excluded.

## Pretraining protocol

- Corpus: `new/MARCH/data/packed_16k`, 4,197,404 sequences x 16,384 tokens = 68.77B tokens
- Model: d_model 256, 2 layers, 4 heads, d_key/d_value 64, d_route 32, FF mult 4, vocab 32,000
- Chunk length 256, batch 4 x 16,384 = 65,536 tokens/step, AdamW, lr 3e-4, wd 0.1
- Grad checkpointing, eval every 500 steps on 4 sampled batches, checkpoints every 10,000 steps
- 4-day time budget on 1x H100

Evaluation is **not** a fixed held-out panel: train and eval draw from the same corpus with
different RNG streams, and `eval_gen` advances at every eval call (see `KNOWN_ISSUES.md`,
P0-C). Same-step comparisons across arms are still paired because the eval schedule and
seed are identical.

## Legacy pilot results (do not use as method evidence)

Branch v2 vs Branch v1 (eval loss):

| step | v1 | v2 | v2 - v1 |
|---:|---:|---:|---:|
| 2,000 | 4.997 | 4.866 | -0.132 |
| 5,000 | 4.475 | 4.233 | -0.242 |
| 8,000 | 4.250 | 4.006 | -0.244 |
| 10,000 | 4.164 | 3.854 | -0.309 |
| 11,500 | 3.960 | 3.580 | -0.381 |
| 12,000 | 4.310 | 4.014 | -0.295 |

v2 wins all 25 common eval points (mean -0.222). This is **not** a state-capacity result:
v1 ran an earlier, unarchived source version, and v2 never activates more than one state
per head. Part of the gap may come from the different implementation, not from 64 states.

Branch v2 vs MARCH (eval loss):

| step | v2 | MARCH | v2 - MARCH |
|---:|---:|---:|---:|
| 8,000 | 4.006 | 4.022 | -0.016 |
| 10,000 | 3.854 | 3.884 | -0.029 |
| 12,000 | 4.014 | 4.024 | -0.010 |
| 13,500 | 3.621 | 3.655 | -0.034 |
| 15,500 | 3.746 | 3.783 | -0.036 |
| 16,500 | 3.769 | 3.789 | -0.021 |

From 8k steps onward v2 is lower on all 18 matched eval points (mean -0.029), but the
MARCH arm is handicapped by content-independent anchor keys (P0-B), so this is not a
valid test of Branch vs a faithful MARCH. MARCH also processes roughly 4x more tokens per
wall-clock second, and its eval curve flattens to about 3.2–3.3 after 40k steps.

These are preliminary single-seed curves; use `results/*/metrics.jsonl` for the full traces.

## Reproduce

```bash
pip install -r new/branch_memory/requirements.txt
python -m pytest -q new/fair_mqar/tests new/branch_memory/tests

# short paired comparison
python new/fair_mqar/compare.py --steps 500 --batch-size 16 --seq-len 128 \
  --num-kv-pairs 8 --vocab-size 8192 --d-model 128 --layers 2 --heads 2 \
  --d-key 32 --d-value 32 --d-route 32 --anchor-interval 16 --max-anchors 8 \
  --read-topk 4 --max-states 8 --differentiate-covariance --lr 1e-3 \
  --threads 8 --output-root results/fair

# full pretraining runs (need the packed corpus and a GPU); read KNOWN_ISSUES.md first
bash new/fair_mqar/run_pretrain_full.sh   # Branch v2
bash new/fair_mqar/run_march_pretrain.sh  # MARCH port
```

The `qz_job_*.json` files record the exact platform submissions used for the runs in
`results/`.
