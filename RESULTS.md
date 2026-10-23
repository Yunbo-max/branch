# Controlled results (fix/correctness)

All numbers below were produced by the corrected code. Raw JSON is committed next to this
file. Comparisons use the same parameter count, initialization, data order, optimizer and
budget; each arm runs from scratch unless stated otherwise.

## 1. Single memory (M=1) vs addable fork (M=64), 50 steps

Files: `results/bench_single_vs_addable/bench.json`
Job: `job-3797369e-e697-4b91-b2b1-ae98150b0ca1`

Packed 16K corpus, seq 16,384, batch 4, d_model 256, 2 layers, 4 heads, chunk 256,
grad checkpoint on, identical fixed panel (sha256 prefix `9c3736a6c36a`).

| metric | single (M=1) | addable (M=64) | delta |
|---|---:|---:|---:|
| train s/step | **2.30** | 3.77 | +64% |
| peak allocated | **37.5 GB** | 65.1 GB | +74% |
| eval loss | **6.9867** | 7.0197 | +0.033 |
| eval token accuracy | **6.82%** | 6.57% | -0.26 pt |
| active states / head | 1.0 | 45.8 (max 60.3) | |
| fork fraction | 0% | 4.0% | |

At 50 steps the fork arm uses its capacity but does not improve loss/accuracy; it is more
expensive. This is an early point (3.3M tokens), not a final verdict.

## 2. MQAR geometry: isotropic vs Rayleigh vs full

### 1,500 steps (single seed, 1,024 eval sequences/cell)

Job: `job-6151204e-fcaa-4ec4-bd1c-09526754a300`
Raw: `results/bench_mqar_geometry/bench_mqar.json`

Full geometry beat isotropic by +4.5..+7.1 points on high-load/long-gap cells (16-24
pairs, gap 32-128); Rayleigh was worse than isotropic in every cell. At this point the
full benefit looked anisotropic and robust.

### 5,000 steps, 3 seeds (17/23/42)

Job: `job-55c36550-048f-4ca7-9227-e7e857586e30`
Raw: `results/sg_delta/mqar_5k/mqar_modes.json`
Mean accuracy over seeds (percentage points).

| cell | isotropic | Rayleigh | full | full - iso |
|---|---:|---:|---:|---:|
| 16/32 | **94.79** | 95.08 | 94.53 | -0.26 |
| 16/64 | **95.51** | 95.15 | 94.73 | -0.78 |
| 16/128 | 94.34 | 95.21 | **95.35** | +1.01 |
| 24/32 | 82.52 | 85.32 | **85.55** | +3.03 |
| 24/64 | 82.52 | 84.11 | **84.93** | +2.41 |
| 24/128 | 82.13 | 83.56 | **83.79** | +1.66 |
| 16/0 | 79.46 | **80.70** | 77.41 | -2.05 |
| 24/0 | **64.29** | 61.23 | 59.24 | -5.05 |

The 1.5k advantage shrinks substantially by 5k: full is only +1.0..+3.0 on the high-load
cells, slightly worse at 16/32 and 16/64, and clearly worse at gap 0. Rayleigh is now
essentially tied with full (24/32: 85.32 vs 85.55; 24/128: 83.56 vs 83.79), so the earlier
"Rayleigh is uniformly worse" conclusion does not survive longer training. Full geometry's
MQAR benefit is early/transient.

## 3. MARCH vs Branch at 10k, same fixed panel

Files: `results/eval_10k_same_panel/eval.json`
Job: `job-51569ea3-79de-49b1-b1b7-4bc525187625`

All checkpoints are step 10,000, scored on the same 262,128-token panel (sha256 prefix
`9c3736a6c36a`). MARCH is evaluated through the archived `chunked_march.py` that produced
it (constant anchor keys).

| model | loss | token accuracy |
|---|---:|---:|
| MARCH (64 anchors) | 3.5852 | 34.10% |
| Branch single-state isotropic (M=1) | 3.6040 | 34.54% |
| Branch fixed fork (M=64) | 4.1169 | 26.16% |

As originally reported, the fix fork is far worse and the isotropic model is essentially
tied with MARCH. The legacy comparison in the next section changes the picture for the
full-geometry single state.

## 4. Legacy v2 at M=1: the earlier 4.483 was invalid

Files: `results/eval_10k_same_panel/eval_m1.json`
Job: `job-5cdbf610-9e34-4a68-b231-8034df8ba842`

The legacy v2 checkpoint was trained replace-only (the fork allocator never fired). The
earlier same-panel evaluation loaded it into the corrected code where forks can trigger,
so that forward no longer matched training and gave 4.4829. Re-evaluating the same
checkpoint with `max_states=1` restores its training-time path:

| model | step | loss | token accuracy |
|---|---:|---:|---:|
| legacy v2, forced M=1 (replace-only, full geometry) | 10,000 | **3.4914** | **36.77%** |
| legacy v2, forced M=1 | 30,000 | **3.0945** | **41.94%** |
| MARCH | 10,000 | 3.5852 | 34.10% |
| single-state isotropic | 10,000 | 3.6040 | 34.54% |

So legacy v2 was genuinely strong: it is a full-geometry single-state model. The earlier
4.483 number must not be used. At 10k it is 0.094 better than MARCH and 0.113 better than
isotropic on NLL, and 2.2-2.7 points better on token accuracy. Its 30k checkpoint reaches
3.0945/41.94%.

## 5. Full vs isotropic LM at 10k (same panel)

Files: `results/bench_geometry_lm/isotropic/metrics.jsonl`,
`results/bench_geometry_lm/full/metrics.jsonl`
Job: `job-a0a38179-7f8d-4e6f-8671-d3aef4d9ac21`

Same backbone, chunk 256, data order, split and fixed panel; M=1; only `protect` changes
(0 vs 8). Both arms reached 10,000 steps.

| step | isotropic eval | full eval | full - iso |
|---:|---:|---:|---:|
| 2,500 | 4.338 | 4.381 | +0.043 |
| 3,500 | 4.176 | 4.161 | -0.015 |
| 5,000 | 3.967 | 3.920 | -0.047 |
| 7,500 | 3.797 | 3.723 | -0.074 |
| 10,000 | **3.6040** | **3.5133** | **-0.0907** |

Full geometry starts slightly behind but crosses over around 3.5k and keeps widening the
lead. At 10k it is 0.091 NLL better (2.5% relative). This is consistent with the legacy
full-geometry checkpoint (3.4914) and shows the earlier "full is worse" reading at 2.5k
was an early-training artifact.

## 6. Honest isotropic vs full efficiency (2k steps)

Files: `results/sg_delta/lm_efficiency_2k/bench.json`
Job: `job-48f3a9f4-8125-4e5e-99f4-c24bff942ff7`

This comparison uses `single_geo` isotropic, which really skips the covariance and the
Cholesky factor. The earlier long-run "isotropic" arm used the chunked mixer with
`protect=0` and still paid for the factor, so it was not an efficient baseline.

| | s/step | peak allocated | eval loss | token accuracy |
|---|---:|---:|---:|---:|
| optimized isotropic | **1.123** | 36.95 GB | **4.4585** | **24.21%** |
| full | 1.554 | 36.99 GB | 4.4932 | 23.78% |

At 2k steps full is 38% slower, uses the same memory, and is slightly worse on the fixed
panel. The 0.091 NLL LM gain only appears later (10k). The trade-off is therefore
compute-for-quality, not free.

## Current long-run status (2026-10-23 ~15:00Z)

| run | step | s/step | last train loss |
|---|---:|---:|---:|
| legacy v2 (replace-only, full M=1 at eval) | 37,977 | 3.99 | 3.235 |
| fixed fork M=64 | 16,574 | 4.36 | 4.281 |
| MARCH | 178,294 | 1.01 | 2.922 |

The geometry run (isotropic + full, 10k each) has completed. The three long runs above
continue.

## Interpretation

- The strongest measured single-state model at 10k is **full geometry** (legacy 3.4914;
  new full 3.5133), about 0.09-0.11 NLL better than isotropic and 0.07-0.09 better than
  MARCH, at roughly +38% step time.
- The earlier claim "MARCH is tied with our best" was based on the isotropic arm only;
  it does not hold for the full-geometry single state.
- On MQAR, full geometry's early advantage fades by 5k and Rayleigh catches up. The
  memory-task evidence is not robust enough to claim a large mechanism win.
- SG-Delta's conditional compute is NO-GO at chunk 256; see `SG_DELTA_RESULTS.md` on the
  experiment branch.

## Raw outputs

- `results/bench_single_vs_addable/bench.json`
- `results/bench_mqar_geometry/bench_mqar.json`
- `results/sg_delta/mqar_5k/mqar_modes.json`
- `results/eval_10k_same_panel/eval.json`
- `results/eval_10k_same_panel/eval_m1.json`
- `results/bench_geometry_lm/isotropic/metrics.jsonl`
- `results/bench_geometry_lm/full/metrics.jsonl`
- `results/sg_delta/lm_efficiency_2k/bench.json`
