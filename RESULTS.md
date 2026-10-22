# Controlled results (fix/correctness)

All numbers below were produced by the corrected code on 1xH100. The raw JSON is committed
next to this file. Both comparisons use the same parameter count, initialization, data
order, optimizer and budget; each arm runs from scratch.

## 1. Single memory (M=1) vs addable fork (M=64), 50 steps

Files: `results/bench_single_vs_addable/bench.json`
Job: `job-3797369e-e697-4b91-b2b1-ae98150b0ca1`

Setting: packed 16K corpus, seq_len 16,384, batch 4, d_model 256, 2 layers, 4 heads,
chunk 256, grad checkpoint on, identical fixed panel (sha256 prefix `9c3736a6c36a`).

| metric | single (M=1) | addable (M=64) | delta |
|---|---:|---:|---:|
| train s/step | **2.30** | 3.77 | +64% |
| peak allocated | **37.5 GB** | 65.1 GB | +74% |
| eval loss | **6.9867** | 7.0197 | +0.033 |
| eval token accuracy | **6.82%** | 6.57% | -0.26 pt |
| active states / head | 1.0 | 45.8 (max 60.3) | |
| fork fraction | 0% | 4.0% | |

At 50 steps the fork arm uses its capacity but does not improve loss/accuracy; it is
clearly more expensive. This is an early point (3.3M tokens), not a final verdict.

## 2. Single-state geometry: isotropic vs full covariance, MQAR

Files: `results/bench_mqar_geometry/bench_mqar.json`
Job: `job-6151204e-fcaa-4ec4-bd1c-09526754a300`

Only `protect` changes: `H = lambda I` (isotropic, protect=0) vs
`H = lambda I + mu C` (full, protect=8). Both are M=1, same backbone, chunk 32, read path,
data and 1,500 training steps (batch 32, seq 256). Training distribution: 8 pairs, gap 32.

Training loss (last 50 steps): isotropic 0.422, full **0.356**.
Time and peak memory are identical (0.136 s/step, 0.375 GB), as expected.

Accuracy (%) on a grid over association count and retention gap:

| pairs | gap | isotropic | full | full - iso |
|---:|---:|---:|---:|---:|
| 2 | 0 | **95.31** | 94.92 | -0.39 |
| 2 | 32 | 99.22 | **99.61** | +0.39 |
| 2 | 64 | 99.22 | **100.00** | +0.78 |
| 2 | 128 | 100.00 | 100.00 | 0.00 |
| 4 | 0 | 91.02 | **93.36** | +2.34 |
| 4 | 32 | 99.22 | 99.22 | 0.00 |
| 4 | 64 | 98.83 | **99.22** | +0.39 |
| 4 | 128 | 98.44 | **99.22** | +0.78 |
| 8 | 0 | 76.56 | **77.34** | +0.78 |
| 8 | 32 | 93.75 | **96.09** | +2.34 |
| 8 | 64 | 95.70 | 95.70 | 0.00 |
| 8 | 128 | 94.53 | **94.92** | +0.39 |
| 16 | 0 | **58.59** | 57.81 | -0.78 |
| 16 | 32 | 73.44 | **76.17** | +2.73 |
| 16 | 64 | 71.48 | **75.39** | +3.91 |
| 16 | 128 | 72.27 | **77.73** | +5.47 |

Reading: full covariance is neutral when the gap is zero, and its advantage grows with
association count and retention interval, reaching +5.5 points at 16 pairs / gap 128.
This is the regime that average language-model loss does not expose.

## What these results do and do not show

- They support "full covariance improves multi-association retention over long gaps".
- They do not yet show a long-horizon language-model gain: 50 steps is too early, and the
  two running pretraining jobs differ in fork behaviour, not only geometry.
- The MQAR diagnostic is a separate small training from scratch, not a fine-tune of the
  pretrained checkpoints.
- Single-seed MQAR; repeat with 3 seeds and ~5k steps before treating +2..5 pt as final.

## Pending: 10k-step LM geometry run

Job `job-a0a38179-7f8d-4e6f-8671-d3aef4d9ac21` runs the same single-state geometry pair
(`protect=0` vs `protect=8`, M=1) for 10,000 steps each on the packed 16K corpus. Runner:
`new/fair_mqar/run_geometry_lm.sh`; outputs: `results/bench_geometry_lm/{isotropic,full}`.
This is the run that will answer whether full covariance improves LM loss at matched
tokens, not just MQAR accuracy.

## Raw outputs

- `results/bench_single_vs_addable/bench.json`
- `results/bench_single_vs_addable/single.json`
- `results/bench_single_vs_addable/addable.json`
- `results/bench_mqar_geometry/bench_mqar.json`
