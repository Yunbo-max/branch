# SG-Delta v0.1 — stage T0/G3 results (proposal branch)

Branch: `exp/sg-delta-certified-geometry` (not pushed)
Base: `fix/correctness` / `4b2039bd`
Date: 2026-10-23

This is a **negative engineering result** for the group-level conditional solver at the
target chunk length, but G2 shows that **full geometry has a real MQAR quality benefit**
that Rayleigh does not reproduce. No long SG-Delta LM training was run.

## T0 — pure math

`new/fair_mqar/tests/test_geometry_certificate.py`: **11 passed** in FP64.

Covered: random PSD bound validity over d/lambda/mu/tau, exact direct-solve identity,
non-normalized keys, eigenvector exactness, C=0 / mu=0, non-eigenvector residual,
orthogonal equivariance, zero keys, FP32-vs-FP64 stability, group-max semantics.

## G3 — measured geometry cost on 1xH100

Job: `job-194bf8ec-e0c9-4588-831b-0fa1f1816590`
Raw: `results/sg_delta/bench_g3_v2/perf.json`

Times are ms per (batch=1, head=1, chunk). `full` = Cholesky + triangular solve + scalar;
`easy` = the current certificate path (Ck + c/r/U + scalar); `Ck` is the matrix-vector
lower bound. `group_easy` uses the prescribed group-max rule, thresholds from token-level
U quantiles over 256 independent key groups.

| d | L | full | Ck | Cholesky | easy (current) | group easy p90 | group easy p99 | token exact p99 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 32 | 0.131 | 0.015 | 0.055 | 0.387 | 0.035 | 0.738 | 0.010 |
| 64 | 64 | 0.135 | 0.014 | 0.051 | 0.404 | 0.000 | 0.508 | 0.010 |
| 64 | 128 | 0.160 | 0.015 | 0.055 | 0.421 | 0.000 | 0.285 | 0.010 |
| 64 | 256 | **0.132** | 0.014 | 0.050 | 0.387 | **0.000** | **0.070** | 0.010 |
| 128 | 32 | 0.147 | 0.019 | 0.075 | 0.410 | 0.039 | 0.711 | 0.010 |
| 128 | 64 | 0.147 | 0.014 | 0.073 | 0.395 | 0.000 | 0.508 | 0.010 |
| 128 | 128 | 0.147 | 0.011 | 0.075 | 0.387 | 0.000 | 0.297 | 0.010 |
| 128 | 256 | **0.148** | 0.011 | 0.075 | 0.386 | **0.000** | **0.070** | 0.010 |

### Why this is a NO-GO for chunk 256

1. **Group-max is the binding constraint.** At L=256 the fraction of fully-easy groups is
   **0 at the p90 threshold** and only **7% at p99**. A token-level threshold of p99 leaves
   99% of tokens below it, but the max over 256 tokens makes almost every group exact. To
   get 50% easy groups the threshold has to be near the per-group maximum (p100).
2. **Even an idealized easy path cannot meet the 15% target at L=256.** With `Ck` +
   scalar ≈ 0.02 ms (a fused kernel, not the current 0.39 ms), 7% easy groups give
   `0.93*0.132 + 0.07*0.02 ≈ 0.124 ms`, about **6% less geometry time**. Geometry is only
   a fraction of an LM step, so end-to-end is far below 15%.
3. **The current certificate is launch-bound, not FLOP-bound.** `Ck` alone is 0.011–0.019
   ms, but the full certificate takes 0.38–0.42 ms, about 3x *slower* than the full path.
   This is ~20 small tensor ops on L=256 rows; a fused kernel would be needed even for the
   idealized estimate above.
4. **The optimistic regime is small chunks, not the target.** At L=32, d=128, p99 gives
   71% easy groups and an idealized geometry saving of ~60%. That requires a separate
   chunk-length study (quality and throughput), and it is not the 16K/chunk-256 protocol
   in the proposal.

## G2 — isotropic vs Rayleigh vs full on MQAR

Job: `job-16a76c32-5479-4400-a2f8-5e099d089f72`
Raw: `results/sg_delta/mqar_g2/mqar_modes.json`
3 seeds (17/23/42), 1,500 steps, batch 32, seq 256, chunk 32, identical init/data per
seed; eval 1,024 sequences per cell. Values are mean accuracy over seeds (std in
parentheses; percentage points).

| cell (pairs/gap) | isotropic | Rayleigh | full | full - iso | Rayleigh - iso |
|---|---:|---:|---:|---:|---:|
| 8/128 | 93.13 (1.84) | 91.31 (2.43) | **95.08 (1.09)** | +1.95 | -1.82 |
| 16/32 | 73.76 (6.05) | 71.22 (7.53) | **78.29 (4.88)** | +4.52 | -2.54 |
| 16/64 | 73.96 (5.50) | 72.69 (6.75) | **78.58 (5.19)** | +4.62 | -1.27 |
| 16/128 | 72.85 (3.69) | 71.16 (6.93) | **78.84 (4.07)** | +5.99 | -1.69 |
| 24/32 | 54.13 (4.55) | 52.18 (5.83) | **61.20 (4.87)** | +7.06 | -1.95 |
| 24/64 | 52.25 (5.25) | 51.33 (7.47) | **59.24 (5.22)** | +7.00 | -0.91 |
| 24/128 | 52.99 (5.13) | 49.12 (5.49) | **59.38 (4.29)** | +6.38 | -3.87 |
| 16/0 | 55.92 (6.22) | 52.67 (6.63) | 55.63 (5.61) | -0.29 | -3.26 |
| 24/0 | 44.66 (4.46) | 41.24 (4.28) | 42.12 (3.62) | -2.54 | -3.42 |

Reading:

* **Full geometry wins the high-load / long-gap cells** by +4.5 to +7.1 points, consistent
  across seeds (e.g. 16/128 full-iso per seed: +5.76, +6.54, +5.66).
* **Rayleigh is worse than isotropic in every cell**, including the cells where full wins.
  The benefit is therefore *not* the scalar curvature `k^T C k`; it requires the
  non-eigenvector / anisotropic part of the geometry.
* At gap 0 there is no full benefit, matching the earlier single-seed MQAR result.

### 5,000-step confirmation (3 seeds)

Job: `job-55c36550-048f-4ca7-9227-e7e857586e30`
Raw: `results/sg_delta/mqar_5k/mqar_modes.json`

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

The 1.5k advantage shrinks by 5k: full is only +1.0..+3.0 on high-load cells, slightly
worse at 16/32 and 16/64, and clearly worse at gap 0. Rayleigh now ties full on the
high-load cells (24/32: 85.32 vs 85.55; 24/128: 83.56 vs 83.79). The earlier "Rayleigh is
uniformly worse" conclusion is therefore transient and must not be stated as final.

### Honest isotropic vs full LM efficiency (2k steps)

Job: `job-48f3a9f4-8125-4e5e-99f4-c24bff942ff7`
Raw: `results/sg_delta/lm_efficiency_2k/bench.json`

`single_geo` isotropic really skips the covariance and Cholesky factor.

| | s/step | peak allocated | eval loss | token accuracy |
|---|---:|---:|---:|---:|
| optimized isotropic | **1.123** | 36.95 GB | **4.4585** | **24.21%** |
| full | 1.554 | 36.99 GB | 4.4932 | 23.78% |

At 2k full is 38% slower, same memory, slightly worse quality. The quality gain appears
later: at 10k the full-vs-isotropic LM run reached **3.5133 vs 3.6040** (-0.091 NLL), and
the legacy full-geometry single-state checkpoint reached 3.4914 at 10k and 3.0945 at 30k.
So full geometry is a real but compute-expensive quality option, not a free win.

## Decision

- **NO-GO** for the group-level certified solver at the target `chunk_len=256`.
- **Do not treat Rayleigh as a free quality-preserving substitute**: it is worse at 1.5k,
  ties full at 5k, and the MQAR full-geometry advantage itself fades by 5k.
- Full geometry is the best measured single-state LM option at 10k (~0.09 NLL better than
  optimized isotropic, ~0.07-0.09 better than MARCH) but costs ~38% more step time.
- The defensible follow-up is to decide whether that NLL gain is worth the compute, and if
  so **optimize full geometry** (active-only / blocked Cholesky), not to gate it with this
  certificate. A per-token packed selective solve remains speculative.

## Files

- `new/fair_mqar/geometry_certificate.py`
- `new/fair_mqar/tests/test_geometry_certificate.py`
- `new/fair_mqar/bench_geometry_conditional.py` (G3 perf only)
- `results/sg_delta/bench_g3_v2/perf.json`
- `results/sg_delta/bench_g3/perf.json` (previous, pre-optimization)
