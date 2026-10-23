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

Consequence for SG-Delta: the fast Rayleigh approximation is not a quality-preserving
substitute for full geometry, so the conditional solver's "easy" branch would lose the
very quality that motivates full geometry. The group-level compute saving is also dead at
L=256 (above). The project should not pursue the Rayleigh/conditional path as specified.

At the time of writing, the 10k LM full-vs-isotropic run had reached step 5,700: full is
ahead by 0.026/0.033/0.047/0.045 eval at 4k/4.5k/5k/5.5k (it was behind at 2.5k). The
10k endpoint is still required before any LM claim.

## Decision

- **NO-GO** for the group-level certified solver at the target `chunk_len=256`.
- **NO-GO for Rayleigh as a fast substitute**: it is worse than isotropic in every MQAR
  cell, while full geometry is +4.5..+7.1 points on high-load/long-gap cells.
- Do not run G4 for the SG-Delta conditional claim. The existing full-vs-isotropic LM run
  continues as a separate full-geometry measurement.
- The defensible follow-up is to **keep full geometry and optimize its compute**
  (active-only / blocked Cholesky, better kernels), not to gate it with this certificate.
  A per-token packed selective solve remains a speculative systems project; it would need
  a fused certificate kernel and an end-to-end win at L=32/64 first.

## Files

- `new/fair_mqar/geometry_certificate.py`
- `new/fair_mqar/tests/test_geometry_certificate.py`
- `new/fair_mqar/bench_geometry_conditional.py` (G3 perf only)
- `results/sg_delta/bench_g3_v2/perf.json`
- `results/sg_delta/bench_g3/perf.json` (previous, pre-optimization)
