# SG-Delta v0.1 — stage T0/G3 results (proposal branch)

Branch: `exp/sg-delta-certified-geometry` (not pushed)
Base: `fix/correctness` / `4b2039bd`
Date: 2026-10-23

This is a **negative engineering result** for the group-level conditional solver at the
target chunk length. No long SG-Delta training was run. The Rayleigh/full quality
comparison (G2) was not started because G3 already hits the prescribed stop condition.

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

## Decision

- **NO-GO** for the group-level certified solver at the target `chunk_len=256`.
- Do not run G4 (16K LM) or implement the full `single_geo` pipeline for this claim.
- The rayleigh-vs-full MQAR mechanism (G2) remains scientifically open, but it does not
  rescue the conditional-compute contribution; it would only test whether the full
  geometry benefit is mostly scalar curvature.
- If the direction is revisited, the only defensible next step is a **fused certificate
  kernel + per-token/sub-chunk selective solve**, benchmarked end-to-end at L=32/64 with a
  separate quality check. That is a new engineering proposal, not the current one.

## Files

- `new/fair_mqar/geometry_certificate.py`
- `new/fair_mqar/tests/test_geometry_certificate.py`
- `new/fair_mqar/bench_geometry_conditional.py` (G3 perf only)
- `results/sg_delta/bench_g3_v2/perf.json`
- `results/sg_delta/bench_g3/perf.json` (previous, pre-optimization)
