# GeoDelta Fast Operator: two independently verifiable paths (2026-10-27)

**Status:** experimental independent branch. **Not integrated into the 0.79B training mixer yet; not benchmarked on H200.** No throughput claim vs fused GDN/MARCH until GPU measurement.

Branch: exp/geodelta-querysketch-woodbury-20261027 (based on fix/geodelta-079b-train-20261027).

## Why this work exists

Existing GeoDelta FullC computes a dense Query covariance C in R^{d x d}, factors H=lambda I+mu C (Cholesky), computes directions A=H^{-1}K/(tau+diag(K H^{-1}K^T)), then runs a generic PyTorch triangular Delta recurrence. Small model results show FullC can improve NLL but this implementation is slower than Isotropic and much slower than optimized FLA kernels. At 0.79B, d_key=256 and 21 layers amplify the cost.

The same-dimensional state recurrence already costs O(T d_key d_value), so adding any extra full covariance and matrix solve **cannot guarantee** better wall time than an optimized plain GDN. To beat GDN in absolute tokens/s, we may need both a very cheap covariance approximation AND a more efficient state/head layout. Time-to-quality is a distinct, worthwhile target.

## Path A: EXACT FullC via whitening and stock FLA DeltaNet

For one frozen geometry block, let H=LL^T, S in R^{d x v}, and define

    k' = L^{-1} k;  q' = L^{-1} q;  W = L^T S
    beta = 1 / (tau + ||k'||^2)
    W_t = W_{t-1} + beta_t k'_t (v_t - W_{t-1}^T k'_t)^T
    o_t = W_t^T q'_t
    S_end = L^{-T} W_end.

This is *exactly equivalent* to GeoDelta's full SPD write in real arithmetic. It moves the nonstandard write direction into a preconditioned coordinate system and lets us call a conventional, optimized Delta Rule kernel instead of implementing an asymmetric-K/A Triton backward from scratch.

The experimental adapter in experimental_whitened_fla.py calls FLA's chunk_delta_rule with beta, q', k', v, scale=1, and a converted starting state. It uses BF16 q'/k'/v due to FLA requirements. This is not bit-exact to FP32 full geometry, and the adapter is NOT validated on H200.

This exact path still pays one d x d Cholesky, two Q/K triangular solves, and state basis transforms per geometry block. It may speed up the recurrent core, but **does not make the SPD geometry cheap**. It is valuable as a reference engineering route.

CPU FP64 proofs: three independent seeds, outputs/final-state/gradients including gradients wrt H agree at tight tolerances. These tests were executed locally against matching reference code and added to tests/.

**GDN extension:** When adding scalar retention `alpha_t`, the same fixed-basis transform gives an exact GDN recurrence:
    W_t = alpha_t W_(t-1) + beta_t k'_t (v_t - alpha_t W_(t-1)^T k'_t)^T.
Thus experimental_whitened_fla.py also accepts `log_gate` and can call
FLA's `chunk_gated_delta_rule` directly. Three more independent FP64
reference tests checked outputs, final states, and all input/geometry/gate
gradients. This does NOT claim that the BF16 fused FLA backend has passed
GPU verification; CUDA-only tests are included but have not run.

The H200 fused-kernel benchmark `bench_whitened_fla.py` now includes a
same-shape native `chunk_gated_delta_rule` timing control. Its GDN gate
and learning rule differ from the ungated GeoDelta arm, so that row is
for wall-time comparison, not model-quality equivalence.

## Path B: FAST fading-window full-correlation Query sketch via Woodbury

Original C_t = (1-gamma) sum_{s<=t} gamma^{t-s} w_s q_s q_s^T.
For gamma=.95 and normalized q, the contribution from older than r is bounded in operator norm:
    ||C_t - C_t^(r)||_2 <= w_max gamma^r.
If 0 <= w <= 2 (as in the existing single-state routing, where w=1+pi), then
    ||C_t - C_t^(r)||_2 <= 2 gamma^r.
The inverse perturbation obeys
    ||H_t^{-1} - H_t^(r)-1||_2 <= (2 mu / lambda^2) gamma^r.
These are worst-case bounds on the retained geometry, NOT global LM output/loss guarantees.

Define weighted last-r Queries B in R^{r x d}:
    B_i = sqrt(mu*(1-gamma)*gamma^{age_i}*w_i) q_i^T,
    H_r = lambda I + B^T B.

By Woodbury:
    H_r^{-1} K^T =
      (K^T - B^T (lambda I_r + B B^T)^{-1} B K^T)/lambda.

The dual covariance is r x r rather than d x d; all selected Query off-diagonal correlations survive, but history older than r is dropped. It is an approximation to FullC, not a lossless redesign.

At gamma=.95:
  r=64:  tail ||C-C_r|| <=0.07505; inverse error <=0.6004 for lambda=1, mu=8.
  r=96:  tail <=0.01454; inverse error <=0.1163.
  r=128: tail <=0.002816; inverse error <=0.02253.
A larger r has smaller worst-case error but potentially less speedup.

Independent random FP64 test at d=256,L=256, gamma=.95, mu=8, normalized random Queries and w in [1,2] observed *mean relative direction errors*:
  r=32 2.14%; r=64 0.43%; r=96 0.084%; r=128 0.017%.
These are synthetic diagnostic outcomes, not checkpoint-derived LM accuracy.

CPU microbenchmark of geometry stage ONLY (d=256, B=1, H=1, L=256, FP32-ish BLAS setup in separate local reproduction):
  Dense FullC: 1.76 ms
  Woodbury r=64: 0.77 ms (2.30x)
  Woodbury r=96: 1.04 ms (1.69x)
  Woodbury r=128: 1.35 ms (1.30x).
Backward microbenchmarks also indicate reduction on CPU, but the timings are noisier. **These are not H200 results and not a whole LM speedup.**

Crucial limitation: if gamma is changed for 4K-long protection (e.g. ~0.999831), gamma^128≈0.979 and the rank-128 sketch is NOT a high-fidelity representation. This trick depends on the current SHORT EMA half-life, not on the 4K context length itself. We must not sell it as a general 4K-retention mechanism.

The prototype functions accept only previously observed Queries and weights. To train a real causal LM, the decoder needs a persistent last-r Query+weight cache, updated AFTER the current chunk write. At chunk boundaries it must choose the correct recent history; streaming must match full prefill. The end-to-end model cache, gradients and fused kernel are future tasks.

## PyTorch tests actually reproduced locally

- Full-rank Woodbury = dense full-covariance reference for forward and input/importance/Key gradients (when the sketch includes all history).
- Rank-truncated Woodbury = exact dense solve of the corresponding truncated C.
- Positive-semidefinite truncation residual, spectral norm never above 2 gamma^r.
- Empty history and causal history-prefix invariance.
- Three independent exact-whitening forward/backward seeds.

The above CPU test batch returned 15 passed (9 Woodbury + 6 whitening including GDN gate). The GitHub tests reproduce those checks. The separate FLA tests are GPU-conditional and have NOT run.

## Required H200 gate (before any 50B-token 0.79B training)

1. Checkout this branch, then:
   python -m pytest -q new/fair_mqar_fix/tests/test_fast_woodbury_geometry.py
   python -m pytest -q new/fair_mqar_fix/tests/test_whitened_delta_reference.py
   python -m pytest -q new/fair_mqar_fix/tests/test_experimental_whitened_fla.py

2. GPU geometry comparison:
   cd new/fair_mqar_fix
   python bench_fast_woodbury.py --device cuda --d-key 256 --history 256 --chunk 256 --heads 6 --ranks 32 64 96 128 160 --warmup 8 --repeats 30 --output ../../results/woodbury_h200_fwd.json
   python bench_fast_woodbury.py --device cuda --d-key 256 --history 256 --chunk 256 --heads 6 --ranks 32 64 96 128 160 --train --warmup 8 --repeats 30 --output ../../results/woodbury_h200_bwd.json

3. GPU fused recurrent comparison:
   python bench_whitened_fla.py --d-key 256 --d-value 256 --heads 6 --chunk 256 --output ../../results/whitened_fla_fwd.json
   python bench_whitened_fla.py --d-key 256 --d-value 256 --heads 6 --chunk 256 --include-backward --output ../../results/whitened_fla_bwd.json

4. Measure native optimized FLA GDN on the same GPU, input layout, d_key, d_value, batch, precision and recurrent kernel chunk size. The existing delta_proxy is NOT GDN.
5. Only if the isolated kernel meets the speed and numerical gates, wire a stateful geometry sketch into SingleGeometryMixer (new optional mode); verify exact causal full/prefill/decode cache semantics, gradients, and compare 4K LM pilot quality. Do not silently replace FullC mode.

**GO/NO-GO for engineering target:**
- Match FullC geometry directions within a specified relative tolerance and reproduce 4K LM/MQAR quality within experimental uncertainty; report actual drift (synthetic mean error is insufficient).
- H200 block-level forward+backward speedup compared with the old dense full-C path.
- Full 0.79B 4K training tokens/s and wall-time to target validation loss comparable to or better than a native fused GDN baseline. If slower at equal quality, it does not meet the user's stated success criterion.
- Evaluate d_key in {128,192,256}: a reduced head state may be necessary to beat fused GDN; adjust other widths for controlled parameter matching.
- Include peak VRAM, prefill throughput, decode token latency, kernel profiler traces and quality. Optimize both time-per-step and time-to-quality.

**Interpretation:** Path A is an exact high-performance-kernel compatibility route. Path B is a bounded-approximation reduced-geometry route. Combining B + a fused asymmetric Delta WY kernel (or an implicit low-rank whitening implementation) is a possible later phase. Neither currently demonstrates superiority to optimized GDN or official MARCH on H200.
