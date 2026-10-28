# Gated Query-Covariance Woodbury — math, code state, scalability gates

**Date:** 2026-10-27.
**Status:** standalone geometry-only CPU-tested prototype; **not wired into SingleGeometryMixer**, not trained at 0.79B, not H200 benchmarked, not a fully-fledged fused GDN replacement.

## 1. Important distinction: three separate gates

* Main-memory retention \`alpha_t\`: GDN state forgets old value-key associations.
* Protection-statistics retention \`rho_t\`: decides how long previous Query directions influence C.
* Query importance \`w_t\`: weights the new Query in C. In existing single-state GeoDelta it is \`w_t = 1 + pi_t\`, bounded by 2 for null-aware router output \`pi_t in [0,1]\`.

A learnable \`rho_t\` is **not** yet a learnable main-memory GDN gate. The current trained \`single_geo\` uses \`alpha_t=1\`, fixed \`rho=.95\`, and \`w_t=1+pi_t\`. The experimental \`whitened_gated_geo_reference\` supports a scalar \`alpha\`, but this is not installed in the training mixer.

## 2. Learnable covariance gate and exact truncated weights

\`QueryProtectionGate(d_model,n_heads)\` predicts per-head
\`rho_t=sigmoid(W_rho h_t+b_rho)\`, initialized to 0.95 so the model begins as the old fixed decay.

\`\`\`
C_t = rho_t * C_(t-1) + (1-rho_t)*w_t*q_t*q_t^T
C_0 = 0.
\`\`\`

Expanding gives
\`\`\`
C_t = sum_{s=1}^t [(1-rho_s)*w_s*prod_{j=s+1}^t rho_j] q_s q_s^T.
\`\`\`

For the last \`r\` observed Queries, define \`B\` with rows
\`\`\`
B_s = sqrt(mu*(1-rho_s)*w_s*prod_{j=s+1}^t rho_j) q_s^T.
\`\`\`

Then \`H_r=lambda I+B^T B\`. Solve \`H_r^{-1}k\` by Woodbury using an \`r x r\` Cholesky. This is exact for this truncated covariance and retains ALL cross-feature / off-diagonal correlation among selected queries; it is generally only an approximation to the entire history.

**No future leakage:** for any geometry chunk \`b\`, only queries with token position < b may be used for that chunk's write directions. The current chunk's gates and queries update the persisted sketch after that chunk is committed.

## 3. Variable-gate truncation error bound

Since \`rho in [0,1]\`, \`||q||<=1\`, \`0<=w<=w_max\`, and \`C_0=0\`, one has \`||C_u||_2 <= w_max\` for all \`u\`.

Truncate before the last \`r\` elements. The exact positive-semidefinite discarded covariance is
\`\`\`
C_t - C_t^(r) = [prod_{j=t-r+1}^t rho_j] * C_(t-r).
\`\`\`
Thus
\`\`\`
||C_t-C_t^(r)||_2 <= w_max * prod_{j=t-r+1}^t rho_j.
\`\`\`
For \`H=lambda I+mu C\` the inverse perturbation is bounded by
\`\`\`
||H^{-1}-H_r^{-1}||_2 <= mu*w_max/lambda^2 * prod_{last r} rho_j.
\`\`\`
This is a **local geometric bound**: not a guarantee of the final LM NLL, long-horizon model memory, or future Query retrievability.

\`selected_rank(retention, tolerance, max_rank)\` computes the smallest **group-wide** \`r\` with a guaranteed covariance tail <= tolerance, and returns \`None\` if impossible within the maximum.

**Important production limitation:** the reference selection uses a Python-side loop and reductions; it cannot be used inside a high-throughput GPU graph. Replace with a fused prefix-log-product and a small fixed-bucket rank selection (e.g. 32/64/96/128) without CPU synchronization.

## 4. Long-horizon gate trap and cost

A Gate does NOT simultaneously guarantee long-lived protection AND a small query list. If the model learns \`rho_t≈.999831\`, 128 tokens retain a factor of roughly .979, so last-128 is not a good approximation to 4K protection. When active rank \`r>=d_key\`, Woodbury dual is unlikely to beat dense \`d x d\` factorization; use a dense-C fallback or compress old query directions (e.g. streaming covariance sketch/anchors with measured approximation error). The current pure reference does not implement that fallback; \`None\` signals this required action to the caller.

A learned Gate also consumes compute and can collapse to trivial forgetting if trained only to optimize throughput. Optimize with genuine LM loss and evaluate old-read drift and new-query retrievability, subject to an explicit compute budget.

## 5. DeltaNet-family identity

In original key coordinates, GeoDelta is NOT the plain DeltaNet update:
\`\`\`
DeltaNet:       S_t = alpha_t*S_(t-1) + beta_t*k_t*e_t^T.
Gated GeoDelta: S_t = alpha_t*S_(t-1) + a_t(C_b,k_t)*e_t^T
                 e_t = v_t - (alpha_t*S_(t-1))^T k_t
                 a_t = (lambda I+mu C_b)^{-1} k_t /
                       [tau + k_t^T (lambda I+mu C_b)^{-1} k_t].
\`\`\`
But for fixed block geometry \`H_b=LL^T\`, setting \`k'=L^{-1}k\`, \`q'=L^{-1}q\`, \`W=L^T S\` converts it exactly into ordinary (gated) DeltaNet in a rotated/rescaled coordinate system. Therefore **GeoDelta is a preconditioned Delta/GDN family architecture, not an unrelated recurrence**. Its engineering value is query-protection geometry and potential quality or time-to-quality gains, not escaping the Delta rule.

## 6. Verification evidence and next gate

New files:
- \`gated_woodbury_geometry.py\`: learnable gate, full-precision gated Query covariance, dual solver, truncation bound, adaptive rank reference.
- \`tests/test_gated_woodbury_geometry.py\`: 10 pytest cases.

Executed CPU PyTorch 2.10.0: **10 passed** (\`python -m pytest -q\` in an independent temporary verification workspace; same exact source copied to the GitHub branch). Verified dual=dense for all retained history; backward w.r.t Query importance, retention gate, keys; PSD tail bound; rank fallback; causal prefix invariance; gate init and gradient.

**Not verified:** integration into full LM, 4K prefill/decode streaming, H200 kernels/speed and GDN quality. No trained Gate weights; prototype creates parameters but doesn't fit them on language data.

Next: verify repository pytest; implement a fixed-bucket ring buffer or online covariance compression; H200 compare 256-dimensional dense Cholesky vs r64/r96/r128, including gating and ring update; only then introduce a \`geometry_mode=gated_woodbury\` into 0.79B training with fixed-seed accuracy/throughput gates. Never silently switch the old \`single_full\` mode.
