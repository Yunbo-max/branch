# Real Gated DeltaNet baseline: gate does not rescue online MQAR (2026-10-27)

Added `gated_delta` — a proper Gated DeltaNet with a learnable, input-dependent
forget gate `alpha_t = sigmoid(W_g x_t + b_g)` per head, beta matched to
`single_geo` isotropic (`1/(tau*ridge + ||k||^2)`), gate applied to the old
state (GDN convention). The only difference vs `delta` is the gate.

## Online MQAR, 8 pairs, seq 128, 2000 steps, H100

| arm | valid acc | test acc | valid CE | train acc (last) |
|---|---|---|---|---|
| `delta` (fixed beta, no gate) | 0.0007 | 0.0010 | 8.253 | 0.0000 |
| `gated_delta` (learnable gate) | 0.0000 | 0.0007 | 8.325 | 0.0000 |
| `single_geo` (full-C geometry) | 0.0000 | 0.0005 | 8.323 | 0.0000 |
| `transformer` | **0.992** | 0.993 | 0.108 | 1.0000 |

## Conclusion

The learnable forget gate does **not** rescue the delta rule: `gated_delta`
performs identically to `delta` (both ~0%, loss diverging to ~8.3), and no
better than the full-C geometry. This rules out "the delta baseline was a
fixed-coefficient delta rule, not a real GDN" as the explanation — a genuine
GDN fails the same way.

Combined with the geometry-isolation result (C has only a tiny, fragile effect)
and the H200 timing (whitened GeoDelta ≈ 2.5× slower than native GDN), the
single-state delta-rule family (plain, gated, and geometry-conditioned) does
not learn multi-association associative recall, while exact attention does.
No further work on the geometry or a 50B pretrain is justified by these data.
