# Step 3 diagnostics: 2-pair easy + gradient-through-C (2026-10-27)

## Easy task (2 KV pairs, seq 64, online)

| arm | valid acc | test acc | train acc (last) |
|---|---|---|---|
| `delta` (plain DeltaNet) | **0.200** | 0.213 | 0.156 |
| `single_geo` (Full-C GeoDelta) | 0.079 | 0.085 | 0.047 |
| `transformer` | 0.994 | 0.991 | 1.0 |

Random chance is ~2% (48 value symbols). Both recurrent arms now learn above
chance — confirming the 8-pair failure is a **capacity** limit, not a broken
recurrence. But the full-C geometry is **worse** than plain delta (7.9% vs 20%).

## Gradient through C (8 pairs, seq 128)

`single_geo` with `--differentiate-covariance`: valid acc 0.0000, test 0.0005.
Letting gradients flow through the covariance does **not** rescue learning.

## Conclusion

The protection geometry is not just unhelpful — at the capacity boundary where
both arms can learn, it is **counterproductive** (single_geo ≈ 0.4× delta on the
2-pair task), and it never rescues the 8-pair case. Combined with the H200
timing (whitened GeoDelta ≈ 2.5× slower than native GDN), there is no evidence
justifying further work on the full-C / Gated-Woodbury geometry, nor a 50B-token
0.79B pretrain of this operator.

## Status

- Do not integrate Gated Woodbury into 0.79B at this stage.
- The single-state delta rule itself is capacity-limited (2 pairs OK, 8 pairs fail);
  exact attention remains the only arm that generalises.
