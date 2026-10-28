# Oracle check: orthogonal addresses rescue the delta rule (2026-10-27)

Pre-check for BindDelta. `oracle_orthogonal_bind.py` runs a single-state delta
rule with FIXED mutually-orthogonal key addresses (48 keys x 64 dims, Gram
off-diagonal 1.6e-7) on online MQAR, sweeping num_pairs.

## Result

| num_pairs | analytic (no training) | trained (learned value codebook) |
|---|---|---|
| 2 | 1.000 | 1.000 |
| 4 | 1.000 | 1.000 |
| 8 | 1.000 | 1.000 |
| 16 | 1.000 | 1.000 |

Analytic retrieval is exact because with orthogonal addresses the delta-rule
removal term is identically zero, so the state is `S = beta * sum_i b_i w_i^T`
and `S^T b_q = beta w_q` for every query.

## Conclusion

The single-state delta rule's CAPACITY is not the bottleneck: with orthogonal
addresses it stores/retrieves >=16 associations perfectly. The online-MQAR
failure of the learned-key delta LM (~0% valid acc) is therefore due to ADDRESS
learning/interference, not capacity. This supports the BindDelta hypothesis
(split Q/K into Semantic + Binding codes, train binding codes to be separable).

Run: `python oracle_orthogonal_bind.py --pairs 2 4 8 16`
