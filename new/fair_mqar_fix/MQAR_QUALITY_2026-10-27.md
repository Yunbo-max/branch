# MQAR quality comparison (2026-10-27) — negative result

Three arms, identical backbone / data / optimizer / budget; only the memory
mechanism differs. 1x H100, 8000 steps, MQAR seq 128 / 8 KV pairs / vocab 8192.

## Results

| arm | valid acc | test acc | valid loss | mixer params | wall (s) |
|---|---|---|---|---|---|
| `delta` (plain DeltaNet, no geometry) | 0.0004 | 0.0007 | 20.17 | 525,824 | 1447.6 |
| `single_geo` (Full-C GeoDelta) | 0.0002 | 0.0001 | 20.55 | 627,208 | 248.7 |
| `transformer` (attention) | **0.968** | 0.967 | 0.16 | 525,824 | 54.6 |

## Conclusion

- **The full-C geometry buys no quality over the plain delta rule on MQAR.**
  Both recurrent arms memorise the training set (train acc → 1.0) and fail to
  generalise (valid/test acc ≈ random); their valid loss diverges to ~20,
  far above the ~9.2 random baseline.
- The protection geometry is meant to reduce write interference across the
  multiple KV associations, but it does not change the outcome: `single_geo`
  is no better (marginally worse) than `delta`.
- Attention generalises to 96.8% because MQAR is exact in-context lookup.

## Scope / caveat

MQAR with 8 KV pairs may be beyond what a *single* compressed recurrent state
can represent regardless of geometry, so this is strong but not exhaustive
evidence. The remaining un-tested question is whether the geometry helps on
language-modelling pretraining (4K LM pilot); earlier full-C LM curves
plateaued at ~3.0 (see FINDINGS_2026-10-26.md).

Reproduced by: `bash run_mqar_quality.sh` (in `new/fair_mqar_fix/`).
