# Geometry isolation: does C causally improve MQAR? (2026-10-27)

Matched-architecture `single_geo` Full-C vs Isotropic, identical seed/budget,
`--read-mode current_only`, online MQAR, 1000 steps. Chunk size controls whether
the covariance C is ever non-zero (one chunk => C inactive) or updated during
KV writes (short chunks => C active).

## Acceptance checks

- Single-block equivalence: `chunk_64` full vs isotropic are numerically
  identical (2-pair 0.00098 vs 0.00098; 8-pair 0.0 vs 0.0). ✓ C is provably
  inactive for the only chunk, confirming the isolation control.
- Multi-block: full and isotropic do produce different write directions. ✓

## Results

2-pair (seq 64), chance ≈ 0.0244%:

| chunk | full valid acc | iso valid acc | Δ acc | Δ CE |
|---|---|---|---|---|
| 64 (C inactive) | 0.00098 | 0.00098 | +0.00000 | -0.00001 |
| 16 | 0.0 | 0.00098 | -0.00098 | -0.027 |
| 2 (C active every write) | **0.00977** | 0.00098 | **+0.00879** | **-0.312** |

8-pair (seq 128): all configs ~0.0; Δ ≈ 0.

## Interpretation

- When C updates every 2 tokens, Full-C is ~10× Isotropic on valid acc
  (0.98% vs 0.098%) with lower CE — a real but **tiny** positive causal effect.
- It is fragile: at chunk 16 it is slightly negative, and on 8-pair it vanishes.
- Even the best case (0.98%) is far below what the plain delta rule reached
  with `read_mode=chunk_start` (~20%) and far from actually solving the task.

## Verdict

The geometry can be shown to have a positive causal effect only in a narrow
regime (C refreshed nearly every token, low associative load), and the effect
does not survive a larger load. This neither rescues the 8-pair failure nor
justifies the 2.5× H200 slowdown. Recommend keeping the 0.79B / 50B-token work
paused; a genuine Gated-DeltaNet (learnable forget gate) baseline is the next
fair comparison before any decision on the geometry.
