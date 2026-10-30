# GDN census results — puigde 360M (2026-10-30)

First end-to-end run of the census on a real pretrained GDN (1xH100).
`puigde/gated-deltanet-360M-15B-slimpajama` loads exactly into FLA 0.5.0
(no `attn.D` mismatch), all three stages ran with 0 errors / 0 OOM.

## Execution

- smoke (72 cases) → competence (640) → screen (4800) on one card, sequential.
- chance accuracy = 0.25 (4-choice).

## Competence (40 seeds, 512/1024 tok)

| condition | 512 | 1024 |
|---|---|---|
| revision_earliest (read original) | 0.65 | 0.70 |
| plain | 0.50 | 0.575 |
| collision | 0.475 | 0.50 |
| unrelated | 0.40 | 0.425 |
| no_target | 0.325 | 0.25 |
| alias | 0.275 | 0.275 |
| revision_latest (read updated) | 0.225 | 0.275 |
| two_hop | 0.25 | 0.25 |

Gate: `INSUFFICIENT_SHORT_CONTROL` (plain 50% < 60%). no_target ≈ 0.29 (no leakage).

## Screen length scan (20 seeds)

| condition | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| plain | 0.45 | 0.48 | 0.66 | 0.74 | 0.77 |
| collision | 0.53 | 0.47 | 0.61 | 0.63 | 0.68 |
| unrelated | 0.48 | 0.47 | 0.46 | 0.50 | 0.63 |
| revision_earliest | 0.68 | 0.65 | 0.74 | 0.54 | 0.51 |
| revision_latest | 0.18 | 0.27 | 0.25 | 0.27 | 0.27 |
| two_hop | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |
| alias | 0.25 | 0.27 | 0.25 | 0.25 | 0.25 |
| no_target | 0.33 | 0.28 | 0.28 | 0.28 | 0.31 |

## Robust findings

1. `revision_latest` is ~chance (0.18–0.27) at every length: the GDN **cannot
   read the updated value** — it keeps reading the stale original.
2. `two_hop` / `alias` are exactly at chance (0.25) everywhere: **no
   association composition**.
3. `no_target` ≈ 0.28 with no clear above-chance flag: **no leakage**, clean
   control.
4. Basic `plain` retrieval does NOT degrade with length (0.45 → 0.77); the
   failure is not "long-context lookup" per se.

## Hint-level (noisy, needs new-seed confirmation)

- `revision_earliest` may degrade at 4K+ (0.74 → 0.51) but per-cell variance
  is large (0.3–1.0 within a length).

## Next

- Confirm the update/composition failures on a larger FLA-compatible GDN and
  disjoint seeds; run `full` (16K/32K) for the long-context retention curve.
