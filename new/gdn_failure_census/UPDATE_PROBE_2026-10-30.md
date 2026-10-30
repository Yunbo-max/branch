# GDN memory-update probe (2026-10-30)

Targeted probe for the `revision_latest` failure seen in the main census.
Sweeps the number of same-key revisions (1..16) at fixed 1024 tokens, pos 0.5,
40 seeds. chance = 0.25.

## Result

| updates | revision_latest | revision_earliest | plain |
|---|---|---|---|
| 1 | 0.40 | 0.825 | 0.575 |
| 2 | 0.45 | 0.70 | 0.575 |
| 4 | 0.15 | 0.70 | 0.575 |
| 8 | 0.275 | 0.70 | 0.575 |
| 16 | 0.15 | 0.70 | 0.575 |

## Conclusion

- `revision_latest` is always <= 0.45 and below `plain` (0.575); at >=4 updates
  it falls to ~chance (0.15-0.275). Even a single update (0.40) already hurts.
- `revision_earliest` is always 0.70-0.825, above `plain`: the GDN retains the
  FIRST write strongly and stably.

Interpretation: the delta-rule memory is **write-once dominant**. The removal
term does not cleanly overwrite an existing address on a repeated key, so the
stale original dominates and the updated value cannot be read back. This is
consistent with the BindDelta "address interference" hypothesis: the failure
is in overwriting, not in storing.

Reproduce: `bash run_update_probe.sh` (in `new/gdn_failure_census/`).
