# GDN update-failure discriminating controls (2026-10-30)

Minimal controls to separate query-wording, repetition, and value-change (update)
as causes of the `revision_latest` failure. 1024 tok, pos 0.5, 40 seeds, chance 0.25.

## Results

| condition | acc |
|---|---|
| earliest_alone (single value, ask EARLIEST) | 0.975 |
| latest_alone (single value, ask MOST RECENT) | 0.775 |
| repeat_same (A then A, ask MOST RECENT) | 0.800 |
| explicit_update (A then "updated to B", ask MOST RECENT) | 0.500 |

repeat_same sweep (2/4/8/16 repeats): 0.85 / 0.85 / 0.825 / 0.75.

## Conclusion

The failure is the **value-change (overwrite) itself**, not query wording or
repetition:

- Query wording is understood: `latest_alone` 77.5%, `earliest_alone` 97.5%.
- Repetition without change is fine: `repeat_same` 80% (75–85% up to 16 repeats).
- One explicit update halves accuracy: `explicit_update` 50%; the log-style
  `revision_latest` drops further to chance with 4+ updates.

Combined with the StaleRate analysis (54–71% of errors read back the initial
value), the model **retains the first write and cannot overwrite it**. This
narrows the cause to (a) address drift across writes or (b) insufficient
write/gating — to be distinguished by internal Q/K/V/gate intervention, not by
more behavioral sweeps.

## New conditions added to spec.py

`latest_alone`, `earliest_alone`, `explicit_update`, `repeat_same`
(reproduced by `run_discriminate.sh`).
