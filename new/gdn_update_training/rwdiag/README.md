# 冻结 GDN：读、写和保留的配对诊断

Original design: `GDN_read_write_diagnostic_v1.md`. This package is a runnable first implementation; the design is not a results report. Start in `new/gdn_update_training`. Do **not** use the SFT/B checkpoint as the unadapted baseline. No existing census or training paths are replaced. All CLI output directories must be new.

```bash
export MODEL=puigde/gated-deltanet-360M-15B-slimpajama  # or use the local original checkpoint
python -m rwdiag.run plan --out /tmp/rwd-plan --panels swap cross --stages write other
# On allocated CUDA with the existing FLA 0.5.0 + Transformers environment:
python -m rwdiag.run check --model "$MODEL" --out runs/rwd_check
python -m rwdiag.run fit --model "$MODEL" --out runs/rwd_fit --scenes 24 --steps 192
python -m rwdiag.run eval --model "$MODEL" --reader runs/rwd_fit --out runs/rwd_dev \
    --split dev --scenes 24 --panels swap cross --stages write other --questions x_now
# After choosing and locking site, probe capacity, gate strength and scoring on dev:
# use independent test IDs; include all four questions and both placement orders.
python -m rwdiag.run eval --model "$MODEL" --reader runs/rwd_fit --out runs/rwd_test \
    --split test --scenes 480 --panels swap swap_yfirst cross order new_only repeat absent \
    --stages write other neutral records --distances 128 512 1024 --questions x_now y_now x_first z_now
```

`plan` loads no weights. `check` compares untouched vs tapped full forward, self-replacement vs native, and the normalized last q times the returned K×V state against native single-head output with the native head scale. A failed gate produces `failure.json` and no success receipt. Unit tests run on CPU with a small explicit recurrence; they are not a CUDA parity test. No GPU run is launched by this change. For the full-size design use 1,920/240/480 scenes; the small fit example is a plumbing pilot. `steps` is the number of diagnostic-reader training examples, **not epochs**: increase it independently and check held-out positive controls and convergence. Do not use test for site selection.

### What is actually separated

- Pre-question native state-only reader (`probe`), paired-world state swap, zero state, independently trained shuffled-label probe, and a trained question-only control. `probe_state_swap` is scored against the recipient; join opposite world to obtain donor-following. All four-value heads have their own optimization objective. No entity role/gold/seed/history is input to the small question encoders. Entity and question text are legitimate features.
- `q` changes only the final answer-position q in one native layer/head. The question-only rank-limited direction is learned on training histories through the frozen LM's original output. `q_sham` matches angular perturbation and `q_wrong_entity` is a negative control. Other layers act naturally. Success demonstrates a sufficient intervention at this location, not a unique cause.
- `k`, `beta`, `v` are three **separate** write perturbations with the native read q clamped at the selected site. `q+k` etc. supply the factorial; `*_free_q` reports the unforced native local q. K uses a canonical entity-only prefix; beta uses a fixed dev-chosen strength; V uses the clean-context donor, with a wrong-value donor control. V donor is privileged, not a deployment method; these arms have different capacities, so their relative magnitudes are not attribution percentages. `--span value|terminal|record` changes the write sites. Wrong-site runs are supplied for the value-token setting.
- At `write`, X's full update has just ended. `other` includes an additional Z record; `neutral` and `records` append equal **numbers of tokens**, not equal numbers of words. `actual_gap` includes every token after X, including Y/Z where applicable. `pre` is a before-update control; do not score absent-target rows or pre/new_only as answerable accuracy. Counterbalanced groups are complete blocks of 24. Panel `cross` holds X fixed while changing Y. Only `swap`/`swap_yfirst` are the primary paired binding test.
- Results: `rows.jsonl`, `summary.json`, `qualification.json`, `manifest.json`, `status.json`. Summary reports two-world accuracy, baseline wrong→right and right→wrong, stale and identifiable distractor rates, both-world distractor-following, three factorial contrasts and their joint effect. Intervals bootstrap paired scenes, not independent head/query rows. Four-question joint success, across-stage paired intervals and donor-following can be reconstructed from the raw rows; do not infer them from pooled row accuracy. `absent` uses a hidden, unanswerable label: interpret only as a cue control. Conditioning on distinct labels changes guess baselines.

### Implementation limits / cost

Serial single-layer/single-head full recomputation; no mutable shared prefix cache is used, including for branches. This deliberately avoids cross-question contamination; segmented-cache parity is not claimed or required for this path. No model-sized snapshots are serialized. Peak storage includes one selected head's native K×V matrix plus temporary per-head trajectories; raw row aggregation scales with evaluated cases. This is intervention-heavy, not a speed benchmark. For large confirmations, first profile small dev cells, lock the design, then split by panel/stage/site into separate output directories rather than budgeting from native-forward throughput. The implementation does not certify global information loss when a probe fails, or causality outside the selected intervention location. Version-1 engineering target: pure `gated_deltanet`, FLA 0.5.0, equal K/V heads, nonnegative ordinary beta, no packing/checkpoint recomputation, single process. Inputs beyond the configured window are rejected unless explicitly allowed and must be reported separately.
