# Correctness fixes (`fix/correctness`)

This branch addresses the P0/P1 findings in `KNOWN_ISSUES.md`. The legacy curves in
`results/` were produced by the old code and are **not** re-labelled by these fixes.
A new controlled run is required before making method claims.

## P0-A — fork allocation and copy-on-write

File: `new/fair_mqar/chunked_branch.py`

- `_allocate_children` now compares free-slot and fork-parent ranks on the correct axes.
  A single active parent that forks receives the next free slot (verified for
  `M = 2, 4, 8, 64`).
- Counters are capped by the number of free slots; unassigned fork requests fall back to
  replace and the telemetry records the effective decision.
- Fork writes are redirected to the child (`destination = child` for fork tokens); the
  parent receives only replace tokens and keeps its chunk-start state when it has none.
- The child copies the parent state at the chunk start and starts with zero protection,
  so both its routing cost and its update use the isotropic `ridge I` geometry. This is
  the explicit definition of the fork objective.
- The post-hoc scatter (6-D index vs 5-D expand, and wrong write semantics) is removed.
- Telemetry (`fork`, `parent`, `destination`, `write_cost`, `active_states`) now reports
  the actual decisions.
- Additional bug found while fixing: `_route` built `kb` with
  `kk.permute(0, 2, 3, 1, 4)`, mixing the token and head axes. This made the replace
  score use the wrong token/head keys (negative `h`, near-zero denominators). Fixed to
  `permute(0, 2, 1, 3, 4)` and covered by a test that checks the isotropic identity
  `cost_fork == cost_replace + fork_cost` at `C = 0`.

## P0-B — content-conditioned MARCH anchor key

File: `new/fair_mqar/chunked_march.py`

- Removed the shared constant `anchor_embed`.
- The anchor key is now `anchor_k_proj(hidden at the last token of the chunk)`, so
  different prefixes and different anchors produce different keys.
- Tests assert that anchors within a run differ and that two different inputs produce
  different anchor keys.
- This is a **content-conditioned boundary descriptor**, not the official MARCH
  cross-layer anchor-token stream. It should be labelled as a custom control unless it is
  checked against that full official path.

## P0-C — fixed held-out evaluation

Files: `new/fair_mqar/data_split.py`, `new/fair_mqar/pretrain_branch.py`

- Deterministic row-level split (`--split-seed`, `--val-per-mille`, `--test-per-mille`);
  train rows are rejection-sampled from the train partition only.
- The validation panel is drawn once, stored as tokenized rows, and reused for every
  checkpoint. Its row indices and token bytes are hashed.
- `split_manifest.json` records the split, counts, panel size and both SHA-256 hashes;
  the same block is copied into `report.json`.
- The manifest states the limitation explicitly: the packed corpus has no document ids,
  so row-disjoint splits cannot exclude document-level overlap.
- The test partition is reserved and is never used by `evaluate()`.

## P1 — configuration wiring

Files: `new/fair_mqar/fair_models.py`, `new/fair_mqar/pretrain_branch.py`

- `fork_cost`, `covariance` and `differentiate_covariance` are forwarded from
  `FairLMConfig` to `ChunkedBranchConfig`.
- `covariance="diagonal"` now keeps a diagonal protection matrix and builds the geometry
  from it; `differentiate_covariance=False` detaches the incoming protection matrix.
- The actual mixer config is printed at startup as `[mixer-config] ...`.
- Tests assert that all three options reach the mixer and change the geometry/gradient
  path; `run_pretrain_full.sh` and `run_march_pretrain.sh` pass them explicitly.

## Tests

```
cd new/fair_mqar && python -m pytest -q
cd new/branch_memory && python -m pytest -q
```

The new `tests/test_chunked_fixes.py` covers allocation, active growth, parent
preservation, batch/head isolation, full-pool fallback, telemetry, content-conditioned
anchor keys, config wiring, diagonal geometry, covariance gradients, the isotropic routing
identity, split disjointness and fixed-panel determinism.

## Resume support

File: `new/fair_mqar/pretrain_branch.py`

- `--resume checkpoint_<step>.pt` restores the model, optimizer, step, elapsed time,
  torch/numpy/python RNG states, the `torch.Generator` used by the data sampler, and the
  split/panel hashes.
- `--steps` is the **total** step target; the loop continues from `start_step + 1`.
- The metrics log is truncated to the checkpoint step before appending, so a restart does
  not duplicate or retain future rows.
- `report.json` records `resume.start_step`, `data_stream_continuous`,
  `optimizer_restored` and `rng_restored`.
- Numbered checkpoints and the final `checkpoint.pt` both contain the optimizer and all
  RNG states. Previously the final checkpoint omitted the optimizer and no RNG state was
  saved, so a restart silently re-initialized the run.
- `tests/test_resume_and_geometry.py::test_resume_matches_uninterrupted_run` runs 4 steps
  uninterrupted vs 2 + resume-to-4 and asserts identical losses and weights.

## Single-state geometry ablation

Files: `bench_single_vs_addable.py` (state-count comparison) and
`bench_mqar_geometry.py` (geometry-only diagnostic).

The geometry comparison keeps `max_states=1`, the backbone, chunk length, read path, data
order and budget fixed, and changes only:

```
isotropic: H = lambda I          (protect = 0)
full:      H = lambda I + mu C   (protect > 0)
```

`tests/test_resume_and_geometry.py` checks that `protect=0` gives exactly
`chol(ridge I)` regardless of `C`, that `protect>0` changes the factor, and that both arms
have identical parameter names/shapes. Do not add fork/read/routing/state-count changes to
this ablation.

## MQAR diagnostic

Files: `mqar.py`, `bench_mqar_geometry.py`, `tests/test_mqar.py`.

Fixed-length synthetic multi-query associative recall:

```
[filler]*prefix  k1 v1 ... kn vn  [filler]*gap  q  v_q
```

The supervised token is the final `v_q`. The benchmark trains the isotropic and full
single-state arms on the same batches, then evaluates a grid over `num_pairs` (association
count) and `gap` (retention interval). This separates write-geometry effects from average
language-model loss.

## Still open

- Performance: the chunked writer still expands all `M` slots and builds an `L x L`
  system for inactive slots; this branch does not claim a speedup. Profile before
  changing the geometry or writing a sparse kernel.
- Document-level split: needs source document ids; the row-level split is the strongest
  guarantee available from the packed shards.
- New runs: all experiments must be re-run with this branch, a fixed split/panel and the
  corrected fork/routing code. The legacy `results/` curves remain as pilot data.
