# Branch Memory — full-covariance reference, v0.2.0

This is the exact algorithm source from the previously supplied
`branch_memory_full_covariance.zip`, recovered through a standalone Markdown handoff.
No model algorithm was changed while preparing this delivery.

Read `CLAUDE.md` and the parent `BRANCH_MEMORY_FULL_COV_CLAUDE_REPRODUCTION.md` for the
complete mathematical specification, method narrative, source audit and research plan.
All program/test file hashes are in `SOURCE_BUNDLE_MANIFEST.json`.

## Run

```bash
python -m pip install -e '.[test]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python train.py --mode branch --covariance full --differentiate-covariance \
  --task recall-diagnostic --steps 100 --seq-len 24 --batch-size 8 \
  --threads 1 --output results/fresh_full_grad
```

Use `--fixed-batch --steps 60 --seed 7` only for the documented overfitting check.
`--covariance diagonal` is an explicit ablation, not a fallback. `--mode snapshot`
uses our shared reader and is not official MARCH. Snapshot capacity exhaustion raises.
CUDA accepts `--device cuda` but was not tested on the handoff authoring host.

## Source layout

- `branch_memory/math.py`: full SPD solver and joint parent/replace/fork policy.
- `branch_memory/geometry.py`: full causal query second moment.
- `branch_memory/mixer.py`: projections, content addresses, reader and serial writer.
- `branch_memory/cache.py`: functional cache, byte accounting and serialization.
- `branch_memory/model.py`: tiny from-scratch LM and the once-shifted LM loss.
- `train.py`: diagnostic or UTF-8 byte training, logs and checkpoints.
- `tests/`: 54 reference regression cases.
- `tools/validate_scale.py`: engineering dimension checks, not a task benchmark.
- `evidence/`: historical reports and separately labelled fresh handoff rechecks.

## Boundaries

Full covariance is supported; the flag `--differentiate-covariance` additionally
backpropagates through its statistics. Hard argmin decisions do not receive
ordinary derivatives across switches. Physical memory is preallocated with a
fixed cap. Dynamic deletion, cross-layer CED/global KV, retrieval trees, production
kernels, native MQAR integration and pretrained adaptation are not implemented.

The reference tests establish software behavior, not a long-context performance
advantage. Historical tiny training checks overfit and worsen held-out NLL.

A future native MQAR adapter must use its already aligned query-position labels;
it must NOT call the current next-token shifting loss on those labels.
