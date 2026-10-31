# GDN update learning — SFT / counterfactual consistency / native-write distillation

**Status:** implementation + CPU regression checks; no native CUDA training,
quality improvement, deployment speed or full-model parity result is claimed.
Based on census commit `6bbba481a56b254df965238f91e427fc9fce030e`.
All old census results and scripts remain unchanged.

## What is implemented

| CLI mode | Common objective | Additional training signal | Export |
|---|---|---|---|
| `sft` | Full-vocabulary answer-token CE, all eight group queries | None | Native GDN |
| `consistency` | Same CE, data and scope | JS between four-choice distributions of two current-value counterfactual histories | Native GDN |
| `write_distill` | Same CE, data and scope | Accepted teacher key-direction + beta targets at one selected layer/head/update-value token | Native GDN |

Do not combine A and B in the first comparison. SFT already sees the paired
examples: extra data is not credited as a method improvement. Default trainable
scope is the existing `q_proj,k_proj,v_proj,a_proj,b_proj`; no adapter or new
inference parameter is added. Trainable weights use FP32 optimizer storage with
BF16 forward autocast; original storage dtypes are restored before final evaluation
and native export. Dropout is disabled (`eval()` with autograd enabled), equally
for all arms. There is no gradient checkpointing in v1.

## Data / interpretation

Each block covers all 12 ordered, distinct old/new A/B/C/D transitions. Both
counterfactual old-value arms are balanced separately. The new answer is not
chosen as the first non-old letter. Entities and current-query templates are
split-specific. Training and test entity IDs never overlap.

One group has eight queries: two histories with different old values but the
same final value; new-only and repeated-new controls; two unchanged-entity
queries; and two earliest-value queries. Only the first two share a JS target.
Historical answers remain different. The protected entity is present in every
history; 'new-only' means no earlier write for the target entity.

Prompts are assembled in token space with exact length and no truncation. The
initial-write slot is padded equally; the final update is at the SAME token
position in all eight arms. The intervention token is the last token of the new
value, NOT guessed from the entity name. `gap` is minimum filler after the update;
actual gap is logged (different query wordings have slightly different lengths).
No claim of byte-identical decode/re-tokenize equivalence is made.

Changing length at fixed gap changes prefix context, not retention distance.
For retention, sweep `--gap` as well as `--length`, use matched groups and report
actual gap. This v1 dataset tests one target update, not an update-count sweep.
Use unseen templates/entities and held-out label-balanced groups for conclusions.
The old census is a separate frozen diagnostic, not replaced or relabelled here.

## Method B: precise implementation boundary

`NativeTap` temporarily observes the REAL post-short-convolution K and sigmoid
beta immediately before FLA's native delta kernel. Candidate normalized keys
retain the original raw key norm; beta stays in [0,1]. The existing kernel is
called with its original q, v, gates and normalization flags. One same key is
used for residual retrieval and writeback: this is not an erase/write split.

The training-only teacher optimizes key direction and beta directly through the
actual downstream forward/backward pass. It minimizes
`softplus(margin + logit_old - logit_new)` with a trust penalty. Every candidate
is replayed and must improve that objective while keeping both unchanged-entity
and earliest-value CE within `protect_tolerance` of their original values.
A rejected search produces no distillation target. Guarding two CE values does
not prove preservation of every fact or guarantee unchanged argmax predictions.

**This is a direct-gradient, replay-checked realization of the feasible native
write family, NOT the earlier proposed state-gradient/closed-form beta oracle.**
It uses a maximum angular key deviation of 0.35 radians and a finite search
budget. Teacher gradients are detached before student supervision. Ground-truth
future questions influence the training teacher only; the student model sees
only its causal input. The privileged teacher may be unlearnable from the prefix;
acceptance rate and eventual student held-out gains must be reported separately.

All hooks, functions and requires-grad flags are restored on exception. Native
exports contain no teacher, hook, history bank or new state. Unsupported FLA
layouts fail rather than switching to a hand-written recurrence. v1 supports
FLA 0.5.0, pure GDN, no negative beta eigenvalues, equal K/V head counts, batch=1
per forward, full no-cache sequences, no packed/varlen inputs, no concurrent
model forward, no distributed teacher and no checkpoint recomputation.

## Run

Use the project's existing working CUDA/FLA 0.5.0 environment. CPU tests need
PyTorch and pytest only; no pretrained weights are downloaded by tests.

```bash
cd new/gdn_update_training
python -m pytest -q tests

# No downloads or CUDA calls: budget/config plan only (separate output directory).
ACTION=plan MODE=write_distill RUN_TAG=plan bash run.sh

# Point this to the already-downloaded compatible puigde checkpoint.
export MODEL=/path/to/puigde_360M
# First native checks: two optimization steps for each arm, one layer/head for B.
MODE=sft STEPS=2 TRAIN_GROUPS=12 EVAL_GROUPS=12 RUN_TAG=smoke bash run.sh
MODE=consistency STEPS=2 TRAIN_GROUPS=12 EVAL_GROUPS=12 RUN_TAG=smoke bash run.sh
MODE=write_distill STEPS=2 TRAIN_GROUPS=12 EVAL_GROUPS=12 \
  TEACHER_STEPS=1 RUN_TAG=smoke bash run.sh

# Pilot: invoke arms separately; do not claim equal GPU cost for B.
MODE=sft RUN_TAG=pilot bash run.sh
MODE=consistency RUN_TAG=pilot bash run.sh
MODE=write_distill RUN_TAG=pilot bash run.sh

# Ordinary native export, without training-only interventions.
python train.py eval --model ../../results/gdn_update_learning/pilot/consistency_seed42/model \
  --out ../../results/gdn_update_learning/eval_a_test_2k \
  --eval-split test --eval-groups 120 --length 2048 --gap 1024
# Longer-than-configured contexts additionally require --allow-extrapolation.
```

Outputs refuse overwrite. `model/` is a normal HF native checkpoint and tokenizer.
There is no optimizer-resume support in v1; start a new output directory. A failed
run writes `failure.json`, never a success receipt. A successful train run contains
config, immutable input fingerprints, code hashes, selected parameter names,
per-step teacher cost and acceptance, before/after raw rows and summary, and export.

Each optimizer step processes one eight-query group. B costs more than the other
arms: one capture plus (1+2*teacher_steps) replays, each of three real forwards,
plus student passes and backwards. Compare both matched student tokens and actual
GPU time. No performance evidence is inferred from CPU references.

## Qualification / limits

The first required GPU acceptance is strict model load -> one native backward
per arm -> nonzero finite projection/gate gradients -> teacher capture/replay ->
export -> strict native reload. These have NOT been performed in this session.
The CPU reference checks plumbing and mathematics, not FLA/CUDA gradients or
pretrained quality. Native API shape was reviewed against FLA v0.5.0
`fla/layers/gated_deltanet.py`; compiled-kernel gradient compatibility needs smoke.

The common objective is **answer-token SFT**, not full-text LM pretraining.
Unchanged-entity/history controls protect only this task. General LM loss/data
mixing and general benchmark retention are not implemented or established.
Summaries are descriptive; groups and repeated queries are not independent model
training seeds. No significance or first-in-world novelty claim is made.
