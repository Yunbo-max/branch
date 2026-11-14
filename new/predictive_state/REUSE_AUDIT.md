# Source/code reuse audit — 2026-11-14

The following *source* was read before designing this branch; listed assets do not imply any GPU experiment has been run in this new project.

| Source | Inspected code | Reused principle | Explicitly not transferred |
|---|---|---|---|
| `Yunbo-max/branch` `exp/gdn-update-learning-20261030` | `new/MARCH/code/march/recurrence.py`, `new/branch_memory/branch_memory/mixer.py`, `new/gdn_update_training/train.py` | Original decay-then-Delta-write formula, per-head recurrent state shape, CPU math checks | Branch fork/routing/covariance, history state anchors, current-value collision losses |
| `fla-org/flash-linear-attention` `main` | `fla/layers/gated_deltanet.py`, `fla/ops/gated_delta_rule/chunk.py` | Official accelerated layer has projection, short-conv, chunk training, fused recurrent decoding | We do not call v0 PyTorch code equivalent in speed or layer-by-layer numerical parity |
| `Yunbo-max/AKR` `research/mtp-opsd-baseline` | `mtp_opsd/backend.py`, `losses.py`, `train.py`, `qualification.py`, pinned `common.json` | Frozen Qwen3.5-4B model identity, official tokenizer, teacher/student prefix row alignment, full-vocab forward KL, provenance/qualification | Same-model, answer-majority, reference-conditioned OPSD; native MTP is kept as separate baseline |
| `Yunbo-max/SPECTRUM` `main` | `src/improving/modeling.py`, `src/improving/training.py` | Native chat template, no gold answer leakage, full example tracking, apples-to-apples sampling budgets | Spectral KV perturbation, OPD as alleged new idea |
| `Joshua-Zhang-Jiaquan/LaLanD` `diffbirwkv` | `laland_1m/experiments/diffumamba_mamba2_100m/model.py` | FLA Mamba/RWKV implementation and architecture ablation conventions | Bidirectional masked diffusion denoiser as an autoregressive drafter |

External research baselines needing direct comparison: [Mamba Drafters](https://arxiv.org/abs/2506.01206), [Draft-OPD](https://aclanthology.org/2026.emnlp-main.1139/), [SpecOPD](https://arxiv.org/abs/2611.02846), [SpecLA](https://arxiv.org/search/?query=SpecLA&searchtype=all), and AKR's official Qwen native MTP.

## Implementation deltas / boundary

- This branch adds *new* `new/predictive_state/` files only. Previous GDN training, MARCH, Branch and results remain unchanged.
- v0 uses an explicit reference GDN; no copied FLA implementation or third-party kernel/vendor tree. Source names above are attribution, not API assumptions.
- v0 uses direct full-prefix target forwards (2 for OPD) for correctness. Cache-aware teacher verification and performance comparison are **missing**; this is a deliberate qualification blocker, not an unnoticed performance claim.
- Pinned manifest and model loading must be qualified on the actual machine; CPU unit tests do not validate the released Qwen weights.
- Stage B5 verifier state and B6 multi-horizon new objectives are research drafts, not existing code or positive results.
