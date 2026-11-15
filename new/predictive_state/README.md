# Active main study: pretrained Qwen3.5-4B × native GDN/GQA × native MTP (2026-11-15)

**The active implementation is now [qwen35_predictive/](qwen35_predictive/README.md); the complete [Chinese GPU runbook](qwen35_predictive/RUNBOOK_ZH.md) specifies independent C00–C04 runs.** It uses the pinned pretrained Qwen3.5-4B Backbone and actual released MTP weights, with a frozen target and small trainable readout/state modules. Existing [psdraft/](psdraft/) below is retained for reference as the earlier randomly initialized tiny GDN drafter, **not** the selected main model.

**Evidence boundary:** GitHub CPU mock CI passes; CUDA-native Qwen3.5 loading, GPU qualification, 4B training, true task accuracy and end-to-end cached speculative decoding speed are still unverified. C04 predicts a real GDN state **sketch for auxiliary supervision**, but does not yet use predicted state to skip K Backbone steps.

---

# Predictive-State Drafter — GDN + Teacher-Anchored OPD

**Research bootstrap, 2026-11-14. No pretrained drafter, completed large-model training, CUDA kernel qualification, acceptance-length improvement or measured speedup is claimed.**

The goal is to stop treating a fixed-size linear-attention state as a lossless historical database. Instead, learn a compact state that predicts the frozen Target LM's next **several** token distributions on the draft model's **own** token prefixes.

## Explicit teacher–student boundary

- **Frozen target/teacher:** pinned `Qwen/Qwen3.5-4B` text backbone, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Use the same model to *verify* speculative proposals. Its native MTP head is an **independent baseline**, not the distribution used to verify next tokens.
- **Student:** independent small Gated DeltaNet recurrent draft head conditioned at the beginning of each draft block on the target's *accepted-prefix* last hidden state. The student shares the target tokenizer/Token IDs but **not** its parameters. Token embedding and output projector are tied to avoid allocating duplicate vocab-sized matrices.
- **No teacher leakage:** student never receives a target hidden state for a proposed token before verification. Student recurrent states are derived from the boundary hidden plus its own proposed tokens. At the next accepted boundary the state is reconstructed from the newly verified target hidden (no rejected token is ever committed).
- **Reference-only v0:** a differentiable Torch rank-one GDN loop without FLA short conv, fused normalization, chunk kernels or cached target verification. Its compute is **not** representative of production GDN throughput. FLA integration is a separate qualification gate.

## Reproduce CPU correctness checks

~~~bash
cd new/predictive_state
python -m pip install -e '.[test]'
python -m pytest -q
~~~

These tests require no network, model checkpoints or GPU. They check the exact GDN update algebra, gradient flow, draft/replay alignment, target stop-gradient, teacher forcing, OPD, target-conditioned acceptance/residual resampling and output distribution in a simple known case.

## GPU qualification / two-stage pilot

Reuse the AKR branch's pinned snapshot and manifest; don't treat unverified model files as qualified. In the existing [AKR](https://github.com/Yunbo-max/AKR/tree/research/mtp-opsd-baseline) checkout:

~~~bash
python -m mtp_opsd.cli download --assets-root assets/mtp_opsd
python -m mtp_opsd.cli check --assets-root assets/mtp_opsd --full-hashes
~~~

On the actual GPU host, in a **separate environment** from older GDN/AKR runs:

~~~bash
cd new/predictive_state
python -m pip install -e '.[train,test]'
python -m pytest -q
export MODEL_DIR=/absolute/path/AKR/assets/mtp_opsd/models/Qwen3.5-4B

# Explicit, prompt-only input. These two example prompts are only a plumbing smoke.
python -m psdraft.cli train --model-dir "$MODEL_DIR" --data data/smoke_prompts.jsonl \
  --mode tf --updates 2 --horizon 4 --out runs/smoke_tf --device cuda:0
python -m psdraft.cli train --model-dir "$MODEL_DIR" --data data/smoke_prompts.jsonl \
  --mode opd --init runs/smoke_tf --updates 2 --horizon 4 \
  --out runs/smoke_opd --device cuda:0
python -m psdraft.cli verify --model-dir "$MODEL_DIR" --data data/smoke_prompts.jsonl \
  --out runs/smoke_opd --horizon 4 --device cuda:0
~~~

**For a real pilot:** export the pinned GSM8K **train questions only** using `scripts/prepare_gsm8k.py` and set `--data` to the generated JSONL. Questions/answers for MATH-500 test are not training inputs. Save each case separately and report actual GPU model loading, finite gradients, memory, target calls and wall clock. A 2-step example is *not* a quality or throughput result. `psdraft.cli verify` runs full-prefix teacher recomputation and is **not** a production inference benchmark.

## Benchmark ladder (distinct experiments; no cherry-picked replacements)

| Case | Model | Training / decoder | Scientific decision |
|---|---|---|---|
| B0 | Frozen target | Native next-token decoding | Reference quality, TTFT and throughput |
| B1 | Frozen target | Native released MTP head (AKR) | Strong target-internal speculative baseline |
| B2 | Small GDN drafter | Teacher forcing on target-generated tokens | Independent drafter capability |
| B3 | Same-size GDN | Continue B2 using student-rollout forward KL | OPD uplift vs same-time/cost B2 |
| B4 | Matched-size Transformer or Mamba draft head | Same conditioning, TF/OPD budget | Is GDN worth choosing? |
| B5 | GDN + feedback state | Learned verification feedback, **not implemented yet** | Does feedback update help without gradient at inference? |
| B6 | GDN + future-state objective | Independent multi-horizon objective, **not implemented yet** | New mechanism beyond ordinary OPD |

For B2–B4 measure matched training tokens, teacher forwards, optimizer steps, parameters, real drafting cost, exact speculative acceptance length, end-to-end tokens/s and GPU memory. Actual speculative decoding must use `p`/`q` acceptance and residual correction, not argmax agreement. Verify target output distribution and state rollback for each target backend. Native Qwen target contains full attention as well as GDN; target KV/cache cost still exists.

## Files

- `psdraft/gdn.py`: readable, differentiable reference recurrence, using the source project's GDN semantics.
- `psdraft/model.py`: fixed-state, boundary-conditioned recurrent drafter; no future-token teacher leakage.
- `psdraft/losses.py`: teacher-to-student all-vocab forward KL and teacher forcing.
- `psdraft/speculative.py`: exact single-block token acceptance and residual sampling, **reference** implementation.
- `psdraft/teacher.py`: pinned native Qwen3.5-4B text-target loading and causally aligned distributions.
- `psdraft/training.py`, `psdraft/cli.py`: one-prompt-at-a-time TF/OPD pilot; checkpoint and receipt.
- `METHOD.md`: mathematical objective, invariants and next-stage hypothesis.
- `REUSE_AUDIT.md`: actual existing code inspected, reuse decisions and missing features.

## Research boundaries

The recurrent state is **not** asserted to store the entire past; it is a compressed statistic selected by future predictive usefulness. OPD, multi-token drafting and GDN are all established ideas. No originality is attributed to merely composing them. The novel candidate is **learning a verifier-conditioned, constant-size predictive adaptation state**, but that algorithm must be specified, implemented and tested after the plain OPD baseline passes.
