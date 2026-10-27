# GeoDelta 0.79B: 4K → 16K training, reliability gates, and fair baselines
**Date:** 2026-10-27 · **Repo:** Yunbo-max/branch · **Implementation branch:** `fix/geodelta-079b-train-20261027`

## Executive decision

Train the **single-state GeoDelta Full** arm first. Keep the full Query covariance `C` and the exact direction `a = H^-1 k/(tau+k^T H^-1 k)`, with `H = ridge I + protect C`; **do not add Selective RoPE, learned gate, low-rank C, or extra states to the first run**. The first stage trains and profiles at **4,096 context**, then a separate 16,384 context continuation is considered after passing the gates below. 0.79B here means the actual `FairLM` configuration with ≈793M parameters—not a pretrained checkpoint or the native FLA Gated DeltaNet implementation.

**No H200 training has been executed as part of this change.** Every result below is a configuration target or a historical small-model observation unless explicitly qualified.

## 1. What the patch changes and why

| Component | Previous | New training branch | Required verification |
|---|---|---|---|
| DDP | All 32 accumulated microbatches triggered gradient all-reduce | `grad_sync_context` uses DDP `no_sync()` on all but last microbatch; encloses both forward and backward | 8-rank train loss & gradient sanity; speed measurements |
| Mixed precision | `--amp none` | Launcher defaults to `--amp bf16`; mixer Delta recurrence, covariance, and Cholesky execute with autocast disabled in FP32 | BF16/FP32 finite outputs, gradients, and fixed validation panel |
| Optimizer | AdamW baseline | Optional `--fused-adamw` enabled by launch script | Fused and nonfused one-step output/loss sanity |
| 4K data | `PackedCorpus.gather` always extracted first 4K from packed 16K row | `--random-crop` samples deterministic random offsets for training only; panel remains fixed prefix | Same seed/restart reproduces offsets; train/val indices disjoint |
| Full geometry | Complete 256-token Delta triangular solve | `--delta-tile-len 64` optional, exact chronological 4×64 solves using the **same** frozen 256-token `C_b` and `S_b` history snapshot | Full vs tiled forward/final state and all grads |
| Cholesky | Eager duplicate factorizations in old `main` | Integrated lazy factorization from prior `fix/geodelta-causal-efficiency-20261026` | 256-token block counting; H200 profiler |
| C-history | Fixed decay `gamma=.95` | `--protection-decay` exposed, default **unchanged** | Single-change short/long horizon ablations |
| Accounting | `tokens_per_step` meant micro-batch tokens, and resume could misstate step | Report records global tokens/optimizer step, world size, accumulation, segment steps, and packed-row length | Resume from fixed checkpoint, verify step/tokens match |

Current geometry recurrence, **unchanged** by optional inner tiling:

\[
e_t=v_t-S_{t-1}^T k_t,\qquad
a_t={H_b^{-1}k_t\over\tau+k_t^TH_b^{-1}k_t},\qquad
S_t=S_{t-1}+a_te_t^T,
\]

\[
C_{b+1}=\gamma^{L}C_b+(1-\gamma)\sum_{i=1}^{L}\gamma^{L-i}w_i q_iq_i^T,\quad
H_b=\lambda I+\mu C_b,\quad L=256.
\]

`read_mode=chunk_start` remains on: the extra read is `S_b^T q_t` scaled by null-aware router; the chunk-start snapshot must **not** move at 64-token inner tile boundaries.

## 2. Model and compute contract

| Item | Required setting |
|---|---|
| Model | `FairLM(single_geo, geometry_mode=full, max_states=1)` |
| Approx. parameters | ~792.6M (log actual `sum(p.numel())` at launch), not native GDN |
| Layers / hidden | 21 / 1,536 |
| Heads / K / V | 6 / 256 / 256 |
| Main recurrent geometry | FP32 full `[256,256]` per head |
| Training stage 1 | 4,096 tokens; packed rows still 16,384 |
| Geometry chunk / optional recurrence tile | 256 / 0 for baseline; 64 as candidate |
| Accelerator | 8 × H200 in project script; other counts via `GPUS` |
| Device-local microbatch | 1 sequence |
| Target global tokens per step | 4,194,304 |
| Accum at 4K / 8 GPUs | 128 |
| Accum at 16K / 8 GPUs | 32 |
| Token budget | 50,000,000,000 tokens ≈ 11,921 optimizer steps |
| LR / schedule | 4e-4, WSD; warmup 1%, decay last 20%, floor 10% |
| Optimizer | fused AdamW, weight decay 0.1; gradient clip 1.0 |
| Precision | BF16 projections/MLP, FP32 geometry and recurrence |
| Training data | Row-hash split from 16K packed corpus; training uses random 4K windows; held-out panel fixed |
| Protection | `ridge=1`, `protect=8`, `gamma=.95`, differentiable covariance initially |
| Routing | Current + chunk-start read; do not confuse this with MARCH anchor bank |

**Numerical accounting:** `8 * 1 * 4096 * 128 = 4,194,304` tokens/step; `11,921 * 4,194,304 = 50,000,297,984` processed tokens (not necessarily distinct source tokens). If all 50B tokens are to fit in four days, total wall time must average **≤28.99 s/optimizer step**, including validation and checkpoints. No current H200 rate demonstrates this yet; use measurements to set realistic time and/or token budgets.

**Backbone caution:** This `FairLM` uses 2-linear GELU FFN; the native FLAME/FLA GDN/MARCH reference uses SwiGLU and different head/expansion conventions. Do not report proxy arms as paper-matched GDN/MARCH. Our `march_proxy` uses one anchor per `chunk_len=256` (the old `--anchor-interval 512` had no effect in that implementation). Aligned FFN widths are 7424 (GeoDelta) and 7232 (MARCH proxy), approximate parameter parity (within ~0.03%); log exact counts.

## 3. Preflight commands — mandatory before substantial compute

```bash
git fetch origin
git checkout fix/geodelta-079b-train-20261027

python -m pytest -q new/fair_mqar_fix/tests/test_079b_training.py
python -m pytest -q new/fair_mqar_fix/tests/test_single_geometry_parity.py
python -m pytest -q new/fair_mqar_fix/tests
# If running our original Branch tests as well:
python -m pytest -q new/branch_memory/tests

# No data/GPU needed; check generated 4K and 16K args:
DRY_RUN=1 GPUS=8 SEQ_LEN=4096 bash new/fair_mqar_fix/run_0p79b.sh single_full 10
DRY_RUN=1 GPUS=8 SEQ_LEN=16384 bash new/fair_mqar_fix/run_0p79b.sh single_full 10
```

**Acceptance gate A:** all tests pass; `single_full` forward/backward finite; full vs tiled outputs, final `S,C`, parameter/input gradients agree to stated float32 tolerances; 17-token and 513-token irregular streaming matches whole sequence; no future leakage; BF16 state/geometry stay FP32. Any failure → **do not launch multi-day training**.

**Acceptance gate B:** 8-rank process group starts successfully. First isolate speed using short accumulation (same per-GPU activation shape). On the H200 node:

```bash
# 4K single-step profiling on one GPU (forward/backward/optimizer separately):
cd new/fair_mqar_fix
python profile_079b.py --arm single_full --seq-len 4096 --warmup 2 --repeats 5 \
  --delta-tile-len 0 --output ../../results/profile_079b_full256.json --trace-dir ../../results/traces_full256
python profile_079b.py --arm single_full --seq-len 4096 --warmup 2 --repeats 5 \
  --delta-tile-len 64 --output ../../results/profile_079b_tile64.json --trace-dir ../../results/traces_tile64
python profile_079b.py --arm isotropic --seq-len 4096 --warmup 2 --repeats 5 \
  --output ../../results/profile_079b_iso.json

# Multi-GPU smoke: 8*1*4096 = 32768 tokens/step, accumulation 1.
cd ../..
GPUS=8 SEQ_LEN=4096 GLOBAL_TOKENS=32768 \
TIME_BUDGET_SECONDS=3600 EVAL_EVERY=3 CHECKPOINT_EVERY=0 \
RUN_TAG=smoke_4k bash new/fair_mqar_fix/run_0p79b.sh single_full 5
```

These are **micro/short-budget smoke runs**, not equivalent to full global-batch training. Compare `tile=0` vs `tile=64` in median forward, backward and optimizer timings, peak allocated VRAM and profiler operator tables; do not extrapolate tokens/second linearly across different accumulation counts without a full-batch measurement.

## 4. Short training → scale gates

**Stage 1: 4K convergence pilot** (start with Full and Isotropic only). Keep identical seed, packed rows, training budget and evaluation panel:

```bash
# 8*4096*16 = 524288 tokens/optimizer step; 100 steps ≈52M tokens.
GPUS=8 SEQ_LEN=4096 GLOBAL_TOKENS=524288 \
RUN_TAG=pilot_4k DELTA_TILE_LEN=0 EVAL_EVERY=20 CHECKPOINT_EVERY=50 \
bash new/fair_mqar_fix/run_0p79b.sh single_full 100

GPUS=8 SEQ_LEN=4096 GLOBAL_TOKENS=524288 \
RUN_TAG=pilot_4k DELTA_TILE_LEN=0 EVAL_EVERY=20 CHECKPOINT_EVERY=50 \
bash new/fair_mqar_fix/run_0p79b.sh isotropic 100
```

Report: evaluated NLL and token accuracy, train loss vs tokens, grad norms, NaNs/Inf, tokens/sec, memory, variance across evaluation panels, stability across resume. The old small-model Full advantage of `0.091` NLL at 10k steps is **not** an expected 0.79B effect size.

**Stage 2: 4K full/iso controlled training.** Only after Stage 1 is stable, use the target global batch and same wall/token budget. Start Full with `DELTA_TILE_LEN=0`; if tiled parity and actual throughput are better, select `DELTA_TILE_LEN=64` **before** starting matched full-budget runs.

```bash
# 8*4096*128 = 4194304 tokens/optimizer step.
GPUS=8 SEQ_LEN=4096 GLOBAL_TOKENS=4194304 \
RUN_TAG=full_4k EVAL_EVERY=100 CHECKPOINT_EVERY=500 \
DELTA_TILE_LEN=0 bash new/fair_mqar_fix/run_0p79b.sh single_full 11921

GPUS=8 SEQ_LEN=4096 GLOBAL_TOKENS=4194304 \
RUN_TAG=full_4k EVAL_EVERY=100 CHECKPOINT_EVERY=500 \
DELTA_TILE_LEN=0 bash new/fair_mqar_fix/run_0p79b.sh isotropic 11921
```

An early stop is **not** a failed model. Record step and tokens, checkpoint, throughput, memory and exact reason (OOM, disk, time cap, numerical failure, signal). To resume, use **the same model and schedule configuration** with `RESUME=/path/to/checkpoint.pt`, `RUN_TAG` unchanged and total `STEPS` unchanged; the resumed checkpoint's RNG/optimizer/data generator are restored.

**Stage 3: scientific mechanism ablations**, run smaller budgets and only promote those with measured gains. Isolate one variable at a time:

- `gamma=0.95` vs `0.99` vs `0.999323` (longer-lasting protection), all other hyperparameters frozen.
- `differentiate_covariance=True` vs `False` (forward C unchanged, gradient path different).
- `read_mode=chunk_start` vs `current_only` (determine whether extra snapshot, rather than C, explains the gain).
- Query-covariance `C_q` vs key-covariance `C_k` as a separate **future** code branch; current CLI does not implement `C_k`.
- Per-task evaluation: same-key MQAR, paraphrased-query retrieval, 4K old-read drift, general LM validation NLL. Keep real training vs held-out data separate.

**Stage 4: 16K continuation.** After 4K stability and throughput qualification, profile `SEQ_LEN=16384`, `GLOBAL_TOKENS=4194304` (accumulation 32 at 8 GPUs). The packed dataset is already 16K; no cropping at this length. For a scientifically controlled 16K training comparison, use the same 16K starting checkpoint/initialisation and same token budget for both Full and Isotropic. It is legitimate to continue a pretrained 4K model at 16K, but then label the experiment **context extension**, not from-scratch 16K equivalence.

## 5. Baselines, fairness and performance decision

The internal launcher exposes `single_full`, `isotropic`, `rayleigh`, `delta_proxy`, `march_proxy` and `fork`. **The proxy Delta is fixed-isotropic, no learned GDN retention; the proxy MARCH is not the official GDN+cross-layer-anchor model.** They are useful engineering controls only.

A real published-model comparison requires connecting **FLA's fused Gated DeltaNet**, with its trainable retention `g` and `beta`, and a native MARCH implementation backed by the same optimized recurrent kernels. A separate tree at `new/MARCH/code/march/` is a Python-per-token reference and explicitly not optimized for 0.79B. See `new/MARCH/preprocess/train_commands.md` for the independent FLAME/FLA protocol; the FFN/backbone/head dimensions there differ and should be reported, not silently treated as equal.

**Acceptance for further scaling toward 27B:**
1. GeoDelta works end-to-end with BF16 outer/FP32 geometry and passes train/prefill/decode parity on real chunk boundaries.
2. Robust quality advantage over same-backbone Isotropic/GDN controls survives several seeds and realistic tasks, not only the early 18M pilot.
3. Operator-level profile identifies the bottleneck. Factorization cost scales as `O(d_k^3)` per geometry block, covariance as `O(L d_k^2)`; for `d_k=256` optimizing the recurrent CUDA kernel is necessary before a 27B launch.
4. GPU throughput/memory is measured, and every completed/partial run has a precise checkpoint, git SHA, config and data split.
5. Native FLA GDN and native MARCH are benchmarked at a clearly documented architecture/parameter/compute protocol.

## 6. Diagnostics and stop conditions

Stop/pause the run and keep the latest checkpoint if: nonfinite loss/gradients, persistent diverging validation NLL, forward–stream decode parity failure, repeating CUDA OOM, checkpoint disk-write failure, or actual projected training time exceeds the allowed wall budget by a large margin. Change **one** cause at a time; the initial 4K run should not mix position rotation, new gates, changing `C` horizon and kernel reimplementation.

Checkpoint storage at 0.79B FP32 + Adam moments is roughly **9.5GB or more per checkpoint** (plus metadata and IO buffers). Keep extra free disk space for writing the next checkpoint before deleting the old one. Test restart and exact optimizer-step restoration before long runs. The training sample contains random overlapping 4K windows cut from packed 16K rows; 50B processed tokens are not necessarily 50B unique data tokens, and a row-level holdout is not necessarily document-disjoint.

**Bottom line:** fix training infrastructure, benchmark 256 vs 64 tiles, run 4K Full-vs-Iso pilot, establish true fused GDN baseline, then commit to 50B/16K and future 27B/100B scaling.
