# Branch Memory — reproduction report

Date: 2026-10-19
Location: `new/branch_memory/` (inside the DiffRwkv research repo)
Target: full-covariance reference `branch-memory-prototype v0.2.0`

## 1. Source recovery — read this first

The handoff document arrived as a **chat paste**, not as the original
`BRANCH_MEMORY_FULL_COV_CLAUDE_REPRODUCTION.md` file, and the original ZIP
(`branch_memory_full_covariance.zip`) is not present on this machine. Therefore the
hash-verified extractor in Appendix A **could not be used**: it requires the exact
original bytes, and a chat paste cannot guarantee them.

What was done instead: the 12 algorithm/test files in Appendix B were reconstructed
verbatim into this directory, and correctness was established by running the shipped
test suite and independent checks rather than by matching the recorded SHA-256 values.
**The recorded file hashes were not verified.** No algorithm was changed.

Consequence: if the exact bundle is later available, run the Appendix-A extractor into a
fresh directory and diff against this tree. Any diff is a transcription error, not a
design decision.

## 2. Environment

| item | value |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.8.0a0+5228986c39.nv25.06 |
| pytest | 8.1.1 |
| Device | CPU only (`cuda: False`, 0 devices) |
| Threads | 1 (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`) |

Handoff authoring host was Python 3.13.5 / torch 2.10.0+cpu. The declared floor is
Python>=3.10 / torch>=2.2; this run is within it, but it is not the authoring environment.

## 3. Test suite

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
......................................................  [100%]
54 passed in 7.87s
```

**54/54 passed**, matching the documented count.

## 4. Prescribed training checks

### 4.1 Fixed-batch overfitting check (60 steps, seed 7)

| metric | initial | final |
|---|---:|---:|
| fixed-batch NLL | 3.699092 | **0.373286** |
| fixed-batch answer acc | 0.0 | **1.0** |
| held-out NLL | 3.699244 | 5.119915 |
| held-out answer acc | 0.0 | 0.0 |
| forks | 101 | 112 |
| max active states | 8 | 8 |

Final fixed-batch NLL **0.3732863664627075** matches the recorded historical value
`0.3732863664627075` exactly; held-out NLL `5.119915...` matches `5.119915962219238`.
This is a bit-level reproduction of the recorded diagnostic (same seed, same
environment family).

**Interpretation: overfitting confirmed. Held-out NLL nearly +1.42 nats worse.**

### 4.2 Fresh-batch check (100 steps, seed 7)

| metric | initial | final |
|---|---:|---:|
| batch NLL | 3.699092 | **2.718697** |
| held-out NLL | 3.699244 | **2.723097** |
| held-out answer acc | 0.0 | **0.0** |

The model learns general next-token structure (NLL −0.98) but **never answers the recall
query** on fresh batches within this budget. No positive recall evidence.

### 4.3 Scale check (not run)

`tools/validate_scale.py` was not run; the historical `evidence/historical_scale128.json`
remains the only scale record and is explicitly an engineering check, not a benchmark.

## 5. Independent debug checks (beyond the shipped tests)

All run against the restored implementation:

| check | result |
|---|---|
| Per-sample isolation: batch of 2 vs two separate runs | max abs diff **2.8e-17** (float noise) |
| Streaming cache equivalence: full sequence vs chunked | C tensors identical (`atol=1e-10`) |
| Protected geometry SPD / conditioning | min eig **1.0**, max cond **1.86** |
| Fork preserves parent matrix on that step | yes |
| Fork resets child second moment | yes |
| Fork is chosen under a real protection conflict | yes (parent 0, child slot 2) |
| Fork disabled at capacity | yes |
| Null-only read (no visible anchors) | output and weights exactly zero |
| Gradient through covariance when `differentiate_covariance=True` | `protection.requires_grad=True`, finite grads |
| Causal EMA trace bound (`tr C <= 1`) | holds |

**No defect found in the core closed-form solver, the parent/action selection, the
causal C ordering, or the streaming path.**

## 6. Defects / risks worth recording

1. **Source integrity is unverified** (§1). This is the main caveat.
2. **`historical_read` API footgun**: it takes token-level `x` of shape `(B, D)`, while
   `q`/`r`/`states` are head-shaped. Passing a head-shaped tensor raises a confusing
   `Sizes of tensors must match` error inside `softmax`. Not a math bug; worth a shape
   assertion.
3. **Fresh-batch recall is 0%** after 100 steps. Expected for a 17k-parameter model, but
   it means no recall claim can be made from this run.
4. **Cost is O(T·M·K³)** with a full Cholesky per state per token. At `d_key=128`,
   `M=8`, this is the dominant cost; the doc already states it.
5. **No `--resume`**: checkpoints store RNG state but the CLI cannot resume training.
   Documented in the handoff, confirmed.

## 7. Status against the handoff's two-level acceptance

**Software layer: PASS** — 54/54 tests, exact reproduction of the recorded diagnostic,
causal/streaming/SPD/gradient invariants verified independently.

**Research layer: NOT ESTABLISHED** (as the handoff itself states) — no native MQAR, no
long-context result, no evidence that more associations remain readable, and the tiny
diagnostics overfit and worsen held-out NLL.

## 8. Next step to compete with MARCH

The handoff's Stage B requires a **native MQAR adapter** before any comparison is fair.

**Implemented** (`mqar_adapter.py`, `train_mqar.py`, `tests/test_mqar.py`):

- Zoology pinned at `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`, generator
  `zoology/data/multiquery_ar.py`, vendored under `vendor/zoology`.
- **Aligned labels are used as-is**: `mqar_loss` calls
  `cross_entropy(logits, labels, ignore_index=-100)` and never shifts again.
- Both NumPy and **torch** RNGs are pinned before generation, because
  `random_non_queries` draws its filler from `torch.randint`.
- Every panel records a SHA-256 over inputs+labels; train/valid/test use independent
  seeds and disjoint draws.
- `check_alignment` verifies, for every labeled position, that `inputs[t]` is a key
  whose earlier prefix pair carries `labels[t]`, that key != answer, and that the label
  is not at position 0.

Tests: **63/63 pass** (54 original + 9 MQAR), including alignment, no-double-shift,
scoring denominator, seed determinism, torch-filler determinism, and panel disjointness.

### 8.1 CPU pilots (resource calibration, not results)

| protocol | steps | batch | s/step | final valid loss | valid acc | chance |
|---|---:|---:|---:|---:|---:|---:|
| vocab 8192, seq 128, KV 8 | 20 | 8 | 4.4 | 9.173 | 0.0% | 0.024% |
| vocab 128, seq 64, KV 4 | 200 | 16 | 0.57 | 4.224 | 1.86% | 1.56% |

Both are at chance on query positions. Loss is measured **only** at labeled positions
and matches `ln(vocab)` for the value distribution, which confirms the scoring path is
correct. These budgets are far below any MQAR paper (Zoology trains orders of magnitude
longer), so this is **not** evidence for or against the method.

### 8.2 What a fair MARCH comparison needs

1. GPU node (this host has none).
2. Matched token/step budget and matched parameter count between Branch Memory and MARCH.
3. Both methods on the **same panel hashes** (the adapter now emits them).
4. Report query-position accuracy/loss, forks, slot occupancy, and wall time.
5. Length extrapolation (256/512) and KV load (16/32) panels pre-registered before training.

