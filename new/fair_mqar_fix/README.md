# Fair MQAR comparison — four arms

One backbone, one dataset, one budget, matched parameters. **Only the memory mechanism differs.**

| arm | file | mechanism |
|---|---|---|
| `march` | `march_mixer.py` | evolving delta state + periodic anchor snapshots + content-routed Top-4 read |
| `branch` | `branch_memory/branch_memory/mixer.py` | Branch Memory: per-write parent choice, replace/fork, full-covariance protection, content reader |
| `delta` | `baseline_mixers.py` | single delta-rule state, no bank, no history read (Gated DeltaNet baseline) |
| `transformer` | `baseline_mixers.py` | causal MHA + RoPE with incremental KV cache |

Shared: backbone (embedding → N×(pre-LN mixer, pre-LN MLP) → norm → head), MQAR panels
(identical SHA-256), optimiser / lr / batch / steps / seed, and total parameter count
(auto-matched via `ff_extra`; residual gap printed).

**Fairness caveat:** the short depthwise convolution is applied in **all four** arms,
including the Transformer. The paper's Transformer baseline has no such convolution.
Keeping it makes the arms differ only in the memory mechanism, which is the point of this
harness — but it is a deviation from the paper's Transformer, so state it explicitly.

## MARCH mixer follows the released implementation, not the paper pseudocode

The official source is archived at `../MARCH/code/march_official/`
(HuggingFace `AI45Research/ReSI-Safin-1-35B-A3B`; see its `PROVENANCE.md`).
`march_mixer.py` reproduces, and `tests/test_march_parity.py` verifies:

1. **State layout `[B, M, H, K, V]`** and the read `Sᵀq`, scaled by `head_k_dim**-0.5`.
2. **Boundary visibility is inclusive**: anchor `m` is readable at token `t` iff
   `boundary_m <= t`.
3. **Route scaling**: `route_scale = head_k_dim**-0.5 * route_logit_scale`.
4. **Top-K** over valid anchors, released default **K = 4** (`hsd_read_topk`).
5. **Null option**: an extra candidate that takes softmax mass and is then dropped, so
   anchor weights sum to less than one. Routing calls the **official**
   `route_softmax_weights` imported from the archived source — not a re-derivation.
6. **Additive fusion**: `o = Sᵀq + history_scale · Σ_m w_m S_mᵀq`.

The base recurrence is the same delta update Branch Memory uses in its `delta` control,
so the comparison isolates the memory mechanism rather than the recurrence.

## Run

```bash
# paired run with automatic parameter matching
python compare.py --steps 500 --batch-size 16 --seq-len 128 --num-kv-pairs 8 \
  --vocab-size 8192 --d-model 128 --layers 2 --heads 2 --d-key 32 --d-value 32 \
  --d-route 32 --anchor-interval 16 --max-anchors 8 --read-topk 4 --max-states 8 \
  --differentiate-covariance --lr 1e-3 --threads 8 --output-root results/fair

# single arm
python train_fair.py --mixer march --steps 500 --output results/march_only

# parity + integration tests
python -m pytest -q
```

## Verification state

`python -m pytest -q` → **18/18 pass**:

- `test_march_parity.py` (10): the official routing helper is imported and used, null
  absorbs mass, Top-K bounds the read set, boundary inclusivity is exact, the read matches
  the official formula recomputed by hand, causality and streaming equivalence hold, no
  future gradient, anchor capacity respected, all four arms build in the shared LM.
- `test_baselines.py` (8): both baselines forward, stream and stay causal; the Transformer
  KV cache grows correctly; the delta arm holds a single `[B,H,K,V]` state.

**Bug found and fixed while building the harness:** the first Transformer arm attended over
the whole current chunk without a causal mask, so altering a later token changed earlier
outputs. `test_causality_and_streaming[TransformerMixer]` caught it; the fix applies
`tril(diagonal=offset)` over the cached+current keys.

CPU smoke, 150 steps, vocab 128 / seq 64 / KV 4 / d_model 64 / 1 layer:

| | MARCH | Branch Memory | DeltaNet | Transformer |
|---|---:|---:|---:|---:|
| mixer params | 12,674 | 11,650 | 8,384 | 16,576 |
| total params | 49,888 | 49,896 | 49,984 | 49,920 |
| valid loss / acc | 4.2290 / 1.22% | 4.2114 / 1.86% | 4.2254 / 1.51% | 4.1697 / 1.81% |
| test loss / acc | 4.2253 / 1.17% | 4.2192 / 1.37% | 4.2248 / 1.56% | 4.1739 / 1.81% |
| seconds/step | 0.103 | 0.531 | 0.038 | **0.008** |

Parameter spread 96 params (0.192%); panels identical. Chance is 1/64 = 1.56%, so **all
four arms are at chance** — this validates the harness, it is not a result.

## What is still missing for a publishable claim

1. **GPU.** This host has none; CPU is 0.008–0.53 s/step at toy scale.
2. **Budget.** MQAR papers train for far more steps; 150 steps shows nothing.
3. **Scale.** The paper's MARCH is 793M / 16K context; this harness is a small proxy.
4. **Multiple seeds** and per-seed variance, not a single run.
5. **Pre-registered extrapolation** (length 256/512, KV 16/32) before looking at results.
6. A **no-conv Transformer** variant, to check the shared-convolution choice does not drive
   the ranking.
7. **Memory-matched** comparison (bytes, not just parameters): attention's KV cache and
   Branch Memory's state bank cost very different memory per token.
