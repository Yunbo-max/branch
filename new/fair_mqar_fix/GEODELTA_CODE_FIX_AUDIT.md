# GeoDelta / Branch causal-efficiency code repair

**Base:** `main` at `a407a3c` (2026-10-26)
**Patch branch:** `fix/geodelta-causal-efficiency-20261026`
**Scope:** `new/fair_mqar_fix/`
**Status:** source patches + regression tests committed; full repository pytest and H100 benchmarking are **not** verified in this session.

## 1. What changed compared with the original code?

1. `single_geometry.py`: Cholesky of a frozen full covariance geometry is made **lazily** for a streamed partial chunk. Fully batched chunks use their own `_chunk_step` factorization and do not additionally compute `_make_factor` before/after processing. A factor is materialized on the first streamed token of a chunk and reused until boundary.
2. `chunked_branch.py`: fork children receive free slots ordered by the **first token that requests each parent fork**, with deterministic parent-index tie-breaking. Old parent-index priority allowed a future fork request to preempt an earlier fork when free slots were scarce. The old API without `first_request` remains for direct compatibility; the model always supplies first-event timestamps.
3. `tests/test_causal_geometry_regressions.py`: add Cholesky-call-count tests, arbitrary-split single-state streaming/cache parity, gradient checkpoint invocation and gradient parity, certified-mode rejection, sparse-capacity priority and prefix-causality tests, plus 100 randomized comparisons with an online allocation oracle.

## 2. Why?

- With `chunk_len=256`, the previous full-geometry prefill computed one extra Cholesky before the first chunk and another after *every* complete chunk. For `T=16384`, that was 129 calls before backward recomputation versus the necessary 64 chunk factorizations. The patch removes only this redundant work; it does not claim an end-to-end GPU speedup without measurement.
- The old Branch allocator ranked parent ID rather than request time. With two active parents and one free slot, a fork requested at token 3 from parent 0 could force a token-0 request from parent 1 to replace, violating causal independence. The new priority is first-event time, so adding future events cannot preempt earlier ones.

## 3. Effects observed so far

- Mathematical allocator regression **RED** under old logic: future fork preempted earlier fork; reproduced with actual PyTorch tensors.
- The replacement first-event allocator matched a chronological online oracle in 100 small random trials and one independent B=4,H=4,M=64,L=256 batch with mixed active slots and fork requests.
- Full forward/backward/streaming tests **have been added** but **not executed against the full GitHub tree** in this environment. No new LM loss, H100 latency, memory, or quality data have been produced.

## 4. Agent verification commands (must run before merge)

```bash
git fetch origin
git checkout fix/geodelta-causal-efficiency-20261026
python -m pytest -q new/fair_mqar_fix/tests/test_causal_geometry_regressions.py
python -m pytest -q new/fair_mqar_fix/tests
python -m pytest -q new/branch_memory/tests
```

Run H100 prefill and decode profiling on identical inputs/configs before/after patch:
- `full`, `isotropic`, `rayleigh`; `chunk_len=256`, `seq_len=16384` and 513-token nonuniform streaming.
- Count `torch.linalg.cholesky` calls separately for forward, backward/recompute, and per-token decode.
- Check checkpointed vs non-checkpointed parameter/input gradients, peak allocated VRAM, and tokens/sec.
- Compare quality on the same fixed panel; fork implementation was changed, so old M=64 checkpoints are **not** valid apples-to-apples comparisons for the new allocation policy.

## 5. Limitations and next steps

- **Multi-state Branch still lacks chunk-persistent streaming execution.** Even though fork slot assignments inside a batched chunk now satisfy first-event causality, calling `ChunkedBranchMixer.forward` token by token does not reproduce the frozen full-chunk routing/state visibility semantics. Do not claim full prefill/decode parity for M>1. Implement dedicated per-token causal routing/fork state machine before deploying or claiming an autoregressive generator.
- The single-state `certified` mode remains explicitly disabled for causal LM because its chunk-max selection uses future tokens; the patch does not claim it is fixed.
- Existing MARCH anchor port is not changed; its approximate content-conditioned boundary descriptor should not be presented as exact official cross-layer MARCH.
- No H100 measurements from this code change. Keep this branch isolated until the regression suite and performance measurements pass.
