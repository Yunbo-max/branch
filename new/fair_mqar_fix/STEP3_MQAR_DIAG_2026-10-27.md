# Step 3 MQAR diagnostics — corrected interpretation (2026-10-27)

## Grounded results

Same 2,000-step online MQAR setup, distinct train/validation/test examples.

| task | arm | validation accuracy | test accuracy | last train accuracy |
|---|---|---:|---:|---:|
| 2 KV, seq=64, chunk=64 | `delta` (fixed-beta Delta, not learned GDN) | 20.0195% | 21.2891% | 15.6250% |
| 2 KV, seq=64, chunk=64 | `single_geo` Full | 7.9102% | 8.4961% | 4.6875% |
| 2 KV, seq=64 | Transformer | 99.4141% | 99.1211% | 100% |
| 8 KV, seq=128 | `single_geo` Full, stop-gradient C | 0% | 0.0488% | 0% |
| 8 KV, seq=128 | `single_geo` Full, differentiate C | 0% | 0.0488% | 0% |

The Zoology generator used by `train_fair.py` samples values from `vocab_size//2 ... vocab_size-1`. At vocab_size=8192, **there are 4096 possible value symbols, not 48**; a uniformly random value guess is **1/4096 ≈ 0.0244%**. The previous "2% chance" statement was borrowed from a separate synthetic MQAR configuration with 48 values and does not apply here.

## Critical issue: the 2-KV experiment DID NOT TURN ON THE GEOMETRY

The actual `SingleGeometryMixer` initializes `C_0=0` and `S_0=0`. It freezes the geometry at the beginning of each `chunk_len` block. Here `seq_len=chunk_len=64`: exactly one full block, so the only geometry used is

    H_0 = lambda I + mu C_0 = lambda I
    a_t = k_t / (tau*lambda + ||k_t||²).

This is **exactly the isotropic update** for the whole training and evaluation sequence (floating-point implementation aside). The extra chunk-start historical read is likewise zero because `S_0=0`. The Full-C matrix is only accumulated AFTER the only block and cannot influence the sequence's loss.

**Therefore the observed Full-C-vs-plain-Delta gap (7.9% vs 20%) cannot prove that full Query geometry is counterproductive.** The DeltaNetMixer and SingleGeometryMixer have different parameters, architecture/initialization, computation and number of parameters. Comparing these models is an overall-baseline diagnostic, not a controlled intervention on C.

The new control is `run_mqar_geometry_isolation.sh`, which compares **the same SingleGeometryMixer with identical initial weights**:
- `chunk_len=64`: `full` vs `isotropic` should be numerically equivalent throughout the 64-token sequence, and similar training trajectories.
- `chunk_len=16`: only from the second chunk can `full` use a nonzero C; compare matched `full` vs `isotropic` here to assess whether Query geometry improves or hurts learning. `read_mode=current_only` removes the independent chunk-start history-read confound.

Regression `tests/test_mqar_geometry_activation.py` specifically checks first-chunk parity, gradients, and second-chunk geometry activity. Both new controls are **committed but not yet run on H100/H200**.

## The 8-pair layout also suppresses the intended protection signal

With `num_kv_pairs=8` and `seq_len=128`, the Zoology generator places **all eight KV writes in the first 16 positions**; the current `chunk_len=64` uses `C_0=0` throughout that initial writing phase. Thus Full Query Geometry cannot protect the *initial KV writes against one another*. The later chunks may use a nonzero C when processing queries/filler, but that is a different intervention.

The MQAR placement distribution is weighted toward early query positions (`power_a=.01`). A Monte Carlo simulation of the generator's without-replacement `gap` sampling (10,000 rows, seed=17, 8 gaps from 56 candidate positions) places approximately **76.5% of query labels before token 64**. This is a simulation of the documented sampling law, not a direct statistic computed from the uploaded fixed panels.

To engage C during the original KV writing phase, use `chunk_len=8` for 8 pairs, or interleave new KV writes with prior queries across geometry boundaries. The new `run_mqar_geometry_isolation.sh` supports `SEQ_LEN=128 NUM_KV_PAIRS=8 CHUNKS="64 8"`. Compare Full against Isotropic **within the identical SingleGeometryMixer architecture and seed**, rather than only against the independently initialized DeltaNetMixer.

## Second issue: the 8-pair failure does not prove an information-theoretic capacity limit

A single state `S in R^{d_k x d_v}` can in principle represent `m<=d_k` independent key-to-value associations exactly. If `K in R^{d_k x m}` has full column rank, the linear constraints `S^T K=V` have a solution `S^T=V(K^TK)^{-1}K^T`. Thus with `d_k=64` and `m=8`, there is no immediate algebraic impossibility. The learned update dynamics, representation of token keys/values, initialization, missing trainable GDN gates, positional selectivity, or optimization could prevent training from discovering that solution.

In particular, the `delta` baseline here is a **fixed scalar-beta Delta Rule**, not a fully optimized and learnable Gated DeltaNet. The `single_geo` here likewise has no trainable main-memory gate. Do not extend this negative result to all GDN variants.

## Correct next tests

1. Run matched Full-vs-Isotropic controls with `chunk_len=64` (equivalence check) and `chunk_len=16` (real geometry intervention) on 2-pair online MQAR.
2. Repeat with `chunk_len=16/32`, `read_mode=current_only`, and multiple seeds; record actual `C_b` norms/eigenvalues, `q^T a`, residual norms, training gradients and validation accuracy.
3. Only if geometry is active and beneficial compare with a genuine learnable-GDN baseline, then decide whether to integrate Gated Woodbury or train 0.79B.
4. Do not infer anything about long-context performance or 27B/100B scaling from these short-sequence 2,000-step results.

**Conclusion:** online MQAR is a genuine negative result for the tested recurrent training configurations. The prior stronger claims "full C harms learning" and "8-KV exceeds the single-state's fundamental capacity" are **not supported by these experiments**.
