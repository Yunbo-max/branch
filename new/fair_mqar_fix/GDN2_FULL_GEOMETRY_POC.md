# Protected GDN-2 — first runnable adapter (not a performance claim)

Status: **CPU algebra checked independently**; GPU integration/performance **not verified**.
No changes to existing Branch training paths. This research branch contains an
optional wrapper in `gdn2_full_geometry.py` and a local CPU parity test.

## Motivation and exact scope

Branch Single Full Geometry protects frequently queried directions using a
full SPD metric:

```
H = lambda * I + mu * EMA(q q^T)
u = H^{-1} k
S' = S + [u / (tau + k^T u)] (v - S^T k)^T
```

The fastest available GDN-2 implementations use a *different recurrence*:

```
S' = (I - k (b * k)^T) Diag(exp(g)) S + k (w * v)^T
```

where `b` is a key-channel erase gate, `w` a value-channel write gate,
and `g` a key-channel decay. Simply replacing a gate or using a dense
preconditioner does **not** implement exact original Branch geometry:
a key-side direction rotation is required.

This adapter uses a **mathematically exact restricted case**:
- Full SPD metric `H`, frozen within each geometry chunk;
- `L = cholesky(H)`;
- key/query input `L^{-1} k`, `L^{-1} q`;
- physical state mapped into `Z = L^T S` and back between chunks;
- tied scalar erase/write `b = w = 1/(tau + ||L^{-1}k||^2)`;
- `g = 0` (no per-channel decay);
- no in-kernel Q/K normalization; `beta_guard=False`;
- official `chunk_gdn2` called unchanged, with `scale=1.0`.

Thus the original *chunk-frozen Full Geometry writer* is preserved.
**It is not a claim that unrestricted full GDN-2 is equivalent to Branch.**
It also omits Branch's history reader, per-token Fork allocator and routing.

## Testing

Run CPU algebra and causality tests (only PyTorch / pytest needed):

```bash
cd new/fair_mqar_fix
python -m pytest -q test_gdn2_full_geometry.py
```

For real GPU execution install [NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2)
separately, with its CUDA/Triton/FLA dependencies. The upstream kernel is
NVIDIA Source Code License-NC; **do not copy NVIDIA sources into Branch**.

```python
from gdn2_full_geometry import full_geometry_via_gdn2

# q,k: [B,T,H,K], v: [B,T,H,V], on CUDA, same dtype.
# q/k may be normalized externally, but must NOT be renormalized
# inside the kernel after geometric whitening.
y, state, query_C = full_geometry_via_gdn2(
    q, k, v, chunk_size=64, lam=1.0, mu=8.0, tau=1.0,
    protection_decay=0.95,
)
# Streaming next call: initial_state=state, initial_protection=query_C.
```

## Why Predicted-Skip is a different signal

Predicted-Skip computes `rho=exp(-gamma*relu(cos(S_chunk^T q_prev,v)))`.
This is a *read-conditioned redundancy estimate*; it does not rotate the
key-side write direction. It can in principle scale only GDN-2's `w` or
both `b` and `w`, but these operations are **not equivalent**.
Scaling `w` alone can leave erase active without replenishing the value.
`predicted_skip_gate` is provided as a separate primitive; this adapter
does not apply it automatically and does not make a performance claim.

## Research decision gates

1. Confirm official CUDA `chunk_gdn2` parity, backward gradients, and
   chunk-to-chunk state remapping vs a serial Full Geometry oracle.
2. Benchmark end-to-end tokens/s and peak VRAM against official GDN-2
   with the same backbone and chunk/sequence sizes. Cholesky, triangular
   transforms and a Python outer chunk loop are **not free**.
3. Only if worthwhile, test (i) geometry-protection gains over GDN-2,
   and (ii) Predicted-Skip as separate write-only and coupled erase/write
   gate ablations. The latter is *not* strict Full Geometry by itself.
4. Full, untied GDN-2 channel-wise erase/write and decay do not generally
   commute with a dense moving SPD basis; supporting all of them with an
   exact old-Branch equivalence needs a different recurrence/kernel.
