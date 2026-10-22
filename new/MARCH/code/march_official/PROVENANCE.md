# Official MARCH implementation — provenance

Source: **HuggingFace model repository** `AI45Research/ReSI-Safin-1-35B-A3B`
(fetched 2026-10-19 via `/raw/main/<file>`). There is **no standalone GitHub repository**
for MARCH: `AI45Lab/Safin-1` is a landing page ("code coming soon") and `AI45Lab/ReSI`
is the ReSI alignment project, not the MARCH source.

Files here are the non-weight Python/config files from that repository:

| file | role |
|---|---|
| `march_gdn.py` (104 KB) | `Safin1MoeGatedDeltaNet` — MARCH over Qwen3.5-MoE GDN |
| `fused_history_read.py` (23 KB) | Triton fused reader (`fused_decode_top4_history_read`, `fused_prefill_top4_history_read`) |
| `fused_march_segments.py` (27 KB) | chunked GDN with chunk/anchor states |
| `fused_march_varlen_segments.py` (42 KB) | variable-length variant |
| `ops.py` | `route_softmax_weights` (masked softmax + null option) |
| `modeling_safin1*.py`, `configuration_safin1*.py` | model assembly |
| `config.json`, `model_manifest.json` | 35B-A3B configuration |

License: the linked ReSI repository is MIT. The HF model card does not state a separate
license; treat this as MIT and cite the paper (arXiv:2608.12435).

## Key differences from the paper description and from our `code/march/` reference

1. **Sparse, not dense.** `hsd_read_topk = 4` in the released configuration.
   The paper describes a fused *dense* implementation and a Top-K variant; the released
   model runs Top-4.
2. **State orientation is `[K, V]`.** The read is
   `einsum("bthk,bthmkv->bthmv", read_q, selected_states)`, i.e. `S^T q`.
   Our `code/march/` reference uses `[V, K]` and `S q`. Any port must fix the layout.
3. **Route scaling.** `route_scale = hsd_route_head_dim**-0.5 * hsd_route_logit_scale`
   (both `hsd_use_route_scale=True`, `hsd_route_logit_scale=1.0` by default).
4. **Null option.** `route_softmax_weights` masks invalid anchors with `-inf`, converts
   NaN to zero, renormalises, and multiplies by the validity mask. The null gate is a
   separate projection (`null_gate_proj`) plus a learned bias.
5. **Causality.** `boundary_counts` (per-row anchor boundaries) gate visibility;
   an anchor is readable only for token indices at or after its boundary.
6. **Fused reader.** Triton kernels exist for decode (block length 1, top-k = 4) and
   prefill; the PyTorch fallback is in the same `_history_read` path.

## What this means for reproduction

- The paper's 793M academic model is **not** the released model: this is a 35B-A3B MoE.
- For a scale-matched comparison against Branch Memory, port the memory mechanism from
  `march_gdn.py` into the 793M harness, preserving items 1–5 above and recording that
  the port is a re-implementation, not the released checkpoint.
- Do not call the 35B Safin-1 model "the paper's MARCH checkpoint".
