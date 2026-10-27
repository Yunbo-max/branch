# Shared backbone — fairness contract

All five models use **one identical backbone**. Only the sequence mixer differs. Any
difference in a result must be attributable to the memory mechanism, not to capacity,
data, or optimisation.

| field | value | applies to |
|---|---|---|
| `hidden_size` | 1536 | all |
| `num_hidden_layers` | 21 | all |
| `intermediate_size` | 4096 | all (SwiGLU, = FLA's `int(1536*4*2/3)` rounded to 256) |
| `max_position_embeddings` | 16384 | all |
| `vocab_size` | 32000 | all |
| `tie_word_embeddings` | false | all |
| `initializer_range` | 0.006 | all |
| `use_cache` | true | all |
| `num_heads` | 16 | Transformer only |
| `num_heads` / `head_dim` | 6 / 192 | linear-attention family (GDN, H-GDN, MARCH) |
| `routing_dim` / `anchor_interval` | 64 / 512 | MARCH only |

Training is also identical: 50B tokens of Long-Data-Collections, seq_len 16384, global batch
≈ 4.2M tokens, fused AdamW (β 0.9/0.95, ε 1e-8, wd 0.1), peak lr 4e-4, WSD schedule, same
tokenizer and seed.

## Resulting parameter counts

Analytic, from the architecture definitions:

| model | params |
|---|---:|
| Transformer (21L, 16 heads, FFN 4096) | ~693M |
| Gated DeltaNet (21L, 6×192) | ~644M |
| GDN + Log-Linear | ~same as GDN + log-linear overhead |
| MARCH (GDN + anchors/routing) | ~685M |

Parameter counts are **not** equal even with a fixed backbone, because the mixers differ:
attention costs `4·h²` per layer while GDN's projections cost `3·h·(6·192) + 192·6·h`, and
MARCH adds `d_r`-sized routing projections. This is the price of holding width/depth/data/
optimizer fixed. Report the counts next to the results; if a reviewer demands parameter
parity, the paper's own answer is the optional 24-layer Transformer
(`configs/paper/transformer_mid.json`, ~778M) — it is kept for exactly that check.

## Paper-matched configs

`configs/paper/` keeps the configs shipped with the reference protocol repository
(`HanGuo97/log-linear-attention`). They differ from the unified set in depth for the
24-layer Transformer and reproduce the paper's stated 693M / 778M. Use them only for
paper-reproduction runs, not for the controlled five-way comparison.
