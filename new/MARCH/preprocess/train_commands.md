# Training commands (FLAME)

All five models share the paper's optimisation recipe:

```
fused AdamW, beta=(0.9, 0.95), eps=1e-8, weight_decay=0.1, peak lr 4e-4,
WSD schedule, seq_len 16384, global batch ~= 4.2M tokens -> ~11,900 steps for 50B tokens
```

With `NGPU=8`, `seq_len=16384`, `varlen`, `batch_size=1`:
`gradient_accumulation_steps = 4.2e6 / (8 * 16384) = 32`.

## Baseline: Gated DeltaNet (793M)

```bash
cd code/log-linear-attention
NGPU=8 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/gdn-793M-16K-50B \
  --model.config configs/flame/gated_deltanet_mid.json \
  --model.tokenizer_path <tokenizer-32k> \
  --optimizer.name AdamW --optimizer.eps 1e-8 --optimizer.lr 4e-4 \
  --optimizer.weight_decay 0.1 --optimizer.betas 0.9 0.95 \
  --lr_scheduler.decay_type wsd --lr_scheduler.warmup_steps 512 \
  --lr_scheduler.decay_ratio 0.2 \
  --training.batch_size 1 --training.seq_len 16384 --training.context_len 16384 \
  --training.varlen --training.gradient_accumulation_steps 32 \
  --training.steps 11900 \
  --training.dataset togethercomputer/Long-Data-Collections \
  --training.dataset_name pretrain --training.dataset_split train --training.streaming
```

Swap `--model.config` for:

| Model | config |
|---|---|
| Transformer | `../../configs/transformer.json` |
| Gated DeltaNet | `../../configs/gated_deltanet.json` |
| GDN + Log-Linear | `../../configs/h_gated_deltanet.json` |
| **MARCH** | `../../configs/march.json` |
| Optional param-matched Transformer (24L) | `../../configs/paper/transformer_mid.json` |

Every model uses the same `configs/common.md` backbone: hidden 1536, 21 layers, FFN 4096,
16K context, vocab 32000, untied embeddings, same data/token budget/optimiser/seed. Only the
sequence mixer differs — that is the fairness contract.

## MARCH

`MARCHModel` is a reference `nn.Module`. To train it in FLAME you must either:

1. wrap it as an `fla`-compatible model registered in `fla.models`, or
2. replace `GatedDeltaRuleReference` in `march/recurrence.py` with FLA's fused chunkwise
   kernel and keep the anchor/routing code, or
3. train it with a plain PyTorch loop first (slow, correct) to validate the architecture.

Option 2 is what the paper does; the producer must emit states at anchor boundaries, which
FLA's public chunkwise API does not expose. Plan for a custom Triton kernel or a
chunk-boundary-only state materialisation.

## Tokenizer

The paper uses vocab 32000 but does not name the tokenizer. Whatever you choose must be
recorded and used identically for all five models.
