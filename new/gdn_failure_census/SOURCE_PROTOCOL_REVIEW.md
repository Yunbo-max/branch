# GDN diagnostic suite: source and protocol review

Reviewed: 2026-10-28. This note records inspected primary sources and a proposed evaluation protocol. It contains **no measured language-model result** and does not establish that a particular GDN checkpoint fails.

## 1. Identify the mechanism before interpreting a score

The original Gated DeltaNet paper defines, in its value-by-key state orientation,

\[
S_t=\alpha_t S_{t-1}(I-\beta_t k_t k_t^T)+\beta_t v_t k_t^T.
\]

For a key-by-value implementation, the equivalent update is:

```python
decayed = alpha * state
residual = value - key @ decayed
state = decayed + beta * outer(key, residual)
output = query @ state
```

The residual must use the **decayed** state. Replacing it with the undecayed state changes the operator. The original mechanism uses a scalar decay and scalar write gate per head; the vector-valued output gate is a separate part of the block. Queries and keys are L2-normalized in the paper's block. Its pure recurrent model and its attention-containing H1/H2 hybrids are distinct models. [S1]

The inspected current FLA layer projects decay/write inputs to `num_v_heads`, delegates normalization and gate transformations to kernels, and exposes options including `allow_neg_eigval`. Therefore save the exact implementation revision and resolved model configuration; a shared class name is insufficient to establish identical update conventions. [S3]

**Interpretation limit:** fixed artificial gates, normalized random keys and prescribed values test an operator, not a trained language model. For example, an erase-only probe can verify the product of prescribed decay factors, but it cannot tell us which decay factors a language model will learn. Similar-looking textual identifiers do not guarantee high cosine similarity of learned keys. That relationship requires actual state/key instrumentation.

## 2. Checkpoint availability and model identity

The official NVlabs repository FAQ explicitly states that the authors provide code implementations rather than original pretrained weights. It recommends converting trained weights to the Hugging Face format supported by FLA for evaluation, and warns that recall tasks are sensitive to prompt format for instruction-untuned models. It points to RULER for S-NIAH evaluation. [S2]

Do not invent a downloadable original NVlabs-paper checkpoint. Separate publicly released m-a-p checkpoints were subsequently verified for the recovered project (table below). A real run needs a supplied checkpoint or one of these separately inspected released models. Record the following before launching:

- Model ID/local path and exact weight revision or file hashes; tokenizer revision and chat template.
- Pure recurrent, hybrid, or full attention identity, including layer types/counts, head/state dimensions and relevant kernel settings.
- Training context exposure when documented, supported deployment context, precision and quantization.
- Base versus instruction-tuned status, prompt mode, generation settings and reasoning mode.

A hybrid model's task failure is evidence about that model. Comparing unrelated pretrained models cannot isolate the effect of GDN from training data, scale, tokenizer, alignment or attention layers. Establishing an architectural cause requires matched training or an explicitly justified controlled intervention.

### Verified m-a-p releases for the recovered repository

The actual research destination was recovered as `Yunbo-max/branch`, branch `exp/gdn-long-context-failure-census-20261028`, existing harness `new/gdn_failure_census`. Its three proposed m-a-p releases are public and list both weight and tokenizer files. Use the same pinned repository revision for weights and tokenizer:

| Model ID | Model and tokenizer revision | Inspected configuration |
| --- | --- | --- |
| `m-a-p/340M-20B-GatedDeltaNet-pure-baseline` | `14661a98768e1f5d5493ff7b18f2d6809717a4c8` | 24 layers, hidden 1024, 4 heads, head dimension 256, `attn=null`. |
| `m-a-p/1.3B-100B-GatedDeltaNet-pure` | `930ed6ae4ac629c86cb9855bb3dcb0a0974a29aa` | 24 layers, hidden 2048, 8 heads, head dimension 256, `attn=null`. |
| `m-a-p/1.3B-100B-GatedDeltaNet-hybrid-6-1` | `b2a4ecf3ee629e8ad5538d4ab18e5068b69b341f` | 24 layers; attention layers `[6,11,17,23]`; attention window `null`; RoPE theta 10000. |

All three inspected configs declare `max_position_embeddings=2048`, `qk_norm=l2`, `use_short_conv=true` and `model_type=gated_deltanet`. The inspected 1.3B-pure tokenizer is `LlamaTokenizerFast`, adds BOS and contains a very large `model_max_length` sentinel. The sentinel is not an established context capability. Configured length is not proof of actual training-length exposure. Model names suggest nominal training budgets but no model card was present on the inspected pages, so those names alone do not verify a matched training protocol. [S8–S10]

Inspection verified repository metadata/configuration and file presence, not successful native model loading or numerical parity with the original training implementation.

## 3. Diagnostic design and implementation checks

### Length, load and support location

Use two separate sweeps: hold information load fixed while changing length, and hold length fixed while changing load. Keep neutral filler, competing records and genuine updates distinct. Report requested and realized length, support token spans, query token span and support-to-query distance.

Count tokens **after** applying the full model template. Include BOS/special tokens and reserve output capacity. Never silently truncate evidence to fit a context window. If an input cannot be represented without losing a task record, mark it invalid/unsupported and preserve that row. Unsupported lengths and runtime OOMs are not incorrect answers.

Record both target position and actual position. A requested 95% position can be infeasible when many records must follow it; actual positions expose this instead of concealing a changed treatment. Position-dependent query ordering must also remain fixed or be independently varied.

### Paired controls

Generate a stable underlying instance for each comparison. Preserve its keys, values, question and expected answer across length/position/control variants, and retain a `base_case_id` for pairing.

- **Evidence-only:** retain every support fact needed to solve the query, while removing unrelated records and filler. Counting and history tasks may require several support records; keeping only the final answer-bearing string is not equivalent.
- **No-evidence:** remove all facts that reveal the target while preserving the query and distractor distribution. This is an answer-leakage/background-knowledge control, not a task where the original target remains derivable. Label its interpretation separately.
- **Short full context:** establish that the model can perform the underlying operation and follow the requested output format before attributing a long-context drop to memory.

For explicit-timestamp history tasks, query the value at a stated timestamp; otherwise changing presentation order can unintentionally change the correct answer. For overwrite tasks, genuine later writes must be part of the ground-truth semantics, rather than treated as supposedly harmless noise.

### Metrics and reproducibility

Save raw predictions and parsed outputs. Report exact sequence match, set match or scalar match according to the task, plus per-item recall when appropriate. Avoid accepting arbitrary substring matches: outputting every candidate must not score as perfect retrieval. Keep formatting failures distinguishable from missing values.

Save manifest/config hashes, code version, model/tokenizer identity, random seeds, backend identity, error status, input/output token counts and generation settings. A mock/oracle backend proves only that the harness is connected. Such rows must not appear in model-performance summaries.

Use case-paired comparisons. Length/position variants of one underlying instance are correlated; uncertainty for their differences should resample underlying instances, not treat every variant as an independent trial. Distinguish data-generation seeds from independently trained model seeds. Report denominators and failed/unsupported coverage with every score. Preselect confirmation contrasts or clearly label exploratory multiple comparisons.

### Runtime measurements

Separate model loading, warm-up, prompt processing and generation. Synchronize the accelerator around timings. Report peak allocated/reserved memory only with its measurement scope. Total `generate()` wall time is **not** time to first token; do not publish a TTFT field unless the first-token event is actually measured. Estimate a large sweep from representative measured cases and total planned tokens before running it.

## 4. A staged large test

These are design proposals for a broader follow-up suite, not completed runs or claimed power guarantees. The recovered repository's immediate implementation is the paired four-choice status-log census; do not describe this broader eight-family proposal as already integrated.

1. **Harness gate:** deterministic generator/oracle checks, token accounting, output scoring, resume/sharding identity and analytic operator checks. No model conclusion.
2. **Screen:** all eight task families at 2K, 8K and 32K, three support locations and 16 underlying examples per cell, with paired controls. Only run lengths the selected model supports. Add load/distractor sweeps separately to identify plausible mechanisms.
3. **Freeze:** inspect errors; select explicit contrasts; freeze prompts, scoring and confirmation seeds. Exclude tasks failing short/evidence-only competence from a pure memory-failure claim, but retain and report their outcomes.
4. **Confirm:** use unseen instances across a denser 2K–128K grid within supported context, five locations and at least 128 examples for each selected contrast. Determine final sample sizes from the effect size of interest and precision requirements, not the word “large.”
5. **Transfer:** run the official external benchmark implementations below, with pinned versions and unchanged scoring. Report them in separate tables from this custom suite.

## 5. External validation: referenced, not implemented here

| Benchmark | What the source provides | Appropriate use |
| --- | --- | --- |
| RULER | Configurable synthetic long-context evaluation; the official suite contains 13 tasks across four categories. [S4] | Validate that retrieval/multihop/aggregation findings survive official task construction and scoring. |
| LongBench v2 | 503 multiple-choice questions across six realistic task categories; its length statistics are in **words**, not model tokens. [S5] | Assess real document/code/dialogue reasoning. Re-tokenize each input, preserve official task identity and label any context filtering. |
| BABILong | bAbI reasoning facts embedded in PG-19 distractor text, with 20 reasoning tasks and released evaluation sets at multiple lengths. [S6] | Test whether a finding transfers from opaque key/value records to distributed natural-language facts. |

These sources are future external validation entry points. The custom generated families are not official RULER tasks, and this package must not imply that official benchmark runners or scores are already integrated.

The LongBench v2 project specifically cautions that task distributions differ by length group. Accordingly, a cross-group accuracy trend alone is not a controlled length effect; use within-instance interventions for causal length comparisons. [S7]

## Primary sources inspected

- **[S1]** Yang, Kautz and Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule*, arXiv v1, Eq. 8 and Section 3.3: https://arxiv.org/html/2412.06464v1
- **[S2]** Official NVlabs GatedDeltaNet repository, FAQ 3 and 5: https://github.com/NVlabs/GatedDeltaNet
- **[S3]** FLA GatedDeltaNet layer and reference implementation: https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py and https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py
- **[S4]** Official NVIDIA RULER repository: https://github.com/NVIDIA/RULER
- **[S5]** Bai et al. *LongBench v2*, ACL 2025 paper: https://aclanthology.org/2025.acl-long.183/ ; official code: https://github.com/THUDM/LongBench
- **[S6]** Official BABILong repository: https://github.com/booydar/babilong ; NeurIPS 2024 paper: https://proceedings.neurips.cc/paper_files/paper/2024/file/c0d62e70dbc659cc9bd44cbcf1cb652f-Paper-Datasets_and_Benchmarks_Track.pdf
- **[S7]** Official LongBench v2 project notes: https://longbench2.github.io/
- **[S8]** m-a-p 340M files/config and immutable commit: https://huggingface.co/m-a-p/340M-20B-GatedDeltaNet-pure-baseline/tree/main ; https://huggingface.co/m-a-p/340M-20B-GatedDeltaNet-pure-baseline/blob/main/config.json ; https://huggingface.co/m-a-p/340M-20B-GatedDeltaNet-pure-baseline/commit/14661a98768e1f5d5493ff7b18f2d6809717a4c8
- **[S9]** m-a-p 1.3B pure config, tokenizer and immutable commit: https://huggingface.co/m-a-p/1.3B-100B-GatedDeltaNet-pure/raw/main/config.json ; https://huggingface.co/m-a-p/1.3B-100B-GatedDeltaNet-pure/blob/main/tokenizer_config.json ; https://huggingface.co/m-a-p/1.3B-100B-GatedDeltaNet-pure/commit/930ed6ae4ac629c86cb9855bb3dcb0a0974a29aa
- **[S10]** m-a-p 1.3B hybrid config and immutable commit: https://huggingface.co/m-a-p/1.3B-100B-GatedDeltaNet-hybrid-6-1/blob/main/config.json ; https://huggingface.co/m-a-p/1.3B-100B-GatedDeltaNet-hybrid-6-1/commit/b2a4ecf3ee629e8ad5538d4ab18e5068b69b341f

Repository links above track `main` as inspected on the review date. Before any reproducibility claim, pin the actual external code and data revisions used by that run.
