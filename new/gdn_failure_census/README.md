# GDN Long-Context Failure Census — pre-trained, paired and mechanistic

**2026-10-28 · status: executable research harness committed; no H100/H200 large-model benchmark has been run or claimed by this repository branch.** Run the CPU tests and the \`smoke\` stage before any long sweep. This is a **diagnostic benchmark**; it is not a new model architecture or the official RULER benchmark.

## The question

Before inventing a memory enhancement for Gated DeltaNet, identify which failure modes are present in a **real pre-trained GDN**:

1. **Simple retention**: can the network retrieve a fact from the same text after longer innocuous filler?
2. **Memory load**: do many unrelated facts reduce retrieval?
3. **Address collision**: are similar-but-distinct record keys more damaging than unrelated keys at *the same token length, answer and target location*?
4. **Memory overwrite vs legitimate revision**: after a repeated-key update, can the model distinguish the earliest and latest value?
5. **Lost in the middle / location**: same target fact at early, middle and late positions.
6. **Composition**: can the model retrieve through an alias (one/two links), provided it already passes simple retrieval?
7. **Direct pre/post edit**: when asked *immediately before* a batch of writes, could the model answer the old fact, and does that exact fact then become inaccessible?

The aim is **not** to prove that all failures are due to a state bottleneck. Distinguish task understanding, training-context extrapolation, attention architecture, state memory interference and token-level scoring artifacts.

## Real pre-trained models

| Arm | Checkpoint | Function | Caveat |
| --- | --- | --- | --- |
| GDN-340M (first) | [m-a-p/340M-20B-GatedDeltaNet-pure-baseline](https://huggingface.co/m-a-p/340M-20B-GatedDeltaNet-pure-baseline) | Fast screening | 20B training tokens; config head_dim=256, context=2048 |
| GDN-1.3B (replication) | [m-a-p/1.3B-100B-GatedDeltaNet-pure](https://huggingface.co/m-a-p/1.3B-100B-GatedDeltaNet-pure) | Confirm at more capacity/compute | 100B tokens; config max_position_embeddings=2048 |
| Hybrid-GDN (control) | [m-a-p/1.3B-100B-GatedDeltaNet-hybrid-6-1](https://huggingface.co/m-a-p/1.3B-100B-GatedDeltaNet-hybrid-6-1) | Compare an attention-augmented model using related training series | Not a strict architecture-only causality claim |
| Optional other | User-provided valid HF checkpoint path | Compare transformer/Mamba/local MARCH under same prompts | Different pretraining/tokenizer confounds; report explicitly |
| GDN-2 | [NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2) | Future extension after verified trained checkpoint and conversion | Official code/paper alone is NOT a pre-trained model; do not substitute random weights |

**Important:** pre-trained GDN above has config \`max_position_embeddings=2048\`; 4K/8K/16K/32K runs are deliberately **extrapolation**. They do NOT demonstrate forgetting versus a model proven to support that context. The 512/1024/2048 controls are mandatory. Compare GDN pure against hybrid within the same 1.3B/100B series when possible. FLA GDN-2 is available, but a freshly initialized model MUST NOT enter zero-shot benchmark tables.

## Probe design

The suite is synthetic but *tokenizer-exact*. Each case has a unique pseudorandom record key, one status from A/B/C/D, a formatted log and a query. The same record, answer, needle position and token budget are reused across a matched triplet:

- \`plain\`: no additional KV-like records after the target, only neutral filler.
- \`unrelated\`: same number of irrelevant records with different identifiers.
- \`collision\`: same number of irrelevant records with the SAME key prefix but distinct complete identifier and conflicting status.

Additional matched variants:

- \`revision_earliest\` and \`revision_latest\`: same intervening repeated-key history, different temporal query (old versus new).
- \`alias\` and \`two_hop\`: composition with target status in context.
- \`event_placement=early|late\`: whether the interfering records arrive immediately after the target or just before the query.
- \`positions\`: needle position fractions 0.12/0.50/0.88.
- \`events\`: 8/32/64 distracting writes.
- \`lengths\`: 512/1K/2K/4K/8K/16K/32K tokens.

The generator constructs **exact input token IDs** and fills remaining slots with seeded, neutral sentences. It never truncates a prompt to meet budget. It rejects impossible short-length/high-event combinations. Because interventions contain different words, this is a *paired causal contrast*, not identical token-level input statistics. Every case includes an ID, pair ID, expected answer, exact token length, needle offset, event offset, seed, model ID, tokenizer and prompt hash.

### Scoring — no chat models required

The provided GDN checkpoints are *base* language models. Use a small, fixed two-example demonstration; then score one next-token response constrained to \`A/B/C/D\` by the actual model logits. This avoids variation in free-form decoding, chat templates and generation stopping. Check that each answer label is represented by precisely **one distinct next token** under the **actual tokenizer and suffix**, otherwise FAIL instead of scoring an unrelated ID.

Report both restricted choice accuracy (random baseline 25%), restricted log loss, and the full-vocabulary gold-token NLL. They answer different questions. No training or parameter update occurs.

### Crucial baseline-competence gate

**If the same model fails \`plain\` at 512 tokens (< 60% default), do not call low 8K performance “GDN memory erased.”** The result then demonstrates inability to follow the particular status-lookup format, tokenizer errors, weak model reasoning, or other confounds. First improve prompt validity or choose a benchmark appropriate for pretrained base LMs.

### Direct pre-edit probe

With \`--preedit-probes\`, the evaluator additionally queries a fact **before** the subsequent distractor/revision records are inserted. It scores precisely the *initial* fact at that point. The report estimates:

\[
P(\mathrm{postwrong}\mid\mathrm{precorrect})
\]

per length/condition/event count. This is stronger evidence than a simple near-vs-unrelated drop; however it is an explicit query-insertion intervention, **not a readout of GDN internal state**. On revision tasks, post-edit evaluation against the latest value tests update behavior, not arbitrary “damage.”

## Install / preflight

On the H100/H200 host, with compatible CUDA, PyTorch, Triton and pre-existing FLA:

\`\`\`bash
git fetch origin
git switch exp/gdn-long-context-failure-census-20261028
git pull --ff-only
cd new/gdn_failure_census

python -m pytest -q tests/test_census.py

python -c "import torch; import transformers; import fla.models.gated_deltanet; print(torch.__version__, torch.cuda.get_device_name(0))"
\`\`\`

If FLA is missing, follow the [official FLA installation guide](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md) and verify versions against your installed torch stack. Current FLA CUDA distribution uses \`pip install flash-linear-attention[cuda]\` but **do not blindly replace your CUDA/PyTorch environment**. Use a dedicated environment if necessary. Loading a full model can require several GB of disk and VRAM. Never use \`trust_remote_code=True\` for model loading here.

Optional low-cost manifest-only preflight (downloads tokenizer, not model weights):

\`\`\`bash
DRY_RUN=1 STAGE=smoke bash run_h100.sh
\`\`\`

## Run stages (one GPU per process)

**Stage 0 — smoke**, at most 90 paired probe contexts (invalid short-length/high-load or unattainable needle-position cells are skipped as complete groups), with pre-edit interventions:

\`\`\`bash
CUDA_VISIBLE_DEVICES=0 STAGE=smoke bash run_h100.sh
\`\`\`

**Stage 1 — systematic GDN-340M screening**, a grid of 5 lengths × 20 seeds × 3 positions × 2 event counts × 7 conditions = **4,200 planned cases** (infeasible short-length/high-load cells are omitted as complete matched groups, so the executed count may be lower), plus pre-edit measurements where requested:

\`\`\`bash
CUDA_VISIBLE_DEVICES=0 STAGE=screen MODEL_ID=m-a-p/340M-20B-GatedDeltaNet-pure-baseline bash run_h100.sh
\`\`\`

**Stage 2 — GDN-1.3B**, replicate the identified stress strata; do NOT automatically run every 32K cell before 4K smoke passes:

\`\`\`bash
CUDA_VISIBLE_DEVICES=0 STAGE=screen MODEL_ID=m-a-p/1.3B-100B-GatedDeltaNet-pure SEEDS=12 LENGTHS=512,1024,2048,4096,8192 bash run_h100.sh
\`\`\`

**Stage 3 — attention hybrid control**:

\`\`\`bash
CUDA_VISIBLE_DEVICES=0 STAGE=screen MODEL_ID=m-a-p/1.3B-100B-GatedDeltaNet-hybrid-6-1 SEEDS=12 LENGTHS=512,1024,2048,4096,8192 bash run_h100.sh
\`\`\`

**Stage 4 — high-stress census** (not the first action; 7 context lengths through 32K × 48 seeds × 3 positions × 3 event counts × 7 conditions = **21,168 planned cases**, before rejecting infeasible short-context/high-load or invalid needle-position cells, plus optional pre-edit). Use only if the target checkpoint is able to run the length, short-context task is solved, GPU memory is sufficient, and you have defined a compute budget:

\`\`\`bash
CUDA_VISIBLE_DEVICES=0 STAGE=full MODEL_ID=m-a-p/1.3B-100B-GatedDeltaNet-pure PREEDIT=0 bash run_h100.sh
\`\`\`

Every stage writes under \`results/gdn_failure_census/<model>_<stage>_s<seedstart>_n<nseeds>/\`: \`manifest.json\` (including counts of skipped infeasible grid cells), \`cases.jsonl\`, \`results.jsonl\`, \`summary.json\`, \`groups.csv\`, \`paired.csv\`, \`preedit.csv\` and \`revisions.csv\`. The exact checkpoint and schema are recorded. The launcher uses \`set -euo pipefail\` so it will stop on a model/evaluation error. To resume an interrupted identical run:

\`\`\`bash
RESUME=1 STAGE=screen bash run_h100.sh
\`\`\`

For multiple GPUs, shard *seeds* using \`SEED_START=1000,1020,...\` and different \`OUT_DIR\` locations, then analyze each shard separately (a merge script can be added once GPU tests pass). Do not treat shards from different tokenizers as identical token positions.

## What would constitute meaningful evidence?

1. Short plain lookup >=60% accuracy in a base model, with no tokenization errors.
2. Random/unrelated controls remain measurably stronger than near-key collision under equal token budgets.
3. **Paired CI** for \(\Delta_{\mathrm{collision}}=\mathrm{Acc}(\mathrm{collision})-\mathrm{Acc}(\mathrm{unrelated})\) excludes 0, preferably at both 340M and 1.3B, within 2K first.
4. Cases correct before the interfering writes fail after them, and errors track interference load and key similarity, not just elapsed distance.
5. Hybrid GDN mitigates a specific GDN-pure failure on the same high-level setup, while being explicit about architectural and pretraining differences.
6. A legitimate update is assessed separately: “earliest” should preserve history and “latest” should reflect the most recent assertion. Avoid a mechanism that blindly restores obsolete values.

**Do not** jump from “8K NIAH failed” to “we need a history bank.” Once a reproducible failure type emerges, inspect gate/state behavior on a small sample of trained GDN, then choose the *smallest* architectural intervention and compare with MARCH and HOLA under a fixed memory and compute budget.

## External benchmarks for validation

Our cases are not official benchmark scores. Separately replicate the **exact official tasks** before making performance claims:

- [NVIDIA/RULER](https://github.com/NVIDIA/RULER): S-NIAH, MK-NIAH, variable tracing, common words, aggregation; key tests: 2K/4K/8K/16K.
- [NoLiMa](https://github.com/adobe-research/NoLiMa): lexical-disjoint semantic retrieval.
- [BABILong](https://github.com/booydar/babilong): reasoning over distributed facts.
- [HELM Long Context](https://crfm.stanford.edu/helm/long-context/latest/): standard tasks and evaluation protocol.
- [MARCH](https://arxiv.org/abs/2608.12435): a reference architecture storing *cumulative* state snapshots; do not compare an untrained local approximation as if it were the trained original.

The official benchmark wrappers are **not yet integrated** here. Those evaluations require their own original datasets, prompts, score rules and checkpoint loading. No official RULER score can be claimed from this census.

## Research integrity and compute control

This branch uses only publicly described model IDs; no checkpoint weights are bundled. \`--allow-extrapolation\` is explicit and stored in the manifest. A 32K failure could be due to context extrapolation or model kernel limitations, not necessarily memory capacity. Report OOM/unsupported-context separately from accuracy. Each suite runs on one GPU; 4,200×4K FLA forwards may be expensive, and optional pre-edit scoring adds extra forwards.

The branch is deliberately independent of GeoDelta Full-C and changes **no existing training code or results**. A successful census should identify a concrete failure to justify an architecture, not be used to retrofit a claim that a pre-selected new memory design works.


### Optional plots

After a valid run and \`summarize.py\`, install \`matplotlib\` in a separate evaluation environment if it is not present and run:

\`\`\`bash
python plot.py --run ../../results/gdn_failure_census/m-a-p__340M-20B-GatedDeltaNet-pure-baseline_smoke_s1000_n3
\`\`\`

Creates \`retrieval_accuracy.png\` (length × task) and \`paired_collision_effect.png\` (same-length near-key collision penalty). No plotted value is fabricated; files only exist after real GPU model evaluation.

**CPU test evidence:** GitHub Actions workflow \`.github/workflows/gdn-census-cpu.yml\` exercises \`tests/test_census.py\` without installing \`torch\`, downloading any checkpoint or touching remote accelerators. Green CI means the deterministic probe/analysis plumbing passes; it is not a pretrained GDN benchmark result.
