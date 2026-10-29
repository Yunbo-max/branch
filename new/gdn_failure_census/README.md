# GDN long-context failure census — executable v2

**Update 2026-10-29:** The CPU/software pipeline is validated; `m-a-p/*` checkpoints FAIL strict stock-FLA 0.5.0 loading because 24 `model.layers.*.attn.D` tensors are unexpected. The selected `puigde/gated-deltanet-360M-15B-slimpajama` is downloaded to the project GPU host and has a compatible-looking config, but has **NOT passed a real FLA native forward**. H100 smoke remains blocked by reported queue congestion. The launcher now defaults to the puigde checkpoint, and adds a CPU-only local-weights preflight. Do not interpret CPU/operator or mock evidence as pretrained-model accuracy.

**2026-10-28. Code and CPU verification are implemented. No pretrained GDN/H100/H200 performance result is claimed.** This is a controlled four-choice diagnostic suite, plus separate numerical operator probes. It is not an implementation of the official RULER, BABILong or LongBench benchmarks.

The research question is: **after a pretrained GDN can perform a short lookup, which combinations of distance, write load, lexical interference, legitimate revision and query composition cause observable failure?** Start with the measurements, then choose an architectural intervention from the evidence.

## Quick start on the existing GPU host

From your clone of `Yunbo-max/branch`:

```bash
git fetch origin
git switch exp/gdn-long-context-failure-census-20261028
git pull --ff-only
cd new/gdn_failure_census

# Uses numpy only; no model download or GPU use.
python -m unittest discover -s tests -v

# Genuine pretrained checkpoint loading/forward smoke check.
CUDA_VISIBLE_DEVICES=0 STAGE=smoke bash run_h100.sh

# A smoke run is too small for a memory-mechanism claim.
CUDA_VISIBLE_DEVICES=0 STAGE=competence bash run_h100.sh

# Inspect the competence summary, then conduct the systematic screen.
CUDA_VISIBLE_DEVICES=0 STAGE=screen bash run_h100.sh
```

CPU tests need `numpy` (`python -m pip install -r requirements-cpu.txt`). Real evaluation additionally needs a mutually compatible **PyTorch/CUDA, Transformers, Accelerate, Hugging Face Hub and FLA** environment. Reuse the working GPU environment where possible; follow the [official FLA install instructions](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md) if it is missing. This change does not claim to have verified a new dependency combination or the native FLA forward on this CPU-only workspace. Runtime package versions are recorded and must match on resume.

A tokenizer-only preflight writes a separate plan without allocating model weights:

```bash
DRY_RUN=1 STAGE=screen bash run_h100.sh
```

The primary model is a base language model. Evaluation uses a fixed demonstration prompt and **actual next-token A/B/C/D logits**, rather than assuming instruction following or chat generation. All four labels must map to distinct single next tokens after the actual suffix. Full-vocabulary gold-token NLL and restricted four-choice log loss are recorded separately; neither is a full-document language-model perplexity.

## Fastest next executable check — native pretrained GPU forward

The CPU preflight above **does not** establish compatibility of the loaded weights with the installed FLA kernels. Before committing to the 108-case smoke, run exactly **one strict forward** (no long-text performance claim) on an available CUDA GPU:

\`\`\`bash
# In the project checkout; use the model already downloaded on the remote host.
cd new/gdn_failure_census
MODEL_DIR=/inspire/hdd/global_user/zhangjiaquan-253108540222/gdn_models/puigde_360M

# No GPU needed. Reject known FLAME-only attn.D tensors and verify local SHA.
python checkpoint_preflight.py --model-dir "$MODEL_DIR" --check-tokenizer \
  --expected-weight-sha256 603d6619216ea86d639e7c0c65ac37459e632cc5f80c1cba5c6068571162c1f7

# Needs CUDA and compatible FLA + Transformers; strict weights, ONE 512-token forward.
CUDA_VISIBLE_DEVICES=0 python native_smoke.py --model "$MODEL_DIR" \
  --device cuda --dtype bf16 \
  --out ../../results/gdn_failure_census/puigde_native_smoke.json

# Then a limited, real 2-group (12-case) census. Use a new output directory.
CUDA_VISIBLE_DEVICES=0 MODEL_ID="$MODEL_DIR" STAGE=smoke MAX_SAMPLES=12 \
  OUT_DIR=../../results/gdn_failure_census/puigde_real_smoke12 \
  bash run_h100.sh
\`\`\`

Any available **RTX 4090, H100, or H200** with working BF16 CUDA/FLA can run this 360M smoke. The path above is on the previously reported remote cluster host; if using another machine, copy or download the model and set \`MODEL_DIR\` accordingly. Never interpret the 4090 smoke runtime as H100/H200 throughput. The only architectural conclusions supported by this one-forward gate are that strict model loading works and the inference interface returns finite, correctly-shaped logits.

For a full scientific \`competence\` gate, run the fixed 512/1024 controls with >=20 independent seeds before interpreting long-context failure. GitHub CPU CI passing does not fulfill this criterion.

## What v2 changes

- Removes an answer shortcut: interfering values are independent uniform A/B/C/D draws. Earlier code always excluded the correct label; a model could find the missing label without reading the target. Revision values also include the original label, and changed/unchanged final labels are reported separately.
- Adds `no_target`: the target record is replaced by neutral tokens, with the same query, distractors and token budget. Hidden gold measures answer predictability/leakage, not successful retrieval.
- Uses the same **absolute target offset** across query types. The old suffix-dependent position calculation could reject entire alias/revision matched groups.
- Includes tokenizer special tokens/BOS once, keeps task records intact, and checks exact input length. Inputs are assembled as token segments; neutral filler may end at a partial sentence, but task records start on a newline and are never truncated.
- Streams long-token payloads one matched group at planning time and one example at evaluation time. Only compact case metadata remains in the plan.
- Pins weights and tokenizer independently; records source/dependency identities; refuses mismatched resume or duplicate results; retains error rows.
- Rejects missing/unexpected/mismatched checkpoint weights instead of silently evaluating partially initialized parameters.
- Requests last-position-only logits when the model explicitly supports `logits_to_keep`/`num_logits_to_keep`; otherwise retains the normal forward. This avoids materializing every vocabulary logit for supported implementations.
- Adds whole-group sample/token budgets, a cumulative soft wall budget, output locking, compatible shard merging, and seed-clustered statistics.
- Makes mock execution explicit and excludes its rows from scientific score tables.

V2 changes prompts, label draws and provenance. **Do not resume v1 output or combine v1/v2 results.** Use a fresh output directory; v1 history is preserved on GitHub.

## Which checkpoint is actually suitable?

**Recommended first GPU smoke:** [puigde/gated-deltanet-360M-15B-slimpajama](https://huggingface.co/puigde/gated-deltanet-360M-15B-slimpajama) (21 layers, 1024 hidden, 4 heads, 357.8M parameters, trained on 15B SlimPajama tokens, 2,048 configured context). The previous cluster log states that the safetensors weights have already been downloaded to:

```text
/inspire/hdd/global_user/zhangjiaquan-253108540222/gdn_models/puigde_360M
```

The model card provides standard `import fla.models` + `AutoModelForCausalLM` examples; use `trust_remote_code=False` with the registered FLA architecture. Its config and weight header are *candidates* for compatibility, not a substitute for a successful strict loading test.

**Prior planned m-a-p releases are NOT usable with the installed stock FLA 0.5.0**: they have `attn.D` per layer and `qk_activation/qk_norm/use_beta` settings from the FLAME training fork. We observed `unexpected_keys` on all three. This applies to `m-a-p/340M-20B-GatedDeltaNet-pure-baseline`, `m-a-p/1.3B-100B-GatedDeltaNet-pure`, and `m-a-p/1.3B-100B-GatedDeltaNet-hybrid-6-1`; do not silently ignore these weights or claim they were evaluated. To include them requires an exact compatible FLAME implementation and numerical parity validation, not a generic `strict=False` load.

**No-GPU local weight/header/tokenizer preflight (run before queueing H100):**

```bash
MODEL_DIR=/inspire/hdd/global_user/zhangjiaquan-253108540222/gdn_models/puigde_360M
python checkpoint_preflight.py --model-dir "$MODEL_DIR" --check-tokenizer \
  --expected-weight-sha256 603d6619216ea86d639e7c0c65ac37459e632cc5f80c1cba5c6068571162c1f7
```

This scans only safetensors **headers** (except optional streaming SHA-256), config and tokenizer. It does not load tensors into GPU memory, guarantee output correctness, or reserve H100.

**Genuine short H100 smoke (when available):**

```bash
CUDA_VISIBLE_DEVICES=0 MODEL_ID="$MODEL_DIR" STAGE=smoke bash run_h100.sh
```

The launcher auto-runs local CPU preflight; it can optionally enforce the known weight SHA-256 with `EXPECTED_WEIGHT_SHA256`. For an online checkpoint instead, `MODEL_ID=puigde/gated-deltanet-360M-15B-slimpajama` is the default and requires a functioning HF download/cache.

All 4K–32K tests with this 2,048-context model are **length extrapolation** and must be separated from within-context evidence. Later 1.3B pure/hybrid replication remains blocked until actually compatible pretrained checkpoints are found, loaded without missing/unexpected weights, and their training contexts documented.

## Implemented conditions

| Condition | Measurement | Interpretation |
| --- | --- | --- |
| `plain` | Retrieve one target after neutral filler | Distance/position and baseline task competence |
| `unrelated` | Add unrelated record writes | Information/write-load sensitivity |
| `collision` | Same value sequence, similar identifier prefixes | **Lexical interference proxy**; learned key overlap is not measured |
| `revision_earliest` | Read original status after genuine updates | Historical retention |
| `revision_latest` | Read final status from the same record history | Correct updating; not same-fact erasure |
| `alias` | Follow one alias to the target | Local composition followed by a long gap |
| `two_hop` | Follow two adjacent links | Local two-link composition, not distributed document reasoning |
| `no_target` | Remove the target but retain query and unrelated writes | Chance/leakage control; task deliberately unanswerable |

Matched groups share seed, initial target, target location and total input length. Unrelated/collision cases share the exact interfering **value sequence**, while identifier strings differ. Revision pairs share the record history but have different natural-language queries; differing suffix lengths can slightly change neutral filler/event offsets. Do not claim identical full input bytes.

Four axes are explicit: total length, target position, number of writes, and early/late write placement. This separates passive distance from activity between target and query. Requested/realized offsets are saved. A group is rejected in full if support cannot fit or its target location is unattainable; the manifest reports why. Rejected configurations are not counted as model mistakes or silently dropped from a matched comparison.

## Large-test stages and size

Counts below are **planned before feasibility filtering**, for one model and one event-placement setting. Pre-edit probes add forwards, not independent examples.

| Stage | Lengths | Seeds | Positions × write counts × conditions | Planned cases |
| --- | --- | ---: | --- | ---: |
| `smoke` | 512, 1K, 2K | 3 | 2 × 1 × 6 | 108 |
| `competence` | 512, 1K | 40 | 1 × 1 × 8 | 640 |
| `screen` | 512, 1K, 2K, 4K, 8K | 20 | 3 × 2 × 8 | 4,800 |
| `full` | 512, 1K, 2K, 4K, 8K, 16K, 32K | 48 | 3 × 3 × 8 | 24,192 |

`screen`/`full` explicitly opt into extrapolation; `smoke`/`competence` do not. Override with `ALLOW_EXTRAPOLATION=0` when only supported-context results are desired. Each stage prints the **actual planned forward-input tokens**, including requested pre-edit evaluation. Estimate GPU time from measured native pilot throughput at representative lengths; there is no fabricated H100 runtime estimate.

After initial screening, freeze a small set of contrasts, exact prompts, scoring, lengths and effect-size/precision targets. Confirm on a disjoint seed block (e.g. `SEED_START=100000`) instead of treating exploratory confidence intervals over thousands of cells as confirmatory evidence. The stage launcher accepts `LENGTHS`, `EVENTS`, `POSITIONS`, `CONDITIONS`, `SEEDS`, and `SEED_START` overrides.

```bash
# Same screen, writes near the query instead of immediately after the target.
CUDA_VISIBLE_DEVICES=0 STAGE=screen EVENT_PLACEMENT=late bash run_h100.sh

# Replicate with a larger pure model; inspect native smoke first.
CUDA_VISIBLE_DEVICES=0 STAGE=smoke \
  MODEL_ID=<verified-compatible-pretrained-1.3B-pure> bash run_h100.sh
CUDA_VISIBLE_DEVICES=0 STAGE=screen \
  MODEL_ID=<verified-compatible-pretrained-1.3B-pure> bash run_h100.sh

# Related attention-containing comparison.
CUDA_VISIBLE_DEVICES=0 STAGE=screen \
  MODEL_ID=<verified-compatible-pretrained-hybrid> bash run_h100.sh

# Larger descriptive census after competence, feasibility and resource checks.
CUDA_VISIBLE_DEVICES=0 STAGE=full PREEDIT=0 \
  MODEL_ID=<verified-compatible-pretrained-1.3B-pure> bash run_h100.sh
```

### Budgets and interruption

```bash
STAGE=screen MAX_INPUT_TOKENS=20000000 MAX_RUNTIME_SECONDS=7200 bash run_h100.sh
RESUME=1 STAGE=screen MAX_INPUT_TOKENS=20000000 MAX_RUNTIME_SECONDS=7200 bash run_h100.sh
```

`MAX_INPUT_TOKENS` caps the planned input volume, including pre-edit probes. `MAX_SAMPLES` rounds down to whole condition groups. The wall limit is cumulative over resumptions and is checked between groups; **one complete group's evaluation may overrun it**. It excludes model loading/tokenizer planning. A budget exit is status `partial`/exit 75, never completion. Increasing compute requires an explicit new plan/run; the launcher does not silently enlarge limits.

A stop-on-error is written as an error row before exiting. Resume preserves terminal errors and does not silently retry/replace them. For corrected-code/retried evaluation, use a new output and retain the failed run. A process interrupted while writing its final JSONL line is refused on resume until the original is preserved and the incomplete trailing line is reviewed. Plan/code/tokenizer changes, mixed mock/real results and duplicate rows are refused.

### Several GPUs

Use a separate process/output per GPU with disjoint seed ranges and otherwise identical settings. The merger refuses overlap and checks config, exact weight/tokenizer pins, inference code, loaded architecture/precision, and package versions. It preserves errors and partial coverage.

```bash
CUDA_VISIBLE_DEVICES=0 STAGE=screen SEED_START=1000 OUT_DIR=/absolute/path/run_a bash run_h100.sh
CUDA_VISIBLE_DEVICES=1 STAGE=screen SEED_START=1020 OUT_DIR=/absolute/path/run_b bash run_h100.sh

python merge_runs.py --inputs /absolute/path/run_a /absolute/path/run_b --out /absolute/path/merged
python summarize.py --run /absolute/path/merged
```

Launch the first two commands in separate terminals to run concurrently. Merge after the writers stop. Merged outputs are analysis artifacts, not resumable evaluations.

## Outputs and decision criteria

Each run stores `manifest.json`, compact `cases.jsonl`, append-only `results.jsonl`, `summary.json`, and `groups.csv`, `paired.csv`, `preedit.csv`, `revisions.csv` when applicable. Config, model/tokenizer SHAs, inference-code hash, dependency versions, planned/invalid/deferred/missing counts, errors, GPU peak allocated/reserved bytes, and latency are retained. Timing is a synchronized model forward plus score extraction, **not TTFT or full decoding throughput**. Cold initialization/kernel effects can affect the earliest measured case; inspect representative pilot measurements rather than extrapolating from only that first case.

A short-lookup competence pass requires the shortest eligible plain control at **≤1024 tokens**, within configured context, at least **20 independent data seeds**, and the seed-bootstrap 95% lower confidence bound at or above **60%**. A three-seed smoke can establish executability, not a scientific pass. Data seeds are not independently trained model seeds.

Inspect the paired effect `collision accuracy − unrelated accuracy`, the load effect `unrelated − plain`, history/update separation, position effects and no-target predictability. Confidence intervals resample **whole seeds**, keeping repeated positions/loads within the same seed together. OOM, unsupported inputs, missing cases and incomplete triplets remain separate from accuracy. Cellwise intervals are exploratory; selected findings need new-seed confirmation and relevant multiplicity control before strong claims.

The optional pre-edit read asks the initial fact just before interfering writes. It can estimate loss conditional on successful pre-edit retrieval for the same target; it is a **query insertion intervention**, not a measurement of the hidden recurrent state. `revision_latest` changes the target and is never called loss of that same fact.

Optional plots, after real evaluation:

```bash
python plot.py --run /absolute/path/run
```

Plotting needs matplotlib. Pooled length curves are descriptive: feasible load/position populations can differ by length. Use matched cells for mechanism claims.

## Separate numerical operator probes

```bash
python operator_probe.py --out /absolute/path/operator_probe.json
```

This uses numpy, never a trained model. It verifies the original scalar-gated recurrence in `[key,value]` orientation:

```text
decayed = alpha * state
state = decayed + beta * outer(key, value - key @ decayed)
```

Four probe families check decay alone, orthogonal writes, controlled correlated writes and memory load. Closed-form cases validate implementation. Artificial gates/random keys do not establish real language-model failures. Exact configurations and all operator results are labeled accordingly.

## CPU plumbing without downloads

```bash
python run.py --backend mock --device cpu --lengths 4096 --positions .2 \
  --events 8 --seeds 2 --preedit-probes --out /absolute/path/new_mock_run
python summarize.py --run /absolute/path/new_mock_run
```

Mock lengths are character-test units and responses are deliberately artificial. Summary output is `DIAGNOSTIC_ONLY` with empty scientific score tables; the merge tool rejects mock data. No GPU results should be inferred from passing CPU checks.

## What remains after this code update

1. Run native pretrained smoke and confirm the checkpoint exactly loads into the installed FLA implementation.
2. Establish short-context task competence and inspect no-target leakage, then run the large grid within the selected resources.
3. Replicate the strongest effects with the larger pure model and related hybrid, using disjoint confirmation seeds.
4. Inspect trained-model gates/keys/states on those failure cases to connect textual interference to an internal mechanism.
5. Validate selected findings with original [RULER](https://github.com/NVIDIA/RULER), [BABILong](https://github.com/booydar/babilong), and/or [LongBench v2](https://github.com/THUDM/LongBench) runners and their official metrics. These adapters are not integrated in this commit.

No training script, old model checkpoint, historical result, or proposed GeoDelta/MARCH mechanism is changed by this diagnostic update.
