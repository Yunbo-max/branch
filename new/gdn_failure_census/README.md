# GDN long-context failure census — executable v2

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

## Verified checkpoint identities

These are public **m-a-p releases**, separate from the original NVlabs code release. The checkpoint names contain nominal scale/training-budget labels; no identical training protocol is inferred from their names. The inspected model pages lack a full model card.

| Arm | Model ID | Immutable model/tokenizer revision |
| --- | --- | --- |
| Initial screen | `m-a-p/340M-20B-GatedDeltaNet-pure-baseline` | `14661a98768e1f5d5493ff7b18f2d6809717a4c8` |
| Scale replication | `m-a-p/1.3B-100B-GatedDeltaNet-pure` | `930ed6ae4ac629c86cb9855bb3dcb0a0974a29aa` |
| Attention-containing comparison | `m-a-p/1.3B-100B-GatedDeltaNet-hybrid-6-1` | `b2a4ecf3ee629e8ad5538d4ab18e5068b69b341f` |

`provenance.py` uses those revisions by default. All three inspected configs specify `max_position_embeddings=2048`. **4K–32K are explicit configured-context extrapolation tests.** That configuration is not independent proof of training-length exposure; the tokenizer's huge maximum-length sentinel is not a context capability claim. Pure configs have `attn=null`; the hybrid includes attention at layers `[6,11,17,23]`. It provides a descriptive comparison, not isolated causal identification of GDN versus attention.

Custom HF repositories/local checkpoints are accepted through `MODEL_ID`/`--model`. Remote branch names resolve to exact commit SHAs; local checkpoint/tokenizer files are hashed (this can take time for large weights). Model and tokenizer must match the implementation. No random checkpoint is substituted on a load error, and remote model code is not trusted automatically.

Sources and limitations: [SOURCE_PROTOCOL_REVIEW.md](SOURCE_PROTOCOL_REVIEW.md).

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
  MODEL_ID=m-a-p/1.3B-100B-GatedDeltaNet-pure bash run_h100.sh
CUDA_VISIBLE_DEVICES=0 STAGE=screen \
  MODEL_ID=m-a-p/1.3B-100B-GatedDeltaNet-pure bash run_h100.sh

# Related attention-containing comparison.
CUDA_VISIBLE_DEVICES=0 STAGE=screen \
  MODEL_ID=m-a-p/1.3B-100B-GatedDeltaNet-hybrid-6-1 bash run_h100.sh

# Larger descriptive census after competence, feasibility and resource checks.
CUDA_VISIBLE_DEVICES=0 STAGE=full PREEDIT=0 \
  MODEL_ID=m-a-p/1.3B-100B-GatedDeltaNet-pure bash run_h100.sh
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
