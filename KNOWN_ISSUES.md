# Known correctness issues (verified 2026-10-22)

These findings were reproduced on CPU against the code in this repository. They mean the
pretraining curves in `results/` are **legacy pilot data**, not evidence for the claims
they were originally collected for. Do not start another large run until P0-A and P0-B
are fixed and re-verified.

## P0-A — Branch fork allocation never allocates a child

File: `new/fair_mqar/chunked_branch.py`, `_chunk_step` (slot allocation) and
`initial_cache`.

`take` compares `rank_free` and `rank_fork` after inserting the two singleton axes on
opposite sides, so the equality is indexed `[free, parent]` while the masks
`free` / `any_fork` are broadcast on the transposed axes. With the default cache
(`active[..., 0] = True`) no assignment satisfies all three terms.

Reproduction:

```python
M = 8
active = torch.zeros(1, 1, M, dtype=torch.bool); active[..., 0] = True
any_fork = torch.zeros(1, 1, M, dtype=torch.bool); any_fork[..., 0] = True
free = ~active
rank_free = torch.cumsum(free.long(), -1) - 1
rank_fork = torch.cumsum(any_fork.long(), -1) - 1
take = ((rank_free.unsqueeze(-1) == rank_fork.unsqueeze(-2))
        & free.unsqueeze(-2) & any_fork.unsqueeze(-1))
print(int(take.sum()))   # 0 for M = 2, 4, 8, 64
```

Consequences, all reproduced with the real mixer:

- `active` stays at 1 for every `(batch, head)`; the 64-state run is a
  **single-active-state replace-only** model, not a 64-branch model.
- `stats["fork"]`, `parent`, `destination` and `write_cost` are hard-coded zeros, so they
  cannot be used as telemetry.
- If a child were ever allocated, the scatter index `g5` is `[B,H,M,1,1,1]` (6-D) and
  `.expand(B, 1, H, K, V)` (5-D) raises `RuntimeError`, so the allocation bug currently
  hides a second crash.
- Even with a correct allocation, forked tokens are written to the parent slot
  (`write_mask = onehot(parent)`) and the child is then overwritten with the chunk-start
  snapshot, so the forked tokens' writes never reach the child. The fork semantics need a
  separate fix, not a one-line index swap.

## P0-B — MARCH port has no content-conditioned anchor key

File: `new/fair_mqar/chunked_march.py`, `forward` (anchor creation).

Every boundary projects the same learned constant:

```python
a_hidden = self.anchor_embed.view(1, 1, d_model).expand(B, 1, d_model)
ak = self.anchor_k_proj(a_hidden)
```

There is no input, state descriptor, or boundary index in `a_hidden`. All created anchors
therefore have identical keys, both across slots and across different inputs.

Reproduction (tiny mixer, two chunks, two different inputs):

```
counts after 2 chunks: [2]
slots identical (input1): True
anchor_k identical across different inputs: True
```

Consequently the routing logits are tied across anchors for a fixed query; Top-K selects
by index, not by content. This port is **not** the content-conditioned router described
in the MARCH paper (arXiv:2608.12435); it must be labelled as a custom control, or the
anchor key must be derived from the per-anchor state/hidden descriptor.

## P0-C — Evaluation is not a fixed held-out panel

File: `new/fair_mqar/pretrain_branch.py`.

Both training and `evaluate()` call `PackedCorpus.sample` on the same corpus. The streams
have different seeds (`seed + 1` vs `seed + 999`), but there is no document-level split,
no saved manifest, and no tokenized hash. `eval_gen` advances on every call, so different
checkpoints are evaluated on different samples. Comparisons at the same step with the same
seed and the same eval schedule are still paired (the samples are identical across arms),
but this is not a fixed validation panel and does not rule out train/validation overlap.

## P1 — Chunked Branch ignores several variation knobs

File: `new/fair_mqar/fair_models.py` (chunked branch) and
`new/fair_mqar/chunked_branch.py` (`__init__`).

- `fork_cost` from `FairLMConfig` is not forwarded to `ChunkedBranchConfig`.
- `covariance` is hard-coded to `"full"` and `differentiate_covariance` to `True` in the
  base config passed to `BranchMemoryMixer`.
- Therefore CLI sweeps over these options do not change the chunked computation.

## Performance

`_chunk_step` expands every token to all `M` slots and calls the full `L x L` triangular
update for all slots; `write_mask` only zeroes write directions, it does not skip the
matrix build or solve. Inactive slots are included. `_route` and the writer also solve
`H^{-1}K` separately.

For `B=4, H=4, M=64, L=256` in FP32, one `[B*M,H,L,L]` tensor is 256 MiB, before other
intermediates and backward saves. This is an arithmetic size, not a measured peak.

## Provenance

- The v1 curve (`results/pretrain_full/`) was produced by an earlier version of
  `chunked_branch.py` (the file was modified at 2026-10-21 20:42, after the v1 job
  started). That source version is not archived in this repository, so v1 and v2 are not
  a controlled code comparison.
- The v2 curve (`results/pretrain_full_m64/`) was produced by the current replace-only
  code, with one active state per head.
- `results/pretrain_march/` was produced by the content-independent anchor-key port above.
