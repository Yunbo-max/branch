# Reproduction contract: Branch Memory full-covariance v0.2.0

Read the parent `BRANCH_MEMORY_FULL_COV_CLAUDE_REPRODUCTION.md`, especially sections 0–14.
This directory is the reference implementation restored from that single file.

First reproduce; do not silently redesign.

- S is [B,H,M,V,K]; C is the full [B,H,M,K,K] causal, uncentred second moment.
- Main run: `--mode branch --covariance full --differentiate-covariance`.
- C is updated AFTER the current write/read. The parent/action costs see only earlier C.
- Replace uses ridge I + protect C; fork uses its isotropic objective + fork cost.
- Single hard parent/action in the forward pass. No hidden soft average or straight-through.
- Current destination is excluded from the historical residual reader.
- Fork preserves the parent matrix on that step, resets child C, and adds an active slot.
- Physical capacity is fixed; full pool replaces. No shrinkage, eviction, global KV or CED.
- Do not use future labels to route reads. Latent current v is a write target, not an answer oracle.
- Do not detach states inside a full sequence. Geometry gradient is an explicit flag.
- The `delta`/`snapshot` controls are NOT production GDN or official MARCH.
- Test suite: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q` (54 reference cases).
- Existing benchmark numbers are tiny fixed-batch OVERFITTING diagnostics, not positive findings.
- Native MQAR integration is a separate, clearly named stage. Its labels are already aligned;
  use CE(logits, labels, ignore_index=-100), not the byte LM's shift function.
- Record actual environment, source hashes, commands, failures and consumed data hashes.
- Preserve this reference before feature changes; add tests before changes to its semantics.
- No Docker or Actions. Do not launch remote jobs or scheduled/background services without
  actual user authorization. A tool boundary is not research success.

Deliver real logs, tested source, numerical comparisons and remaining evidence gaps.
A plan or generated directory alone does not count as execution.
