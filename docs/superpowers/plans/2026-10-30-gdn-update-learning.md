# GDN update learning Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans; execute each test cycle before claiming completion.

**Goal:** Three trainable, separately selectable baselines with a native-export path.
**Architecture:** Self-contained package; existing census loader reused at GPU entry; balanced grouped data; shared runner and losses; removable native operator tap for B.
**Tech Stack:** Python 3.11+, PyTorch, FLA 0.5.0, Transformers; CPU pytest; no model download in CPU tests.
**Spec:** docs/superpowers/specs/2026-10-30-gdn-update-learning.md

## Global Constraints
Fixed recurrent state; unchanged exported native inference; no existing data/log edits; no paid GPU jobs; fail closed on unsupported layouts; disclose missing native GPU validation.

## Review Focus
- Label imbalance and train/test entity leakage: enumerate all twelve transitions.
- Wrong historical JS target: only paired current queries enter consistency loss.
- Native tap remains installed on exception or candidate failure: restoration tests.
- Teacher no-op/wrong-direction accepted: candidate replay and safeguard tests.
- Half-written/overwritten outputs and mock results mislabelled: refusal and provenance tests.

### Task 1: data and objectives
Files: data.py, objectives.py, tests/test_learning.py.
- [ ] Write/run failing tests for balance, deterministic splits, exact lengths, choice suffix, shared update positions, and CE/JS gradients.
- [ ] Implement grouped rendering and losses; verify tests.

### Task 2: training-only teacher and kernel tap
Files: teacher.py, native.py, tests/test_native_teacher.py.
- [ ] Write/run failing tests for local update math, teacher improvement/protection, tap gradient and restoration.
- [ ] Implement bounded direct-gradient teacher and FLA boundary adapter; verify CPU reference tests.

### Task 3: native train/eval CLI and documentation
Files: train.py, run.sh, README.md, tests/test_runner.py; independent workflow.
- [ ] Write/run failing tests for mode switching, step updates, export identity and unsafe overwrite.
- [ ] Implement all three arms with matched data/scope, dry-run, manifests, evaluation and export.
- [ ] Run full package suite, shell syntax, compile and workflow checks; obtain remote CI where possible.
- [ ] Publish new branch, preserve source head, read back files; report GPU validation separately.
