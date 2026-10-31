# GDN update learning v1

User-approved scope: implement separate ordinary SFT, counterfactual consistency (A), and native-write distillation (B) experiments on top of the completed GDN census. No growing history cache, new inference modules, kernel replacement, paid job launch, or overwrite of prior results.

Base: Yunbo-max/branch@6bbba481a56b254df965238f91e427fc9fce030e. New code is isolated under new/gdn_update_training; old census/results remain immutable.

Data: exact tokenizer lengths; twelve ordered old/new label transitions per block; split-specific entity IDs and held-out templates; current-value counterfactual pairs; latest-only and repeated-new controls; unchanged-entity and earliest-value queries. Only current-value pairs share JS targets. No target labels or teacher decisions are causal model inputs.

A: common full-vocabulary answer-token CE plus JS of the same four restricted answer candidates for current-value pairs. CE-only sees the same groups and trainable parameters.

B: a bounded training-only native-kernel-boundary K/beta intervention at one known update-value token and one layer/head. Search uses actual downstream gradient and reruns the candidate, accepting only improvement with protected/earliest-query safeguards. Distil accepted normalized keys and beta into existing parameter paths. It is a practical direct-gradient teacher, NOT the unimplemented state-gradient closed-form oracle. All model flags and global functions restored on exception. Unsupported FLA layouts fail closed. No GPU run is claimed from CPU tests.

Train/eval/save: reproducible manifest and configs; no overwrite; native HF export; immutable input hashes; JSONL train/eval logs with teacher cost; evaluation without intervention; repeated seeds reported as data seeds, not training replicas. v1 has no optimizer resume, packed batches, GVA teacher, distributed execution or checkpoint recomputation during taps. It does not mix general LM data automatically; retention is task-level, not a general-capability guarantee.
