# Online MQAR (fresh associations per step) — decisive negative result

Removes the fixed-dataset memorization confound: every optimizer step draws
new key–value associations, so training accuracy can only rise by learning the
retrieval algorithm, not by memorising a fixed set. Held-out valid/test are
fixed. 1x H100, 2000 steps, MQAR seq 128 / 8 KV pairs / vocab 8192.

## Results (2k steps)

| arm | valid acc | test acc | valid CE | train acc (last) | wall (s) |
|---|---|---|---|---|---|
| `delta` (plain DeltaNet) | 0.0007 | 0.0010 | 8.25 | 0.0000 | 380.8 |
| `single_geo` (Full-C GeoDelta) | 0.0000 | 0.0005 | 8.32 | 0.0000 | 86.6 |
| `transformer` (attention) | **0.992** | 0.993 | 0.108 | 1.0000 | 45.5 |

## Conclusion

- With memorisation removed, **neither the plain delta rule nor the Full-C
  GeoDelta learns the task at all**: train accuracy stays 0 and valid/test are
  at chance. The full-C protection geometry adds nothing over the plain delta
  rule (marginally worse).
- Attention learns the algorithm and generalises to ~99%.
- This is stronger than the fixed-set result: the tested recurrent arms do not
  learn to retrieve unseen KV combinations under this training protocol. It
  does **not** prove that a single recurrent state is algebraically unable to
  represent eight associations: 8 independent keys are representable within
  a 64-dimensional key state. Missing trainable retention gates, poor
  optimization, and representation/position learning remain possible causes.

## Next per plan

Step 3: reduce difficulty to 2 KV pairs to distinguish capacity/interference
from a broken recurrence/training path; also try differentiating through C
(`DIFFERENTIATE_COVARIANCE=1`). Step 4 (50B pretrain / fused kernel) is not warranted until the memory mechanism shows learnable retrieval. Also see STEP3_MQAR_DIAG_2026-10-27.md: the 2-pair seq64/full-C comparison had exactly one geometry chunk, so C was inactive and that experiment cannot establish a negative effect of Query protection.

Reproduced by: `bash run_mqar_online.sh 2000` (in `new/fair_mqar_fix/`).
