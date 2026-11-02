# 原生 GDN 读写诊断：防位置与排除法的配对面板 v2

这是测试协议修复，不是新的 GDN 成绩。`run_rwdiag.sh` 默认改跑 v2；旧面板和旧结果保留，旧读出器不得混用。项目原始读写干预、输出层和主干权重不改。只在真实原始 checkpoint 上解释 `rwdiag` 的 CUDA 输出。

## 从仓库根目录启动

```bash
# 在已有 CUDA/FLA 0.5.0 环境；默认权重是本仓库的 models/puigde_360M
bash new/gdn_update_training/run_rwdiag.sh pilot
# 可选：只审核出题，不加载权重，也不要求 GPU
bash new/gdn_update_training/run_rwdiag.sh audit
```

`MODEL` 可覆盖为原始本地权重的绝对路径，或固定的 puigde Hub ID。`OUT_ROOT` 可改，默认 `new/gdn_update_training/runs/binding_v2/`。每次自动创建独立目录并打印 `RUN_DIR=...`；不覆盖、不删除、不把旧结果算作这次结果。例：`MODEL=/absolute/puigde_360M OUT_ROOT=/scratch/gdn-b2 bash new/gdn_update_training/run_rwdiag.sh pilot`。可用 `TRAIN_SCENES` / `DEV_SCENES`（正的 24 倍数）、`FIT_STEPS`、`LAYER`、`HEAD` 控制。pilot 默认 24 训练场景 / 192 个诊断读出器训练步 / 24 个独立 dev 场景，主干冻结。无作业提交和 SFT。仅该层/头的局部解释，不凭零收益判定整网没有信息。

保留测试可复用上述读出器：

```bash
READER_DIR=/absolute/path/to/pilot_XXXXXXXX/fit \
  DISTANCE=128 bash new/gdn_update_training/run_rwdiag.sh retention
```

`READER_DIR` 省略时重新拟合；已有读出器需匹配模型、代码、层头、维度、数据种子、六个 v2 面板。先 pilot 后决定是否跑大网格，512/1024 距离应单独比较（距离是填充 token 数，实际距离记在原始行）。所有报告都是 dev 筛查，不是独立 test 确认。

## 生成规则与主指标

每个场景有六个面板：`bind_x_xy/yx`（只改 X），`bind_y_xy/yx`（只改 Y），`bind_swap_xy/yx`（交换 X/Y 的终值）。后缀为两种实际更新顺序。每个场景都取两种顺序、同一组实体、相同底值、同一 Z 尾记录；每份历史既问 X 当前值，也问 Y 当前值。单实体变化的两世界在完整更新后只有被修改记录的值不同；初值与另一实体固定值用独立的均衡列生成，不再用其余两个字母做补集，允许实体同值及重复原值。交换组保留两个不同值的相同多重集，但不能只用交换组证明记忆里存在 X。变化组与不变组分开解释。

`audit` 对实际渲染的字符级文本逐组检查：遮住被改变的新值后两份历史相同、问题相同，变化实体的正确答案却不同。因此任何看不到该更新值的确定性读者不能把该对的变化题都答对。交换组检查只抄第一/最后更新值的规则是否能通过联合绑定。**联合 0 不是逐题准确率 0；这不证明没有一切潜在捷径。**

pilot 只在 `other` 阶段（X、Y 均已更新，且有 Z）比较两个当前值问题。24 场景 × 六面板 × 两世界 × 两问题 = 576 个基础问题行，每行有多次实际前向。`retention` 加入 `write/other/neutral/records` 及 X 最早值、Z 当前值；`write` 是紧跟 X 更新后，在 XY 顺序下 Y 可能尚未被更新，不能放入跨顺序终值的主指标。模型干预仍在 X 更新位置：Y-only 变化组是串扰/保护测试，不是对 Y 写入的独立干预。

## 读结果

`dev/decision.json -> crossed_binding` 是 v2 主表：按场景聚合，`role_all4` 是一个实体在两顺序 × 两世界均正确，`all8` 是 X、Y 都正确。先比较 `native` vs `q` 的 `all8_delta`、`net_row_delta`、`rescued_rows/broken_rows`，再比较 `k/beta/v`、联合干预以及 `q_sham`、`question_only`。`probe_state_swap.donor_follow_all_changed` 测的是答案是否跟随供体（只计标签确实变化的问题）；`bind_y.xy_crosstalk_all4` 只在两个世界的 Y 值均不同于固定 X 值时计数，分母另报；是答案跟随模式，不自动证明唯一因果路径。`decision.json.records_minus_neutral` 只在有两种等长延迟时可解释。`dev/summary.json` 保留分顺序/阶段的读写四格与相互作用；其中历史名 `cross_follow_both` 仅适用旧 `cross` 面板，v2 请看 `decision.json`。历史保护题仍在保留测试的逐项结果中分开检查。`base_report.json` 是完整性校验用的中间文件。V 供体是诊断上界的一种干预，不是部署方法。
