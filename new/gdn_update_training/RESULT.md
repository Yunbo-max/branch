# 读写诊断结果：Q 侧寻址不成立，转入行为级更新原型

主候选 `q` 的独立测试（48 场景，`records`@1024，X 当前值）未通过固定判据，路由为 `behavioral_prototype_only`。

## 主结论

- 主指标（Q 相对 native 收益）：**0.0**（95% CI [−0.026, +0.021]），低于 +5pp 且下界未 >0 → **不通过**。
- 相对 `q_sham`（等幅随机方向）：**0.0**（[−0.026, +0.021]）→ 不通过。
- 保护题（下界须 > −5pp）：`y_now` 0.0、`x_first` +0.026、`z_now` 0.0 → 全部通过。

结论：不改 Q/查询寻址；按预注册回退，进入同数据、同可训练参数 SFT 对照下的**行为级反事实更新训练**。

## 探索线索（非二次确认）

- Y 跟随对齐实体：**+0.1125**（[0.025, 0.20]）。
- 长距离中性保留（neutral 1024 − 128）：**+0.010**（[0.0, 0.026]），原生模型基本不随填充长度退化。
- 记录干扰（records 1024 − neutral 1024）：**−0.078**（[−0.146, −0.010]），其他记录造成显著下降。
- `exploratory_design_lead`：**fixed_budget_write_isolation**（等总字节写入隔离 / 固定槽）。

## 工件

- `route.json`：本目录下的最终决策（原位于该 pilot 的 `design_confirmation/route.json`）。
- `reader_sha256`：`e2f21cc315fa48e723148b467a01f0cb68d657d26b35351b633281bc6c252a8e`。
- 完整 55,296 行 test 原始记录与 status 在 `runs/binding_v2/pilot_WWS2kLuP/design_confirmation/test/`（未提交，体积较大）。
