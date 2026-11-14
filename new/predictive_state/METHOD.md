# 数学与实验协议（2026-11-14）

## 1. 新问题而非旧检索问题

原问题在尽量固定记忆下精确恢复历史 Key–Value。新问题是：给定已接受前缀 `c=x_{<=t}`，令冻结 Target 分布为 `p_T`，找到固定大小的学生递归状态，使 **学生自己提出的轨迹上** 的多步条件分布接近 `p_T`，并且推理成本低于直接调用 Target K 次。

起始：`S_{0}=Init_phi(h_T(c))`。其中 `h_T(c)` **只在验证边界可用**。学生提出 `y_j~q_theta(.|S_{j-1})`，随后 `S_j=F_theta(S_{j-1},y_j)`。`F` 是各层的 GDN 更新：

`a=alpha(x), b=beta(x), e=v-a S k, S'=a S+b e k^T, read=S'q`。

状态为 `L x H x d_v x d_k`，固定大小。v0 的 `d_v=d_k`，不包含官方 FLA 的 causal short convolution、fused kernels 或训练 chunk 并行。因此它是 GDN 数学参考与学生功能模型，而不是官方高效层的等价性能实现。

## 2. 两个基础目标严格分开

**B2 / Teacher Forcing**：由冻结 Target 在题目上产生参考续写 `y^T`，用 `CE(q_theta(.|S_j), y^T_{j+1})` 训练学生。此处不选择正确答案组、也不用数据集 gold；只是训练模仿教师的草稿能力。

**B3 / OPD**：从当前学生 `q_theta` 独立采样 K 个草稿 Token `y^S`。冻结教师沿 **这些相同 Token IDs** 评估 `p_T(.|c,y^S_{<j})`，学生在它自己产生的状态 `S_{j-1}` 上最小化

`L_OPD = E_{y^S~q_theta} 1/K sum_{j=1}^K KL(p_T(.|c,y^S_{<j}) || q_theta(.|S_{j-1}))`。

采样动作 stop-gradient；KL 的所有合法位置有梯度，教师参数冻结。**这是普通 OPD 基线，不是新算法。**

与 AKR 的 same-model OPSD 不同：Target 和学生是不同参数/不同架构，没有多数答案参考，也没有在教师 Prompt 内增加学生自采的正确参考。

## 3. 为什么用固定状态 / 为什么这不自动构成创新

固定状态单序列只需 `O(L H d_k d_v)` 个矩阵元素，不随 Prompt 长度增长。**但** Target 自身的 attention KV、学生大词表输出投影、Teacher boundary 提取和验证成本仍然存在。短 K 下小型 Transformer Draft Head 可能更快。因此要在相同参数、训练预算、Target 接口下比较 GDN、Transformer、Mamba Draft Head，而不预设 GDN 更优。

在固定状态容量下并不保证 `p_T(x_{future}|past)` 可以完整由 `S` 表示。只能优化在真实任务分布和有限 K 上的近似预测充分性。未来分布之间的依赖通过 **递归采样与状态转移** 而不是独立位置头来表示。

## 4. 严格解码约束

对教师概率 `p_j` 和学生 *实际提出样本的* 分布 `q_j`，按照 `a_j=min(1,p_j(y_j)/q_j(y_j))` 接受；首个拒绝处从 `normalize(max(p_j-q_j,0))` 采样修正 Token；若全部接受，教师再采样 bonus Token。Teacher 一次 full-prefix forward 得到严格对齐的 `p_1...p_{K+1}`（logits positions `P-1:P+K`）。该算法的输出分布等于 Target 的自回归分布，只要每一个 `p_j,q_j` 使用对应的相同前缀、实际概率和正确状态管理。

当前 Torch verifier **仅检验概率代数，不是可部署的高效 Target cache adapter**。生产版必须实现对已接受 Token 的目标缓存提交、拒绝 Token 的目标状态回滚及严格采样统计检验。现有 SpecLA、SpecOPD 等相关工作覆盖了这些工程挑战，不应重复声称新颖。

## 5. 未来状态学习（尚未实现，避免和 B3 混淆）

**B5 / Verifier-conditioned state**：教师每轮验证后产生 `f_r=Compress(accepted_length, p_r,q_r, rejection)`；学习 `M_{r+1}=U_phi(M_r,f_r)`，并在下一已接受上下文的预测中条件化 `q(.|S,M_r)`。推理时冻结 `theta,phi`，只更新固定状态 `M`。离线训练时必须至少跨两轮 `Draft→Verify` 反向传播，以奖励下一轮匹配提升；仅拟合当前反馈不够。对照：reset M、真实 M、乱序 f、相同参数预算的非反馈状态、在线梯度强基线。拒绝的提议不能写入目标 state，也不能污染学生的下一轮已接受前缀。

**B6 / Multi-horizon state objective**：研究在相同状态容量下，提前预测更远的条件分布是否能提升 *第一步* 决策和平均接受长度。未来 Token 必须来自已声明的 on-policy 分支/目标条件，不能把真实未来偷偷输入学生。普通 `all-prefix KL` **不等于** 新的多步状态目标；在定义数学上不同的监督之前不要实现或声称 B6。

## 6. 关卡与指标

1. CPU 数学：一 token 更新与显式公式相符、学生 sample/replay 概率一致、Teacher stop-gradient、残差重采样的输出分布匹配 Target。
2. 真 GPU：固定 revision + tokenizer 确认、HF `output_hidden_states` 时间轴对齐、K=4 的 forward/backward 非零且有限、checkpoint reload 与 logits 一致。
3. B2→B3：相同 Prompt、Token、Teacher calls、训练 FLOPs，比较 forward KL、target-conditional acceptance、memory、drafter 推理成本。不能只看训练 loss。
4. 最终系统：实现并资格验收缓存增量验证后，报告严格 lossless tokens/s、prefill 与 decode 单独耗时、峰值显存；测短 K=2/4/8 和不同 Prompt 长度。
5. 仅在上述成立后进行反馈状态 B5 或多未来 B6；如果 GDN 没有净收益，保留学习机制可换成其他高效递归架构。
