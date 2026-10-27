# GeoDelta 0.79B 正式训练计划（2026-10-27）

**项目：** [Yunbo-max/branch](https://github.com/Yunbo-max/branch)  
**代码分支：** `fix/geodelta-079b-train-20261027`  
**训练模型：** 单状态 GeoDelta Full Query Geometry（参数规模约 0.79B）  
**状态：** 代码已经提交；**未启动 H200 训练，仓库完整 pytest 和 8 GPU 基准尚待验收**。

英文完整技术协议：`GEODELTA_0P79B_TRAINING_PLAN_2026-10-27.md`。

## 一、我们保留什么、不改什么？

保留真正的研究机制：

\[
C_{b+1}=\gamma^L C_b+(1-\gamma)\sum_{i=1}^{L}\gamma^{L-i}w_iq_iq_i^\top,
\quad H_b=\lambda I+\mu C_b,
\]

\[
a_t=\frac{H_b^{-1}k_t}{\tau+k_t^\top H_b^{-1}k_t},
\qquad
S_t=S_{t-1}+a_t(v_t-S_{t-1}^\top k_t)^\top.
\]

- **完整 (C)**：每个 Head 的保护矩阵大小 256×256，保留全部非对角相关项。
- **状态数 (M=1)**：不使用未验证的多状态 Fork。
- **几何块 256 tokens**：一个块中固定保护矩阵 (C_b)。
- **历史读出**：继续使用原来的块首快照，不额外加入 MARCH Anchor Bank。
- **暂不加入 Selective RoPE、低秩 (C)、新的 Gate**：先验证原方案训练与扩展能力，避免多个变量共同变化。

可选优化只有**内部递推 Tile**：在完整256-token几何块内，按照64-token片段做按序三角求解；块首 (C_b)、写入方向和历史读出始终不变。此优化理论上保持同一递推，但 GPU 速度必须实测，不能假设一定加速。

## 二、模型与训练设定

| 字段 | 设定 |
|---|---|
| 模型 | `FairLM`，21层，hidden=1536 |
| 参数量 | 约793M；以启动时实际参数统计为准 |
| Head | 6；每头 K/V=256/256 |
| 语料 | 已有16K打包数据；4K训练时在每条数据行随机裁剪4K |
| 第一个训练上下文 | **4096** |
| 后续扩展 | 16384（单独验证之后再训练） |
| GPU | 默认 **8×H200**；GPU数量可调整 |
| GPU本地 Batch | 1 |
| 4K梯度累积 | 128 |
| 16K梯度累积 | 32 |
| 全局 Tokens/优化器步 | `8×1×4096×128=4,194,304` |
| 50B Tokens 所需步数 | **11,921** |
| 优化器 | AdamW（CUDA fused 可选，默认训练脚本开启） |
| 峰值学习率 | `4e-4`，WSD，warmup 1%，末尾20%衰减 |
| 精度 | BF16模型主干，FP32完整几何、状态及求解 |
| 梯度检查点 | 开启，单几何块粒度 |
| 几何衰减 | 初始 `gamma=0.95`，后续独立消融 |
| Checkpoint | 每500步，保留最新；约10GB量级/次，需确认剩余空间 |

**4K和16K不是“把 (C) 变成4096×4096”**。(C) 的大小由 Key 维度决定；上下文长度决定模型一次处理多少 token。4K窗口训练时，应当使用随机裁剪，不能总取16K数据行的前4096 tokens。

**时间约束：** 11,921步若要求4天内完成，需要整体平均不超过约29秒/优化器步（包括验证及保存）。目前未拿到0.79B H200真实性能数据，不能保证这个时间预算能够实现。

## 三、已经修复哪些训练问题？

1. **DDP通信**：梯度累积中前127个微步不做全量梯度同步，仅最后一个微步同步；前向和反向都进入同一 `no_sync()` 作用域。
2. **CUDA隐形同步**：不再在每个微步把 Loss 转换成Python浮点数，也不逐微步用Python条件判断有限性；每个优化器步汇总后统一检查，并在DDP各 Rank 协调失败标志。
3. **混合精度**：外层启用BF16降低主干计算和激活开销；`C,H,a,S` 的求解/递推明确关闭 autocast。
4. **数据覆盖**：4K阶段训练从16K打包行随机起点裁剪；验证保持固定面板。裁剪由保存进 Checkpoint 的 Generator 控制，续训保持随机序列的可重现性。
5. **分块求解**：引入 `--delta-tile-len 64` 实验选项，同时合入之前单状态懒 Cholesky 修正。
6. **步数与token统计**：修正全局tokens/step、断点恢复后的最后步数，记录精度、GPU数及累积等。
7. **基线标签**：`delta_proxy` 是固定各向同性 Delta，**不是正式 Gated DeltaNet**；`march_proxy` 是简化分块历史读取，**不是完整原生 MARCH**。未验证的 `fork` 正式规模训练已主动禁止。

## 四、先验证，再决定是否正式投50B

### 第一步：仓库CPU正确性测试

```bash
git fetch origin
git checkout fix/geodelta-079b-train-20261027

python -m pytest -q new/fair_mqar_fix/tests/test_079b_training.py
python -m pytest -q new/fair_mqar_fix/tests/test_single_geometry_parity.py
python -m pytest -q new/fair_mqar_fix/tests
```

验收：Full与Tiled的输出、最终状态、输入和权重梯度一致；不规则切分与流式读取一致；FP32几何不被BF16意外降精；随机裁剪可复现；DDP `no_sync` 仅在最后微步关闭。

另已独立执行FP64小规模数学对照：把同一Delta递推拆成1/2/4-token小块，与整块比较的梯度差异最大约 `1.14e-13`。这**不是仓库完整回归测试通过**，也不证明GPU快。

### 第二步：不占GPU的参数核查

```bash
DRY_RUN=1 GPUS=8 SEQ_LEN=4096 bash new/fair_mqar_fix/run_0p79b.sh single_full 10
DRY_RUN=1 GPUS=8 SEQ_LEN=16384 bash new/fair_mqar_fix/run_0p79b.sh single_full 10
```

检查：4K自动得到累积128、16K得到累积32；确认 `--amp bf16`、`--random-crop`、`--delta-tile-len` 以及所有目录正确。

### 第三步：H200单卡算子分析

```bash
cd new/fair_mqar_fix
python profile_079b.py --arm single_full --seq-len 4096 --delta-tile-len 0 --warmup 2 --repeats 5 --trace-dir ../../results/traces_full
python profile_079b.py --arm single_full --seq-len 4096 --delta-tile-len 64 --warmup 2 --repeats 5 --trace-dir ../../results/traces_tiled
python profile_079b.py --arm isotropic --seq-len 4096 --warmup 2 --repeats 5
```

**重点不是先追速度数字**，而是拆开 Cholesky、Triangular Solve、(C) 更新、反向传播和优化器各自成本。若Tile版本没有加速，保持 `DELTA_TILE_LEN=0`；不能因理论复杂度更低就默认启用。

### 第四步：8卡短 Smoke（仅约16万Token）

```bash
GPUS=8 SEQ_LEN=4096 GLOBAL_TOKENS=32768 \
TIME_BUDGET_SECONDS=3600 EVAL_EVERY=3 CHECKPOINT_EVERY=0 \
RUN_TAG=smoke_4k bash new/fair_mqar_fix/run_0p79b.sh single_full 5
```

观察所有Rank启动、梯度聚合、无NaN、显存峰值、真实端到端速度。这里只验证运行，不据此评价模型质量。

### 第五步：4K收敛预实验

```bash
# 每优化器步约52万Token，100步约5243万Token
GPUS=8 SEQ_LEN=4096 GLOBAL_TOKENS=524288 \
RUN_TAG=pilot_4k EVAL_EVERY=20 CHECKPOINT_EVERY=50 \
bash new/fair_mqar_fix/run_0p79b.sh single_full 100

GPUS=8 SEQ_LEN=4096 GLOBAL_TOKENS=524288 \
RUN_TAG=pilot_4k EVAL_EVERY=20 CHECKPOINT_EVERY=50 \
bash new/fair_mqar_fix/run_0p79b.sh isotropic 100
```

记录训练Loss、独立验证NLL、梯度范数、实际每步时间、显存、吞吐和稳定性。最初的小模型10K步NLL `3.604→3.513` 是历史信号，不是0.79B预实验的预期目标。

### 第六步：通过前述门槛后正式启动50B Tokens

```bash
GPUS=8 SEQ_LEN=4096 GLOBAL_TOKENS=4194304 \
RUN_TAG=full_4k DELTA_TILE_LEN=0 \
bash new/fair_mqar_fix/run_0p79b.sh single_full 11921

# 同样的种子、数据、训练协议和预算，对照Isotropic：
GPUS=8 SEQ_LEN=4096 GLOBAL_TOKENS=4194304 \
RUN_TAG=full_4k DELTA_TILE_LEN=0 \
bash new/fair_mqar_fix/run_0p79b.sh isotropic 11921
```

只有预实验实际验证Tile64吞吐更好且一致性测试全部通过，才把两臂的 `DELTA_TILE_LEN` 同时改成64；不能在两个对照中任意改变其一。真实全局Batch目标约419万tokens/step，50B处理量不等于50B不重复的新数据。

## 五、科学实验和后续16K阶段

正式GeoDelta Full与Isotropic对照之后，才依次做 `gamma=0.95/0.99/0.999323`、`differentiate_covariance=True/False` 和 `read_mode=chunk_start/current_only` 消融。切勿同时改多个变量。除了LM NLL，还应测试历史读出扰动和新Query检索，以及4K MQAR。

单独的官方GDN/MARCH对照需要基于 `new/MARCH/` 下的FLA/FLAME路径，并明确匹配骨干、FFN、头维度和参数量。目前仓库的 `delta_proxy` 与 `march_proxy` 不能当作官方性能。新增MARCH Python逐Token参考实现也没有正式融合训练核，不适合直接宣称0.79B高效。

16K后续继续使用相同完整 (C)，设置 `SEQ_LEN=16384`，全局tokens不变时8卡累积改为32。此阶段应标记为 **4K→16K上下文扩展**，除非从相同随机初始化、相同token预算独立训练16K公平基线。

## 六、停止和验收条件

出现非有限Loss/Gradient、长时间验证恶化、全序列/逐Token解码不一致、反复OOM、磁盘写Checkpoint失败，或H200实测吞吐不足以满足预算时，先停止放大训练并保留最近Checkpoint。断点续训要求相同分支、相同配置、相同训练行随机源、相同学习率总步数，明确报告恢复点；不要把磁盘问题当作模型收敛问题。

**最终目标**：先证明0.79B Full (C) 在受控比较下带来可重复的质量收益，再优化底层独立 `k_t,a_t` 的Delta Kernel，最后才讨论27B/100B扩展。Selective RoPE暂时不纳入第一轮，避免掩盖完整Query保护机制的效果。
