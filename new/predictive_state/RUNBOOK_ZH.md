# GPU 执行与对照：Predictive-State Drafter v0

**状态：代码已经提交；CPU 数学测试 12 项通过（本地 torch 2.10.0+cpu），真 Qwen3.5-4B + CUDA 的模型加载、训练、缓存解码与 tokens/s 均未验收。不能将下面命令解释为实验已经跑完。**

## 0. 先确认仓库与环境

~~~bash
git clone --single-branch -b research/predictive-state-opd-gdn-20261114 \
  https://github.com/Yunbo-max/branch.git branch-predictive
cd branch-predictive/new/predictive_state
python3.11 -m venv .venv-psdraft
source .venv-psdraft/bin/activate
# 依机器 CUDA Driver 选兼容的 torch 2.10.0 wheel；不要直接复用老实验环境。
python -m pip install -e '.[train,test,data]'
python -m pytest -q
~~~

Python 依赖声明见 pyproject.toml。GPU 机器要先核对驱动、PyTorch/CUDA BF16、空闲显存与路径，再进入模型加载。当前 PyTorch reference 逐 Token 循环，不能把它的速度和 FLA fused GDN 比较。

## 1. 固定教师和数据来源

沿用 AKR 原生基线独立环境中的模型与数据 manifest，模型 revision 是 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a。AKR 首先执行：

~~~bash
python -m mtp_opsd.cli download --assets-root assets/mtp_opsd
python -m mtp_opsd.cli check --assets-root assets/mtp_opsd --full-hashes
~~~

在新分支工作目录：

~~~bash
export MODEL_DIR=/absolute/path/to/AKR/assets/mtp_opsd/models/Qwen3.5-4B
python scripts/prepare_gsm8k.py \
  --train-parquet /absolute/path/to/AKR/assets/mtp_opsd/data/gsm8k/main/train-00000-of-00001.parquet \
  --out data/gsm8k_train_questions.jsonl
~~~

只导出 7,473 条 GSM8K train 题目，不包含答案。绝不使用 MATH-500 test 或 GSM8K test 当训练样本；Coding 后续引入独立确定版本数据，不能用手写 smoke prompt 当发布结果。

## 2. 先验收最小端到端闭环

~~~bash
python -m psdraft.cli train --model-dir "$MODEL_DIR" --data data/smoke_prompts.jsonl \
  --mode tf --updates 2 --horizon 4 --out runs/smoke_tf --device cuda:0
python -m psdraft.cli train --model-dir "$MODEL_DIR" --data data/smoke_prompts.jsonl \
  --mode opd --init runs/smoke_tf --updates 2 --horizon 4 \
  --out runs/smoke_opd --device cuda:0
python -m psdraft.cli verify --model-dir "$MODEL_DIR" --data data/smoke_prompts.jsonl \
  --out runs/smoke_opd --horizon 4 --device cuda:0
~~~

记录真实模型完整加载、最后一个 Prompt Token 的 hidden/logits、双输入路径的因果对齐、非零有限梯度、模型保存和重载。GPU 报错直接停，不允许偷偷回退到随机 Teacher、随机 MTP Head 或 NTP 伪装。模型 checkpoint 使用独立输出目录，每次新代码/配置使用新 run ID。

## 3. 开始两条独立 case

~~~bash
python -m psdraft.cli train --model-dir "$MODEL_DIR" \
  --data data/gsm8k_train_questions.jsonl --mode tf \
  --updates 128 --horizon 4 --seed 42 \
  --out runs/b02_tf_seed42 --device cuda:0

python -m psdraft.cli train --model-dir "$MODEL_DIR" \
  --data data/gsm8k_train_questions.jsonl --mode opd \
  --init runs/b02_tf_seed42 --updates 128 --horizon 4 --seed 42 \
  --out runs/b03_opd_seed42 --device cuda:0
~~~

**第一版只是有界开发试验，不是公平的最终比较：** B3 多了 128 次参数更新，且 OPD 额外调用 Teacher，必须额外运行 B2 增量训练的等参数更新/计算对照，并统计真实 GPU FLOPs/Teacher forward 次数。最终至少多 seed、K=2/4/8。

## 4. 首个可以计算的指标：参考接受长度

选一个不与 train 重合的 Prompt-only eval JSONL；本地运行：

~~~bash
python -m psdraft.cli evaluate --model-dir "$MODEL_DIR" \
  --data /absolute/path/to/heldout_prompts.jsonl \
  --out runs/b02_tf_seed42 --report runs/b02_acceptance --horizon 4 --num-samples 4 --seed 43

python -m psdraft.cli evaluate --model-dir "$MODEL_DIR" \
  --data /absolute/path/to/heldout_prompts.jsonl \
  --out runs/b03_opd_seed42 --report runs/b03_acceptance --horizon 4 --num-samples 4 --seed 43
~~~

输出 mean_accepted_draft_tokens、mean_emitted_tokens、fully_accepted_fraction 和每题原始行。**这不是任务成功率或真实 tokens/s**：现阶段 Teacher 全前缀重算，不能直接量化 Speculative Decoding 加速。完整 LM-Harness GSM8K 1,319 / MATH-500 500 仍在 AKR 作为目标质量基准；新 Drafter 的生产 KV/state 验证必须单独开发。

## 5. 目前不可以宣称的事情

- 未证明小型纯 GDN Draft Head 比小型 Transformer/Mamba 快；短 K 下后者可能更有优势。
- 未接 FLA Chunk/Fused recurrent kernels；纯 PyTorch 实现是可微数学原型，缺少 ShortConv 等官方 GDN 组件。
- 未实现多分支 MTP state objective、跨轮次 verifier feedback 学习、跨任务 Continual Learning。
- 未实现真实目标 KV/GDN 状态的增量提交/回滚，也没有发布真实任务准确率或 end-to-end 加速。
- 不把旧 Branch/MARCH 的预训练分数、新 AKR OPSD 尚未运行的 case 解释为本项目正结果。

完成 CPU + 真 GPU 资格验收后，再做 B4 同参数 TinyTransformer/Mamba 对照，B5 反馈状态学习独立实验。
