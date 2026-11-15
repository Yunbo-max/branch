# Qwen3.5-4B × GDN/GQA × 原生 MTP — 运行手册（2026-11-15）

**研究主线**：预训练 4B 全冻结；C00 复用官方 MTP；C01/C02/C03 小型只读适配器；C04 在 C03 基础上增加 *真正的未来 GDN 状态矩阵* 的压缩监督。前一版随机初始化的 `psdraft` 不再是主实验。

## 一、两份独立仓库

在同一个目录下：

~~~bash
git clone --single-branch --branch research/predictive-state-opd-gdn-20261114 \
  https://github.com/Yunbo-max/branch.git branch-predictive
git clone --single-branch --branch research/mtp-opsd-baseline \
  https://github.com/Yunbo-max/AKR.git AKR-native-MTP
~~~

AKR 提供经过代码检查的 Qwen3.5 纯文本 Backbone、15 个原生 MTP 权重加载、官方 Tokenizer、固定 revision/manifest。保留旧 AKR 代码，不要把其同模型 OPSD 与本实验的不同模型 OPD 混为一谈。

在 AKR checkout 中按 [AKR Local Agent Runbook](https://github.com/Yunbo-max/AKR/blob/research/mtp-opsd-baseline/LOCAL_AGENT_RUNBOOK.md) 获取官方模型和 GSM8K train 数据、完整 hashes。必须保存 `MODEL_DIR/model_manifest.json`。

## 二、新研究环境

~~~bash
cd branch-predictive/new/predictive_state
python3.11 -m venv .venv-qwen35
source .venv-qwen35/bin/activate

# 在本机先安装匹配 driver 的 CUDA PyTorch 2.10.0，例如 NVIDIA cu128 兼容机
# python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
export AKR_ROOT="$(realpath ../../../AKR-native-MTP)"
export MODEL_DIR="$AKR_ROOT/assets/mtp_opsd/models/Qwen3.5-4B"
python -m pip install -r "$AKR_ROOT/requirements-mtp-opsd.txt"
python -m pip install -e '.[test,data]'
python -m pip check
python -m pytest -q tests/test_qwen35_predictive.py
python -m qwen35_predictive.cli plan
~~~

AKR 的依赖在 `requirements-mtp-opsd.txt` 中固定了 Transformers 5.16.1 / 其他环境版本；不要复用旧 Branch 的 FLA 0.5.0 环境。不要在 CPU-only Torch 或 GPU 显存不足时假装已运行 4B。

## 三、GPU Qualification：必须先通过

~~~bash
python -m qwen35_predictive.cli qualify \
 --akr-root "$AKR_ROOT" --model-dir "$MODEL_DIR" \
 --prompts data/smoke_prompts.jsonl \
 --out runs/qwen35_qualify_v1.json \
 --horizon 4 --seed 42 --device cuda:0
~~~

检查：官方 24 GDN/8 GQA、固定 2560 Hidden、真实 MTP 权重、26/27 层 Mixer Hook、真实 GDN cache `[1,32,128,128]`、相同 Token IDs 的 Proposal/Replay、Target 第一 token 数值对齐、有限分布与状态差异。

不能缺权重后降级为随机头、MTP 头外接新层或者伪装为 NTP。资格验收失败应保存错误，先修复兼容性，不开始训练。

## 四、先走通 C00–C04 的独立开发试验

~~~bash
# 导出 PINNED GSM8K 7,473 条 train questions ONLY；不读取 train 答案
python scripts/prepare_gsm8k.py \
 --train-parquet "$AKR_ROOT/assets/mtp_opsd/data/gsm8k/main/train-00000-of-00001.parquet" \
 --out data/gsm8k_train_questions.jsonl
export TRAIN=data/gsm8k_train_questions.jsonl

# C00: 无训练，对照原生 MTP 的一块候选
python -m qwen35_predictive.cli evaluate --case c00_native \
 --prompts data/smoke_prompts.jsonl --out runs/c00_smoke \
 --horizon 4 --num-samples 2 --max-prompts 2 --seed 43

# C01-C04：每组独立训练，32 updates 是 bounded pilot，不代表收敛
python -m qwen35_predictive.cli train --case c01_gdn \
 --prompts "$TRAIN" --out runs/c01_seed42 --horizon 4 --updates 32 --max-prompts 64 --seed 42
python -m qwen35_predictive.cli train --case c02_gqa \
 --prompts "$TRAIN" --out runs/c02_seed42 --horizon 4 --updates 32 --max-prompts 64 --seed 42
python -m qwen35_predictive.cli train --case c03_fusion \
 --prompts "$TRAIN" --out runs/c03_seed42 --horizon 4 --updates 32 --max-prompts 64 --seed 42
python -m qwen35_predictive.cli train --case c04_state \
 --prompts "$TRAIN" --out runs/c04_seed42 --horizon 4 --updates 32 --max-prompts 64 --seed 42
~~~

每组输出独立 `run.json`、`events.jsonl`、`adapter.pt`、`summary.json`；禁止覆盖。使用相同 seed/dataset/config 对照。C04 需要对真实 GDN 状态求两个时刻的缓存标签，资源高于 C03，所有 GPU 耗时都要报告。

## 五、独立 HELD-OUT 样本上的 Reference Evaluation

准备一份只含 `{"id":"...","prompt":"..."}` 的 held-out questions JSONL（无训练重复、不得用 test gold 训练）。所有方法相同 seed、K、抽样数：

~~~bash
export EVAL=/absolute/path/to/heldout_prompts.jsonl
python -m qwen35_predictive.cli evaluate --case c00_native --prompts "$EVAL" --out runs/e00 --horizon 4 --max-prompts 64 --num-samples 4 --seed 43
python -m qwen35_predictive.cli evaluate --case c01_gdn --checkpoint runs/c01_seed42/adapter.pt --prompts "$EVAL" --out runs/e01 --horizon 4 --max-prompts 64 --num-samples 4 --seed 43
python -m qwen35_predictive.cli evaluate --case c02_gqa --checkpoint runs/c02_seed42/adapter.pt --prompts "$EVAL" --out runs/e02 --horizon 4 --max-prompts 64 --num-samples 4 --seed 43
python -m qwen35_predictive.cli evaluate --case c03_fusion --checkpoint runs/c03_seed42/adapter.pt --prompts "$EVAL" --out runs/e03 --horizon 4 --max-prompts 64 --num-samples 4 --seed 43
python -m qwen35_predictive.cli evaluate --case c04_state --checkpoint runs/c04_seed42/adapter.pt --prompts "$EVAL" --out runs/e04 --horizon 4 --max-prompts 64 --num-samples 4 --seed 43
~~~

分别 `--horizon 2`、`4`、`8` 评测不同串行深度；每次用全新目录。真正的 target speedup 需要后续独立实现增量 KV/GDN-cache verification + commit/rollback，并与官方 native MTP 解码速度基线实测。上述评测仅提供准确概率下的 Reference Acceptance，不给真实 end-to-end tokens/s。

## 六、代码复用边界和下一步

- Qwen 主干、Tokenizer 和真实 MTP：从 AKR 的严格 pinned backend 读取，不重训预训练权重；
- GDN/GQA Readout：从官方层 `linear_attn` / `self_attn` 的最后 Prompt Token Hook 获取，不把两种功能先验写死；
- 真实 GDN state：从官方 cache 第 26 层 `recurrent_states[0]` 获取，不用虚构的 hidden state；
- OPD：学生先自采 K tokens，冻结 Backbone 再计算相同前缀 KL；训练只作用于少量适配参数；
- C04：用预测出来的未来状态 sketch 作为 auxiliary loss，**并未**实现直接使用预测状态跳过 GDN 更新的架构。

必须首先证明 C03>C00、C04>C03 的真实 held-out 结果，进一步和等额参数/相同 Teacher FLOPs 的 C03 + 非状态辅助预测器比较，才谈新的 Block-Predictive State Jump。

**状态：真实 GPU 尚未执行；CI CPU mock tests 可用于代码契约，不等于模型验收。**
