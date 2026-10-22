---
library_name: transformers
pipeline_tag: text-generation
tags:
- resi
- safety
- merged-model
---

# ReSI-Safin-1-35B-A3B

This repository contains the **Safin-1 (35B-A3B) model after ReSI alignment**, together with its inference code and tokenizer.

## Quick start

Use Python 3.12 with a CUDA-enabled PyTorch installation. After downloading the repository, install the inference dependencies from [requirements.txt](requirements.txt):

```bash
pip install -r requirements.txt
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "AI45Research/ReSI-Safin-1-35B-A3B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
).eval()

messages = [{"role": "user", "content": "Explain why the sky is blue in three sentences."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(
    model.get_input_embeddings().weight.device
)
with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=8192)
print(tokenizer.decode(outputs[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True))
```

Alternatively, run the included script for streaming output:

```bash
python inference.py --model . --prompt "Explain why the sky is blue in three sentences."
```

`--model` accepts a local model directory or the Hugging Face repository ID. Generation follows the saved defaults: temperature **1.0**, top-p **0.95**, and top-k **20**. The script supports overrides and `--greedy`; `--max-new-tokens` defaults to **8,192** and can be increased for longer answers. The model may generate reasoning before its final answer.

## Runtime

The verified loading path uses **Transformers 5.5.4**, BF16 weights, and eager attention. The weights occupy approximately **66.4 GiB**; allow additional memory for inference. Keep the bundled Python files alongside the weights and enable `trust_remote_code=True`. Optional Flash Linear Attention and causal-conv1d packages are not required for this path.

## Code and paper

- **Code:** [AI45Lab/ReSI](https://github.com/AI45Lab/ReSI)
- **Paper:** [ReSI: Recursive Safety Improvement toward Resistant and Resilient AI](https://arxiv.org/abs/2610.12233)

## Citation

If you use ReSI in your research, please cite:

```bibtex
@misc{zheng2026resirecursivesafetyimprovement,
  title = {ReSI: Recursive Safety Improvement toward
           Resistant and Resilient AI},
  author = {Jingnan Zheng and Dongcheng Zhang and Yi Zhang and Ming Zhang
            and Qiaosheng Zhang and Youbang Sun and An Zhang and Xiangnan He
            and Tat-Seng Chua and Xia Hu and Bowen Zhou
            and Chaochao Lu and Xiang Wang},
  year = {2026},
  eprint = {2610.12233},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR},
  url = {https://arxiv.org/abs/2610.12233}
}
```
