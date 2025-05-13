# Qwen3微调实战：医疗R1推理风格聊天 

[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](https://swanlab.cn/@ZeyiLin/qwen3-sft-medical/overview)

算力要求：
- 全参数微调：32GB显存
- LoRA微调：28GB显存

> 如果需要进一步降低显存需求，可以使用Qwen3-0.6B模型，或调低`MAX_LENGTH`。

## 安装环境

```bash
pip install -r requirements.txt
```

## 数据准备

```bash
python data.py
```

## 训练

全参数微调
```bash
python train.py
```

LoRA微调
```bash
python train_lora.py
```

SwanLab训练日志：[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](https://swanlab.cn/@ZeyiLin/qwen3-sft-medical/overview)
