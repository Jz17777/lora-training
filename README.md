# LoRA Training

This repository contains **reproducible LoRA fine-tuning templates** for large language models,
using both **Unsloth** and **HuggingFace Transformers**.

The goal is to provide:
- Clean, runnable training scripts
- Clear separation between frameworks
- Practical configurations for single-GPU LoRA fine-tuning

---

## Supported Frameworks

| Framework | Description | Pros |
|-----------|-------------|------|
| **Unsloth** | Fast LoRA fine-tuning | 2x faster, memory-efficient (4-bit + checkpointing) |
| **Transformers + PEFT** | Standard HuggingFace training pipeline | Maximum compatibility, widely supported |

---

## Repository Structure

```text
├── configs/
│   ├── transformers/           # Transformers 配置文件
│   │   ├── qwen3_8b_lora.yaml  # Qwen3-8B 训练配置
│   │   └── template.yaml       # 配置模板（带详细注释）
│   └── unsloth/                # Unsloth 配置文件
│       ├── qwen3_8b_lora.yaml  # Qwen3-8B 训练配置
│       └── template.yaml       # 配置模板（带详细注释）
├── data/
│   ├── format_chatml.py        # 数据格式化脚本
│   ├── processed/              # 处理后的数据集
│   └── raw/                    # 原始数据集
├── scripts/
│   ├── preprocess_dataset.sh   # 数据预处理脚本
│   ├── run_transformers.sh     # Transformers 训练启动脚本
│   └── run_unsloth.sh          # Unsloth 训练启动脚本
├── transformers/
│   └── train_sft.py            # Transformers + PEFT 训练脚本
├── unsloth/
│   └── train_sft.py            # Unsloth 训练脚本
├── requirements.txt            # Python 依赖
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
# 基础依赖
pip install -r requirements.txt

# Unsloth (可选，如果使用 Unsloth)
pip install unsloth

# Transformers + PEFT (如果使用 Transformers)
pip install transformers peft trl datasets bitsandbytes accelerate
```

### 2. Prepare Dataset

数据集格式为 JSONL，每行一个 JSON 对象：

**方式一：直接使用 text 字段**
```json
{"text": "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n你好！有什么可以帮助你的吗？<|im_end|>"}
```

**方式二：使用 messages 字段（需要启用 chat_template）**
```json
{"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"}]}
```

### 3. Configure Training

复制配置模板并修改：

```bash
# Transformers
cp configs/transformers/template.yaml configs/transformers/my_config.yaml

# Unsloth
cp configs/unsloth/template.yaml configs/unsloth/my_config.yaml
```

主要配置项：
- `model_path`: 基础模型路径
- `output_path`: 输出目录
- `training.dataset_path`: 训练数据路径
- `lora.r`: LoRA rank
- `training.learning_rate`: 学习率

### 4. Start Training

**使用 Transformers + PEFT:**
```bash
bash scripts/run_transformers.sh configs/transformers/qwen3_8b_lora.yaml
```

**使用 Unsloth:**
```bash
bash scripts/run_unsloth.sh configs/unsloth/qwen3_8b_lora.yaml
```

### 5. Monitor Training

```bash
# 查看实时日志
tail -f /path/to/output/train.log

# 查看进程状态
ps -p $(cat /path/to/output/train.pid)

# 停止训练
kill $(cat /path/to/output/train.pid)
```

---

## Configuration Reference

### Base Configuration

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_path` | string | ✅ | 基础模型路径（本地或 HuggingFace） |
| `output_path` | string | ✅ | 模型保存路径 |
| `max_seq_length` | int | ✅ | 最大序列长度 |
| `load_in_4bit` | bool | ✅ | 是否使用 4bit 量化 |
| `load_in_8bit` | bool | ✅ | 是否使用 8bit 量化 |
| `device_map` | string | ✅ | 设备映射方式 |
| `dtype` | string | ❌ | 数据类型（null 自动选择） |
| `attn_implementation` | string | ❌ | 注意力实现方式（仅 Transformers） |

### LoRA Configuration

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `r` | int | ✅ | - | LoRA rank |
| `lora_alpha` | int | ✅ | - | LoRA alpha |
| `lora_dropout` | float | ✅ | - | LoRA dropout |
| `target_modules` | list | ❌ | 见下方 | 目标模块列表 |
| `bias` | string | ❌ | "none" | 偏置参数处理 |
| `use_gradient_checkpointing` | bool/string | ❌ | true/"unsloth" | 梯度检查点 |
| `use_rslora` | bool | ❌ | false | 是否使用 RSLoRA |

默认 `target_modules`:
```yaml
- "q_proj"
- "k_proj"
- "v_proj"
- "o_proj"
- "gate_proj"
- "up_proj"
- "down_proj"
```

### Training Configuration

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `dataset_path` | string | ✅ | - | 训练集路径 |
| `per_device_train_batch_size` | int | ✅ | - | 批次大小 |
| `gradient_accumulation_steps` | int | ✅ | - | 梯度累积步数 |
| `num_train_epochs` | int | ✅ | - | 训练轮数 |
| `learning_rate` | float | ✅ | - | 学习率 |
| `warmup_steps` | int | ✅ | - | 预热步数 |
| `weight_decay` | float | ✅ | - | 权重衰减 |
| `lr_scheduler_type` | string | ✅ | - | 学习率调度器 |
| `optim` | string | ✅ | - | 优化器 |
| `logging_steps` | int | ✅ | - | 日志步数 |
| `logging_strategy` | string | ✅ | - | 日志策略 |
| `report_to` | string | ✅ | - | 日志报告工具 |
| `seed` | int | ✅ | - | 随机种子 |
| `eval_dataset_path` | string | ❌ | null | 验证集路径 |
| `use_chat_template` | bool | ❌ | false | 使用 chat template |
| `packing` | bool | ❌ | false | 启用 packing |
| `bf16` | bool | ❌ | true | 使用 bf16（仅 Transformers） |
| `fp16` | bool | ❌ | false | 使用 fp16（仅 Transformers） |

---

## Framework Comparison

| Feature | Unsloth | Transformers + PEFT |
|---------|---------|---------------------|
| 训练速度 | 🚀 2x faster | 标准速度 |
| 显存占用 | 更低 | 标准 |
| 兼容性 | 部分模型 | 所有模型 |
| Flash Attention | 内置 | 需要配置 |
| 社区支持 | 较新 | 成熟 |

---

## Tips

1. **显存不足？**
   - 减小 `per_device_train_batch_size`
   - 增加 `gradient_accumulation_steps`
   - 启用 `load_in_4bit`
   - 减小 `max_seq_length`

2. **训练不稳定？**
   - 降低 `learning_rate`
   - 增加 `warmup_steps`
   - 设置 `max_grad_norm: 1.0`

3. **快速测试？**
   - 设置 `max_samples: 100`

---

## License

MIT License
