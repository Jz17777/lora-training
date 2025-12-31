#!/bin/bash

# Unsloth LoRA 训练脚本
# 使用方法: bash scripts/run_unsloth.sh [配置文件路径]

# 设置默认配置文件路径
CONFIG_FILE="${1:-configs/unsloth/qwen3_8b_lora.yaml}"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 运行训练脚本
python unsloth/train_sft.py --config "$CONFIG_FILE"

