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

# 从配置文件中读取 output_path
OUTPUT_PATH=$(python -c "
import yaml
import sys
try:
    with open('$CONFIG_FILE', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    output_path = config.get('output_path', '')
    if output_path:
        print(output_path)
    else:
        print('output', file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f'错误: 无法读取配置文件: {e}', file=sys.stderr)
    sys.exit(1)
")

if [ $? -ne 0 ] || [ -z "$OUTPUT_PATH" ]; then
    echo "错误: 无法从配置文件中读取 output_path"
    exit 1
fi

# 确保输出目录存在
mkdir -p "$OUTPUT_PATH"

# 设置日志文件路径（在输出目录下）
LOG_FILE="$OUTPUT_PATH/train.log"

echo "=========================================="
echo "开始训练"
echo "=========================================="
echo "配置文件: $CONFIG_FILE"
echo "输出目录: $OUTPUT_PATH"
echo "日志文件: $LOG_FILE"
echo "=========================================="
echo "训练将在后台运行，日志将保存到: $LOG_FILE"
echo "查看日志: tail -f $LOG_FILE"
echo "=========================================="

# 使用 nohup 在后台运行训练脚本，日志保存到输出目录
nohup python -u unsloth/train_sft.py --config "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &

# 获取进程 ID
TRAIN_PID=$!

echo "训练进程已启动，PID: $TRAIN_PID"
echo "进程 ID 已保存到: $OUTPUT_PATH/train.pid"
echo "$TRAIN_PID" > "$OUTPUT_PATH/train.pid"

echo ""
echo "提示:"
echo "  - 查看实时日志: tail -f $LOG_FILE"
echo "  - 查看进程状态: ps -p $TRAIN_PID"
echo "  - 停止训练: kill $TRAIN_PID"
echo ""

