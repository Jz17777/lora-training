#!/bin/bash

# 数据集预处理脚本
# 使用方法: bash scripts/preprocess_dataset.sh [输入文件] [输出文件] [输入字段名] [输出字段名]
#
# 说明：
# 此脚本调用 data/format_chatml.py 将原始数据集转换为 ChatML 格式
# 具体的数据处理逻辑在 format_chatml.py 中实现

# 设置默认路径（可根据实际情况修改）
INPUT_FILE="${1:-data/raw/train.jsonl}"
OUTPUT_FILE="${2:-data/processed/train.jsonl}"
INPUT_FIELD="${3:-conversations}"
OUTPUT_FIELD="${4:-messages}"

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 输入文件不存在: $INPUT_FILE"
    echo "使用方法: bash scripts/preprocess_dataset.sh [输入文件] [输出文件] [输入字段名] [输出字段名]"
    exit 1
fi

# 创建输出目录（如果不存在）
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "已创建输出目录: $OUTPUT_DIR"
fi

echo "=========================================="
echo "数据集预处理"
echo "=========================================="
echo "输入文件: $INPUT_FILE"
echo "输出文件: $OUTPUT_FILE"
echo "输入字段: $INPUT_FIELD"
echo "输出字段: $OUTPUT_FIELD"
echo "=========================================="

# 调用 format_chatml.py 进行数据转换
python data/format_chatml.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --input_field "$INPUT_FIELD" \
    --output_field "$OUTPUT_FIELD"

# 检查处理结果
if [ $? -eq 0 ] && [ -f "$OUTPUT_FILE" ]; then
    echo "=========================================="
    echo "处理完成！"
    echo "输出文件: $OUTPUT_FILE"
    echo "文件大小: $(du -h "$OUTPUT_FILE" | cut -f1)"
    echo "=========================================="
else
    echo "错误: 处理失败，请检查 format_chatml.py 中的数据处理逻辑是否已实现"
    exit 1
fi
