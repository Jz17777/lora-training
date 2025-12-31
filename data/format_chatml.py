"""
数据集格式转换工具
将原始数据集转换为 ChatML 格式（包含 messages 字段）
"""
import json
import os
import logging
from typing import List, Dict, Any, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)


def format_to_chatml(
    input_path: str,
    output_path: str,
    input_field: str = "conversations",
    output_field: str = "messages"
) -> None:
    """
    将数据集转换为 ChatML 格式并保存
    
    此函数只负责格式转换，将输入数据中的对话格式转换为标准的 messages 格式。
    实际的 chat template 处理将在训练时由模型完成。
    
    Args:
        input_path: 输入数据集路径（JSONL 格式）
        output_path: 输出数据集路径（JSONL 格式）
        input_field: 输入数据中包含对话的字段名，默认为 "conversations"
        output_field: 输出数据中消息列表的字段名，默认为 "messages"
    
    输出格式要求（每行一个 JSON 对象）:
        {"messages": [{"role": "user", "content": "What is 1+1?"}, {"role": "assistant", "content": "It's 2!"}]}
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    # 读取输入文件并处理
    logger.info(f"正在读取输入文件: {input_path}")
    processed_count = 0
    error_count = 0
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"创建输出目录: {output_dir}")
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # 解析 JSON
                data = json.loads(line)
                
                # =============================================================================
                # TODO: 数据处理逻辑部分
                # =============================================================================
                # 请根据实际的数据集格式，在此处添加相应的处理逻辑
                # 
                # 目标：将输入数据转换为包含 messages 字段的格式
                # 输出格式示例：
                #   {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
                #
                # 处理步骤：
                # 1. 从 data 中提取对话数据（根据 input_field 或其他字段）
                # 2. 将对话数据转换为标准的 messages 格式（列表，每个元素包含 role 和 content）
                # 3. 验证消息格式是否正确
                # 4. 构建输出数据：{output_field: messages}
                #
                # 示例处理逻辑（请根据实际数据集格式修改）：
                # messages = []
                # # 根据实际数据格式提取和转换
                # # ...
                # output_data = {output_field: messages}
                # =============================================================================
                
                # 占位符：等待添加具体的数据处理逻辑
                logger.warning(f"第 {line_num} 行：数据处理逻辑尚未实现，请根据数据集格式添加相应的处理代码")
                logger.warning(f"当前数据示例: {json.dumps(data, ensure_ascii=False)[:200]}...")
                error_count += 1
                continue
                
                # =============================================================================
                # 以下代码在实现处理逻辑后取消注释
                # =============================================================================
                # # 验证消息格式
                # if not isinstance(messages, list) or len(messages) == 0:
                #     logger.warning(f"第 {line_num} 行没有有效消息，跳过")
                #     error_count += 1
                #     continue
                # 
                # # 验证每条消息的格式
                # for i, msg in enumerate(messages):
                #     if not isinstance(msg, dict):
                #         logger.warning(f"第 {line_num} 行第 {i+1} 条消息格式不正确，跳过该样本")
                #         break
                #     if "role" not in msg or "content" not in msg:
                #         logger.warning(f"第 {line_num} 行第 {i+1} 条消息缺少 'role' 或 'content' 字段，跳过该样本")
                #         break
                # else:
                #     # 所有消息都验证通过，写入输出文件
                #     output_data = {output_field: messages}
                #     f_out.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                #     processed_count += 1
                #     
                #     # 每处理 1000 条记录输出一次进度
                #     if processed_count % 1000 == 0:
                #         logger.info(f"已处理 {processed_count} 条记录...")
                # =============================================================================
            
            except json.JSONDecodeError as e:
                logger.warning(f"第 {line_num} 行 JSON 解析失败: {str(e)}，跳过")
                error_count += 1
            except Exception as e:
                logger.warning(f"第 {line_num} 行处理失败: {str(e)}，跳过")
                error_count += 1
    
    logger.info("=" * 60)
    logger.info("格式转换完成！")
    logger.info(f"成功处理: {processed_count} 条记录")
    logger.info(f"失败/跳过: {error_count} 条记录")
    logger.info(f"输出文件: {output_path}")
    logger.info("=" * 60)


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='将数据集转换为 ChatML 格式（包含 messages 字段）')
    parser.add_argument('--input', type=str, required=True, help='输入数据集路径（JSONL 格式）')
    parser.add_argument('--output', type=str, required=True, help='输出数据集路径（JSONL 格式）')
    parser.add_argument('--input_field', type=str, default='conversations', 
                       help='输入数据中包含对话的字段名（默认: conversations）')
    parser.add_argument('--output_field', type=str, default='messages', 
                       help='输出数据中消息列表的字段名（默认: messages）')
    
    args = parser.parse_args()
    
    format_to_chatml(
        input_path=args.input,
        output_path=args.output,
        input_field=args.input_field,
        output_field=args.output_field
    )


if __name__ == "__main__":
    main()
