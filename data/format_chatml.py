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
    handlers=[
        logging.StreamHandler()  # 输出到控制台
    ]
)
logger = logging.getLogger(__name__)
# 确保日志输出使用 UTF-8 编码
import sys
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    import io
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


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
                # 数据处理逻辑部分
                # =============================================================================
                # 支持多种输入格式，转换为标准的 messages 格式
                messages = None
                
                # 情况1: 如果输入数据已经有 messages 字段（标准格式），直接使用
                if output_field in data and isinstance(data[output_field], list):
                    messages = data[output_field]
                elif "messages" in data and isinstance(data["messages"], list):
                    messages = data["messages"]
                
                # 情况2: 如果输入数据有 input_field 指定的字段（如 conversations）
                elif input_field in data:
                    conversations = data[input_field]
                    if isinstance(conversations, list):
                        messages = []
                        for item in conversations:
                            if isinstance(item, dict):
                                # 如果已经是标准格式（有 role 和 content）
                                if "role" in item and "content" in item:
                                    messages.append({
                                        "role": item["role"],
                                        "content": item["content"]
                                    })
                                # 如果是其他格式，尝试提取
                                elif "from" in item and "value" in item:
                                    # 支持类似 {"from": "user", "value": "..."} 格式
                                    role = item["from"]
                                    # 将 "human" 转换为 "user"，"gpt" 转换为 "assistant"
                                    if role == "human":
                                        role = "user"
                                    elif role == "gpt" or role == "bot":
                                        role = "assistant"
                                    messages.append({
                                        "role": role,
                                        "content": item["value"]
                                    })
                                elif "speaker" in item and "text" in item:
                                    # 支持类似 {"speaker": "user", "text": "..."} 格式
                                    role = item["speaker"]
                                    if role == "human":
                                        role = "user"
                                    elif role == "gpt" or role == "bot":
                                        role = "assistant"
                                    messages.append({
                                        "role": role,
                                        "content": item["text"]
                                    })
                
                # 情况3: 如果输入数据有 instruction 和 output 字段（单轮对话格式）
                elif "instruction" in data:
                    messages = []
                    if "input" in data and data["input"]:
                        # 有 input 字段的情况
                        messages.append({
                            "role": "user",
                            "content": f"{data['instruction']}\n{data['input']}"
                        })
                    else:
                        # 只有 instruction 的情况
                        messages.append({
                            "role": "user",
                            "content": data["instruction"]
                        })
                    if "output" in data:
                        messages.append({
                            "role": "assistant",
                            "content": data["output"]
                        })
                
                # 情况4: 如果输入数据有 prompt 和 response 字段
                elif "prompt" in data:
                    messages = []
                    messages.append({
                        "role": "user",
                        "content": data["prompt"]
                    })
                    if "response" in data:
                        messages.append({
                            "role": "assistant",
                            "content": data["response"]
                        })
                
                # 情况5: 如果输入数据有 question 和 answer 字段
                elif "question" in data:
                    messages = []
                    messages.append({
                        "role": "user",
                        "content": data["question"]
                    })
                    if "answer" in data:
                        messages.append({
                            "role": "assistant",
                            "content": data["answer"]
                        })
                
                # 如果以上都不匹配，尝试从 data 中查找可能的对话字段
                if messages is None:
                    # 尝试查找常见的对话字段
                    possible_fields = ["conversation", "dialogue", "chat", "history", "context"]
                    for field in possible_fields:
                        if field in data and isinstance(data[field], list):
                            # 递归处理，假设这个字段包含对话数据
                            temp_data = {input_field: data[field]}
                            # 这里简化处理，假设格式类似 conversations
                            messages = []
                            for item in data[field]:
                                if isinstance(item, dict) and "role" in item and "content" in item:
                                    messages.append({
                                        "role": item["role"],
                                        "content": item["content"]
                                    })
                            if messages:
                                break
                
                # 验证消息格式
                if messages is None or not isinstance(messages, list) or len(messages) == 0:
                    logger.warning(f"第 {line_num} 行：无法提取有效消息，跳过。数据键: {list(data.keys())[:5]}")
                    error_count += 1
                    continue
                
                # 验证每条消息的格式
                valid_messages = []
                for i, msg in enumerate(messages):
                    if not isinstance(msg, dict):
                        logger.warning(f"第 {line_num} 行第 {i+1} 条消息格式不正确（不是字典），跳过该样本")
                        break
                    if "role" not in msg or "content" not in msg:
                        logger.warning(f"第 {line_num} 行第 {i+1} 条消息缺少 'role' 或 'content' 字段，跳过该样本")
                        break
                    # 验证并规范化 role 值
                    role = msg["role"].lower() if isinstance(msg["role"], str) else str(msg["role"]).lower()
                    # 角色映射：将常见变体转换为标准值
                    role_mapping = {
                        "human": "user",
                        "gpt": "assistant",
                        "bot": "assistant",
                        "assistant": "assistant",
                        "user": "user",
                        "system": "system",
                        "ai": "assistant",
                        "chatgpt": "assistant",
                    }
                    if role in role_mapping:
                        role = role_mapping[role]
                    elif role not in ["user", "assistant", "system"]:
                        logger.warning(f"第 {line_num} 行第 {i+1} 条消息的 role 值 '{msg['role']}' 不是标准值（user/assistant/system），将跳过")
                        break
                    # 更新 role 为标准值
                    msg["role"] = role
                    # 验证 content 是否为空
                    if not msg["content"] or not str(msg["content"]).strip():
                        logger.warning(f"第 {line_num} 行第 {i+1} 条消息的 content 为空，跳过该样本")
                        break
                    valid_messages.append({
                        "role": msg["role"],
                        "content": str(msg["content"]).strip()
                    })
                else:
                    # 所有消息都验证通过，写入输出文件
                    if len(valid_messages) > 0:
                        output_data = {output_field: valid_messages}
                        f_out.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                        processed_count += 1
                        
                        # 每处理 1000 条记录输出一次进度
                        if processed_count % 1000 == 0:
                            logger.info(f"已处理 {processed_count} 条记录...")
                    else:
                        logger.warning(f"第 {line_num} 行：没有有效消息，跳过")
                        error_count += 1
                        continue
            
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
