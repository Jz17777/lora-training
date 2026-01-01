"""
Transformers LoRA 训练模板
支持模型加载、数据预处理、训练配置、模型保存等功能
使用 HuggingFace Transformers + PEFT 进行 LoRA 微调
"""
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset
from typing import Optional, Dict, Any, List, Tuple
import torch
import os
import logging
import yaml
import argparse
from dataclasses import dataclass, field

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


@dataclass
class LoRAConfigParams:
    """LoRA配置参数"""
    r: int  # LoRA的rank
    lora_alpha: int  # LoRA的alpha参数
    lora_dropout: float  # LoRA的dropout
    target_modules: Optional[List[str]] = None  # 目标模块列表，None则使用默认值
    bias: str = "none"  # 偏置参数
    use_gradient_checkpointing: bool = True  # 是否使用梯度检查点
    random_state: int = 42
    use_rslora: bool = False  # 是否使用rank stabilized LoRA
    task_type: str = "CAUSAL_LM"  # 任务类型
    
    def __post_init__(self):
        """初始化默认target_modules"""
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


@dataclass
class TrainingConfig:
    """训练配置参数"""
    dataset_path: str  # 训练集路径
    per_device_train_batch_size: int  # 每个设备的训练批次大小
    gradient_accumulation_steps: int  # 梯度累积步数
    num_train_epochs: int  # 训练轮数
    learning_rate: float  # 学习率
    warmup_steps: int  # 预热步数
    weight_decay: float  # 权重衰减
    lr_scheduler_type: str  # 学习率调度器类型
    optim: str  # 优化器类型
    logging_steps: int  # 日志步数
    logging_strategy: str  # 日志策略
    report_to: str  # 日志报告工具
    seed: int  # 随机种子
    
    # 可选字段
    eval_dataset_path: Optional[str] = None  # 验证集路径，None表示不使用验证集
    dataset_text_field: str = "text"  # SFTTrainer需要的文本字段名
    per_device_eval_batch_size: Optional[int] = None  # 验证集批次大小，None则使用训练批次大小
    evaluation_strategy: Optional[str] = None  # "no", "steps", "epoch"
    eval_steps: Optional[int] = None  # 评估步数，仅在evaluation_strategy="steps"时有效
    save_strategy: Optional[str] = None  # "no", "steps", "epoch"
    save_steps: Optional[int] = None  # 保存步数，仅在save_strategy="steps"时有效
    load_best_model_at_end: bool = True  # 训练结束时加载最佳模型
    metric_for_best_model: str = "eval_loss"  # 用于选择最佳模型的指标
    greater_is_better: bool = False  # 指标是否越大越好（loss越小越好）
    save_total_limit: Optional[int] = None  # 保存的检查点数量限制
    max_samples: Optional[int] = None  # 限制训练样本数量，用于快速测试
    shuffle_seed: Optional[int] = None  # 数据打乱种子，None则使用seed
    use_chat_template: bool = False  # 是否使用模型的 chat template 处理数据
    messages_field: str = "messages"  # 数据集中包含消息列表的字段名，仅在 use_chat_template=True 时使用
    packing: bool = False  # 是否启用 packing（将多个短序列打包到同一批次以提高训练效率）
    max_grad_norm: Optional[float] = None  # 梯度裁剪的最大范数，None表示不进行梯度裁剪
    bf16: bool = True  # 是否使用 bf16 混合精度训练
    fp16: bool = False  # 是否使用 fp16 混合精度训练


class TransformersTrainer:
    """Transformers LoRA 训练类，封装模型训练功能"""
    
    def __init__(
        self,
        model_path: str,
        output_path: str,
        max_seq_length: int,
        load_in_4bit: bool,
        load_in_8bit: bool,
        device_map: str,
        dtype: Optional[str] = None,
        lora_config: Optional[LoRAConfigParams] = None,
        training_config: Optional[TrainingConfig] = None,
        attn_implementation: Optional[str] = None,
    ):
        """
        初始化训练器
        
        Args:
            model_path: 模型路径（本地路径或HuggingFace模型名称）
            output_path: 模型保存路径
            max_seq_length: 最大序列长度
            load_in_4bit: 是否使用4bit量化
            load_in_8bit: 是否使用8bit量化
            device_map: 设备映射方式
            dtype: 数据类型，None表示自动选择
            lora_config: LoRA配置，必须提供
            training_config: 训练配置，必须提供
            attn_implementation: 注意力实现方式，如 "flash_attention_2"
        """
        if lora_config is None:
            raise ValueError("lora_config必须提供，不能为None")
        if training_config is None:
            raise ValueError("training_config必须提供，不能为None")
        
        self.model_path = model_path
        self.output_path = output_path
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.device_map = device_map
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        
        self.lora_config = lora_config
        self.training_config = training_config
        
        # 模型和训练器
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._is_loaded = False
        
    def _resolve_model_path(self) -> Tuple[str, bool]:
        """解析模型路径，优先使用本地路径
        
        Returns:
            tuple: (model_path, is_local) - 模型路径和是否为本地路径
        """
        if os.path.exists(self.model_path) and os.path.exists(
            os.path.join(self.model_path, 'config.json')
        ):
            logger.info(f"使用本地模型: {self.model_path}")
            return self.model_path, True
        else:
            logger.info(f"本地模型不存在，使用HuggingFace模型: {self.model_path}")
            return self.model_path, False
    
    def _get_torch_dtype(self) -> torch.dtype:
        """获取 torch 数据类型"""
        if self.dtype is None:
            # 自动选择：优先使用 bfloat16（如果支持），否则使用 float16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                logger.info("自动选择数据类型: bfloat16")
                return torch.bfloat16
            logger.info("自动选择数据类型: float16")
            return torch.float16
        
        dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
        }
        if self.dtype not in dtype_map:
            logger.warning(f"未知的数据类型 '{self.dtype}'，使用默认值 float16")
            return torch.float16
        logger.info(f"使用指定的数据类型: {self.dtype}")
        return dtype_map[self.dtype]
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """获取量化配置"""
        if self.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self._get_torch_dtype(),
                bnb_4bit_use_double_quant=True,
            )
        elif self.load_in_8bit:
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        return None
    
    def load_model(self):
        """加载模型和分词器，并配置LoRA"""
        try:
            model_name, is_local = self._resolve_model_path()
            
            # 如果是本地模型路径，设置 local_files_only=True 以避免从 HuggingFace 下载
            local_files_only = is_local
            
            if local_files_only:
                logger.info("检测到本地模型，将使用 local_files_only=True 避免网络下载")
            else:
                logger.info("使用远程模型，将从 HuggingFace 下载（如果需要）")
            
            # 加载分词器
            logger.info("正在加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            
            # 设置 padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # 获取量化配置
            quantization_config = self._get_quantization_config()
            
            # 构建模型加载参数
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": self.device_map,
                "local_files_only": local_files_only,
            }
            
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
            else:
                model_kwargs["torch_dtype"] = self._get_torch_dtype()
            
            if self.attn_implementation is not None:
                model_kwargs["attn_implementation"] = self.attn_implementation
            
            # 加载模型
            logger.info("正在加载基础模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # 如果使用量化，准备模型进行 k-bit 训练
            if self.load_in_4bit or self.load_in_8bit:
                logger.info("正在准备模型进行量化训练...")
                self.model = prepare_model_for_kbit_training(
                    self.model,
                    use_gradient_checkpointing=self.lora_config.use_gradient_checkpointing,
                )
            elif self.lora_config.use_gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            
            # 配置 LoRA
            logger.info("正在配置LoRA参数...")
            peft_config = LoraConfig(
                r=self.lora_config.r,
                lora_alpha=self.lora_config.lora_alpha,
                lora_dropout=self.lora_config.lora_dropout,
                target_modules=self.lora_config.target_modules,
                bias=self.lora_config.bias,
                task_type=TaskType.CAUSAL_LM,
                use_rslora=self.lora_config.use_rslora,
            )
            
            self.model = get_peft_model(self.model, peft_config)
            
            # 打印可训练参数信息
            self.model.print_trainable_parameters()
            
            self._is_loaded = True
            logger.info("模型加载完成")
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def prepare_dataset(
        self,
        dataset_path: Optional[str] = None,
        is_eval: bool = False
    ) -> Dataset:
        """
        准备数据集
        
        Args:
            dataset_path: 数据集路径，None则使用training_config中的路径
            is_eval: 是否为验证集
            
        Returns:
            处理后的数据集
        """
        if dataset_path is None:
            dataset_path = (
                self.training_config.eval_dataset_path if is_eval
                else self.training_config.dataset_path
            )
        
        if dataset_path is None:
            return None
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")
        
        logger.info(f"正在加载数据集: {dataset_path}")
        dataset = load_dataset('json', data_files=dataset_path)['train']
        
        # 如果启用 chat template，使用模型的 chat template 处理数据
        if self.training_config.use_chat_template:
            if not self._is_loaded:
                raise RuntimeError("使用 chat template 需要先加载模型，请先调用 load_model()")
            
            logger.info(f"正在使用模型的 chat template 处理数据集（messages字段: {self.training_config.messages_field}）...")
            
            # 检查 chat template 是否存在
            if self.tokenizer.chat_template is None:
                raise ValueError("模型没有定义 chat_template，无法处理数据。请检查模型配置")
            
            # 定义格式化函数
            def formatting_prompts_func(examples):
                """使用 chat template 格式化消息"""
                messages_field = self.training_config.messages_field
                
                # 获取消息列表
                if messages_field not in examples:
                    raise ValueError(f"数据集中缺少 '{messages_field}' 字段")
                
                convos = examples[messages_field]
                
                # 处理每个对话
                texts = []
                for convo in convos:
                    if not isinstance(convo, list):
                        raise ValueError(f"'{messages_field}' 字段必须是消息列表")
                    
                    # 验证消息格式
                    for i, msg in enumerate(convo):
                        if not isinstance(msg, dict):
                            raise ValueError(f"第 {i+1} 条消息必须是字典格式")
                        if "role" not in msg or "content" not in msg:
                            raise ValueError(f"第 {i+1} 条消息必须包含 'role' 和 'content' 字段")
                    
                    # 使用 chat template 格式化
                    text = self.tokenizer.apply_chat_template(
                        convo,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    texts.append(text)
                
                return {self.training_config.dataset_text_field: texts}
            
            # 应用格式化函数
            dataset = dataset.map(
                formatting_prompts_func,
                batched=True,
                desc="使用 chat template 处理数据"
            )
            logger.info("Chat template 处理完成")
        
        # 限制样本数量（用于快速测试）
        if not is_eval and self.training_config.max_samples:
            shuffle_seed = self.training_config.shuffle_seed or self.training_config.seed
            dataset = dataset.shuffle(seed=shuffle_seed).select(
                range(self.training_config.max_samples)
            )
            logger.info(f"限制训练样本数量为: {self.training_config.max_samples}")
        
        logger.info(f"数据集准备完成，样本数量: {len(dataset)}")
        if len(dataset) > 0:
            logger.debug(f"示例数据: {dataset[0]}")
        
        return dataset
    
    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """设置训练器"""
        if not self._is_loaded:
            raise RuntimeError("模型未加载，请先调用load_model()")
        
        logger.info("正在配置训练器...")
        
        # 配置评估和保存策略
        has_eval_dataset = eval_dataset is not None
        evaluation_strategy = self.training_config.evaluation_strategy
        save_strategy = self.training_config.save_strategy
        
        if has_eval_dataset:
            # 有验证集时的默认策略
            if evaluation_strategy is None:
                evaluation_strategy = "epoch"
                logger.info("检测到验证集，设置评估策略为'epoch'")
            elif evaluation_strategy == "steps":
                if self.training_config.eval_steps is None:
                    raise ValueError("当evaluation_strategy='steps'时，必须设置eval_steps参数")
            
            if save_strategy is None:
                save_strategy = "epoch"
                logger.info("检测到验证集，设置保存策略为'epoch'")
            elif save_strategy == "steps":
                if self.training_config.save_steps is None:
                    raise ValueError("当save_strategy='steps'时，必须设置save_steps参数")
        else:
            # 无验证集时的策略
            if evaluation_strategy is None:
                evaluation_strategy = "no"
            elif evaluation_strategy != "no":
                logger.warning(f"检测到evaluation_strategy='{evaluation_strategy}'，但没有提供验证集，将自动设置为'no'")
                evaluation_strategy = "no"
            
            if save_strategy is None:
                save_strategy = "steps"
            elif save_strategy == "steps" and self.training_config.save_steps is None:
                raise ValueError("当save_strategy='steps'时，必须设置save_steps参数")
        
        # 验证集批次大小
        per_device_eval_batch_size = (
            self.training_config.per_device_eval_batch_size 
            if self.training_config.per_device_eval_batch_size is not None
            else self.training_config.per_device_train_batch_size
        )
        
        # 创建训练配置
        training_args_dict = {
            "output_dir": self.output_path,
            "per_device_train_batch_size": self.training_config.per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "gradient_accumulation_steps": self.training_config.gradient_accumulation_steps,
            "warmup_steps": self.training_config.warmup_steps,
            "num_train_epochs": self.training_config.num_train_epochs,
            "learning_rate": self.training_config.learning_rate,
            "logging_steps": self.training_config.logging_steps,
            "logging_strategy": self.training_config.logging_strategy,
            "eval_strategy": evaluation_strategy,
            "save_strategy": save_strategy,
            "optim": self.training_config.optim,
            "weight_decay": self.training_config.weight_decay,
            "lr_scheduler_type": self.training_config.lr_scheduler_type,
            "seed": self.training_config.seed,
            "report_to": self.training_config.report_to,
            "bf16": self.training_config.bf16,
            "fp16": self.training_config.fp16,
            "max_seq_length": self.max_seq_length,
            "dataset_text_field": self.training_config.dataset_text_field,
        }
        
        # 添加梯度裁剪配置
        if self.training_config.max_grad_norm is not None:
            training_args_dict["max_grad_norm"] = self.training_config.max_grad_norm
        
        # 添加评估和保存步数配置
        if evaluation_strategy == "steps" and self.training_config.eval_steps is not None:
            training_args_dict["eval_steps"] = self.training_config.eval_steps
        
        if save_strategy == "steps" and self.training_config.save_steps is not None:
            training_args_dict["save_steps"] = self.training_config.save_steps
        
        # 添加最佳模型相关配置
        if has_eval_dataset:
            training_args_dict["load_best_model_at_end"] = self.training_config.load_best_model_at_end
            training_args_dict["metric_for_best_model"] = self.training_config.metric_for_best_model
            training_args_dict["greater_is_better"] = self.training_config.greater_is_better
            
            if self.training_config.save_total_limit is not None:
                training_args_dict["save_total_limit"] = self.training_config.save_total_limit
        
        training_args = SFTConfig(**training_args_dict)
        
        # 创建训练器
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            packing=self.training_config.packing,
        )
        
        logger.info("训练器配置完成")
    
    def train(self) -> Dict[str, Any]:
        """
        开始训练
        
        Returns:
            训练统计信息字典
        """
        if self.trainer is None:
            raise RuntimeError("训练器未设置，请先调用 setup_trainer()")
        
        # 显示GPU信息
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            start_gpu_memory = round(
                torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
            )
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
            logger.info(f"{start_gpu_memory} GB of memory reserved.")
        else:
            logger.warning("未检测到CUDA设备，将使用CPU训练（速度较慢）")
            start_gpu_memory = 0
            max_memory = 0
        
        # 开始训练
        logger.info("开始训练...")
        trainer_stats = self.trainer.train()
        
        # 显示训练统计信息
        if torch.cuda.is_available():
            used_memory = round(
                torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
            )
            used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
            used_percentage = round(used_memory / max_memory * 100, 3) if max_memory > 0 else 0
            lora_percentage = round(
                used_memory_for_lora / max_memory * 100, 3
            ) if max_memory > 0 else 0
            
            logger.info("=" * 60)
            logger.info("训练完成！统计信息：")
            logger.info(f"训练时间: {trainer_stats.metrics['train_runtime']:.2f} 秒")
            logger.info(f"训练时间: {round(trainer_stats.metrics['train_runtime']/60, 2):.2f} 分钟")
            logger.info(f"峰值显存占用: {used_memory} GB")
            logger.info(f"训练显存占用: {used_memory_for_lora} GB")
            logger.info(f"峰值显存占用百分比: {used_percentage} %")
            logger.info(f"训练显存占用百分比: {lora_percentage} %")
            logger.info("=" * 60)
        
        return trainer_stats.metrics
    
    def save_model(self, output_path: Optional[str] = None):
        """
        保存模型和分词器
        
        Args:
            output_path: 保存路径，None则使用初始化时的output_path
        """
        if not self._is_loaded:
            raise RuntimeError("模型未加载")
        
        save_path = output_path or self.output_path
        os.makedirs(save_path, exist_ok=True)
        
        logger.info(f"正在保存模型到: {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info("模型保存完成")
    
    def run_full_pipeline(self):
        """
        运行完整的训练流程
        包括：模型加载、数据集准备、训练器设置、训练、模型保存
        
        Returns:
            训练统计信息字典
        """
        self.load_model()
        
        train_dataset = self.prepare_dataset(is_eval=False)
        eval_dataset = None
        if self.training_config.eval_dataset_path:
            eval_dataset = self.prepare_dataset(is_eval=True)
        
        self.setup_trainer(train_dataset, eval_dataset)
        metrics = self.train()
        self.save_model()
        
        return metrics


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """
    从YAML配置文件加载配置
    
    Args:
        config_path: YAML配置文件路径
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"已加载配置文件: {config_path}")
    return config


def create_lora_config_from_dict(config_dict: Dict[str, Any]) -> LoRAConfigParams:
    """
    从字典创建LoRAConfigParams对象
    
    Args:
        config_dict: LoRA配置字典
        
    Returns:
        LoRAConfigParams对象
    """
    lora_dict = config_dict.get('lora', {})
    
    # 必需字段
    if 'r' not in lora_dict:
        raise ValueError("YAML配置文件中缺少必需的LoRA配置参数: r")
    if 'lora_alpha' not in lora_dict:
        raise ValueError("YAML配置文件中缺少必需的LoRA配置参数: lora_alpha")
    if 'lora_dropout' not in lora_dict:
        raise ValueError("YAML配置文件中缺少必需的LoRA配置参数: lora_dropout")
    
    # 处理target_modules，如果为None或空列表则使用None（将在__post_init__中设置默认值）
    target_modules = lora_dict.get('target_modules')
    if isinstance(target_modules, list) and len(target_modules) == 0:
        target_modules = None
    
    # 必需字段直接从YAML读取
    config_kwargs = {
        'r': lora_dict['r'],
        'lora_alpha': lora_dict['lora_alpha'],
        'lora_dropout': lora_dict['lora_dropout'],
    }
    
    # 只传递YAML中存在的可选字段
    if 'target_modules' in lora_dict:
        config_kwargs['target_modules'] = target_modules
    if 'bias' in lora_dict:
        config_kwargs['bias'] = lora_dict['bias']
    if 'use_gradient_checkpointing' in lora_dict:
        config_kwargs['use_gradient_checkpointing'] = lora_dict['use_gradient_checkpointing']
    if 'random_state' in lora_dict:
        config_kwargs['random_state'] = lora_dict['random_state']
    if 'use_rslora' in lora_dict:
        config_kwargs['use_rslora'] = lora_dict['use_rslora']
    if 'task_type' in lora_dict:
        config_kwargs['task_type'] = lora_dict['task_type']
    
    return LoRAConfigParams(**config_kwargs)


def create_training_config_from_dict(config_dict: Dict[str, Any]) -> TrainingConfig:
    """
    从字典创建TrainingConfig对象
    
    Args:
        config_dict: 训练配置字典
        
    Returns:
        TrainingConfig对象
    """
    train_dict = config_dict.get('training', {})
    
    # 必需字段检查
    required_fields = [
        'dataset_path', 'per_device_train_batch_size', 'gradient_accumulation_steps',
        'num_train_epochs', 'learning_rate', 'warmup_steps', 'weight_decay',
        'lr_scheduler_type', 'optim', 'logging_steps', 'logging_strategy',
        'report_to', 'seed'
    ]
    for field in required_fields:
        if field not in train_dict:
            raise ValueError(f"YAML配置文件中缺少必需的训练配置参数: {field}")
    
    # 必需字段直接从YAML读取
    config_kwargs = {
        'dataset_path': train_dict['dataset_path'],
        'per_device_train_batch_size': train_dict['per_device_train_batch_size'],
        'gradient_accumulation_steps': train_dict['gradient_accumulation_steps'],
        'num_train_epochs': train_dict['num_train_epochs'],
        'learning_rate': train_dict['learning_rate'],
        'warmup_steps': train_dict['warmup_steps'],
        'weight_decay': train_dict['weight_decay'],
        'lr_scheduler_type': train_dict['lr_scheduler_type'],
        'optim': train_dict['optim'],
        'logging_steps': train_dict['logging_steps'],
        'logging_strategy': train_dict['logging_strategy'],
        'report_to': train_dict['report_to'],
        'seed': train_dict['seed'],
    }
    
    # 只传递YAML中存在的可选字段
    optional_fields = [
        'eval_dataset_path', 'dataset_text_field', 'per_device_eval_batch_size',
        'evaluation_strategy', 'eval_steps', 'save_strategy', 'save_steps',
        'load_best_model_at_end', 'metric_for_best_model', 'greater_is_better',
        'save_total_limit', 'max_samples', 'shuffle_seed',
        'use_chat_template', 'messages_field',
        'packing', 'max_grad_norm', 'bf16', 'fp16'
    ]
    for field in optional_fields:
        if field in train_dict:
            config_kwargs[field] = train_dict[field]
    
    return TrainingConfig(**config_kwargs)


def main():
    """
    主函数：从YAML配置文件读取参数并执行训练流程
    """
    parser = argparse.ArgumentParser(description='Transformers LoRA 训练脚本')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/transformers/qwen3_8b_lora.yaml',
        help='YAML配置文件路径'
    )
    args = parser.parse_args()
    
    # 从YAML文件加载配置
    try:
        config_dict = load_config_from_yaml(args.config)
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        raise
    
    # 提取基础配置（所有参数都必须从YAML中读取）
    required_base_fields = ['model_path', 'output_path', 'max_seq_length', 'load_in_4bit', 'load_in_8bit', 'device_map']
    for field in required_base_fields:
        if field not in config_dict:
            raise ValueError(f"YAML配置文件中缺少必需的基础配置参数: {field}")
    
    # 验证量化配置的合理性
    load_in_4bit = config_dict['load_in_4bit']
    load_in_8bit = config_dict['load_in_8bit']
    if load_in_4bit and load_in_8bit:
        raise ValueError("load_in_4bit 和 load_in_8bit 不能同时为 true，请只选择一个")
    
    # 验证 max_seq_length
    max_seq_length = config_dict['max_seq_length']
    if not isinstance(max_seq_length, int) or max_seq_length <= 0:
        raise ValueError(f"max_seq_length 必须是正整数，当前值: {max_seq_length}")
    
    model_path = config_dict['model_path']
    output_path = config_dict['output_path']
    device_map = config_dict['device_map']
    dtype = config_dict.get('dtype')  # dtype是可选的
    attn_implementation = config_dict.get('attn_implementation')  # 注意力实现方式
    
    # 创建LoRA配置和训练配置
    lora_config = create_lora_config_from_dict(config_dict)
    training_config = create_training_config_from_dict(config_dict)
    
    # 创建训练器并执行训练
    try:
        trainer = TransformersTrainer(
            model_path=model_path,
            output_path=output_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            device_map=device_map,
            dtype=dtype,
            lora_config=lora_config,
            training_config=training_config,
            attn_implementation=attn_implementation,
        )
        
        # 运行完整训练流程
        logger.info("=" * 60)
        logger.info("开始执行训练流程")
        logger.info("=" * 60)
        metrics = trainer.run_full_pipeline()
        
        logger.info("=" * 60)
        logger.info("训练流程完成！")
        logger.info("=" * 60)
    except KeyboardInterrupt:
        logger.warning("训练被用户中断")
        raise
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        logger.exception("详细错误信息:")
        raise


if __name__ == "__main__":
    main()
