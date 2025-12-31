"""
Unsloth LoRA 训练模板
支持模型加载、数据预处理、训练配置、模型保存等功能
"""
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset
from typing import Optional, Dict, Any, List
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
    encoding='utf-8'
)
logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """LoRA配置参数"""
    r: int  # LoRA的rank
    lora_alpha: int  # LoRA的alpha参数
    lora_dropout: float  # LoRA的dropout
    target_modules: Optional[List[str]] = None  # 目标模块列表，None则使用默认值
    bias: str = "none"  # 偏置参数
    use_gradient_checkpointing: str = "unsloth"  # 使用unsloth的梯度检查点，可减少30%显存占用
    random_state: int = 42
    use_rslora: bool = False  # 是否使用rank stabilized LoRA
    loftq_config: Optional[Any] = None  # LoftQ配置
    
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


class UnslothTrainer:
    """Unsloth LoRA 训练类，封装模型训练功能"""
    
    def __init__(
        self,
        model_path: str,
        output_path: str,
        max_seq_length: int,
        load_in_4bit: bool,
        load_in_8bit: bool,
        device_map: str,
        dtype: Optional[str] = None,
        lora_config: Optional[LoRAConfig] = None,
        training_config: Optional[TrainingConfig] = None,
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
        
        self.lora_config = lora_config
        self.training_config = training_config
        
        # 模型和训练器
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._is_loaded = False
        
    def _resolve_model_path(self) -> str:
        """解析模型路径，优先使用本地路径"""
        if os.path.exists(self.model_path) and os.path.exists(
            os.path.join(self.model_path, 'config.json')
        ):
            logger.info(f"使用本地模型: {self.model_path}")
            return self.model_path
        else:
            logger.info(f"本地模型不存在，使用HuggingFace模型: {self.model_path}")
            return self.model_path
    
    def load_model(self):
        """加载模型和分词器，并配置LoRA"""
        try:
            model_name = self._resolve_model_path()
            
            logger.info("正在加载基础模型和分词器...")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=self.max_seq_length,
                device_map=self.device_map,
                dtype=self.dtype,
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
            )
            
            logger.info("正在配置LoRA参数...")
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.lora_config.r,
                target_modules=self.lora_config.target_modules,
                lora_alpha=self.lora_config.lora_alpha,
                lora_dropout=self.lora_config.lora_dropout,
                bias=self.lora_config.bias,
                use_gradient_checkpointing=self.lora_config.use_gradient_checkpointing,
                random_state=self.lora_config.random_state,
                use_rslora=self.lora_config.use_rslora,
                loftq_config=self.lora_config.loftq_config,
            )
            
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
            "dataset_text_field": self.training_config.dataset_text_field,
            "per_device_train_batch_size": self.training_config.per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "gradient_accumulation_steps": self.training_config.gradient_accumulation_steps,
            "warmup_steps": self.training_config.warmup_steps,
            "num_train_epochs": self.training_config.num_train_epochs,
            "learning_rate": self.training_config.learning_rate,
            "logging_steps": self.training_config.logging_steps,
            "logging_strategy": self.training_config.logging_strategy,
            "eval_strategy": evaluation_strategy,  # SFTConfig使用eval_strategy而不是evaluation_strategy
            "save_strategy": save_strategy,
            "optim": self.training_config.optim,
            "weight_decay": self.training_config.weight_decay,
            "lr_scheduler_type": self.training_config.lr_scheduler_type,
            "seed": self.training_config.seed,
            "report_to": self.training_config.report_to,
        }
        
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


def create_lora_config_from_dict(config_dict: Dict[str, Any]) -> LoRAConfig:
    """
    从字典创建LoRAConfig对象
    
    Args:
        config_dict: LoRA配置字典
        
    Returns:
        LoRAConfig对象
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
    # 可选字段：如果YAML中有值则使用，否则使用dataclass中的默认值（通过不传递参数实现）
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
    if 'loftq_config' in lora_dict:
        config_kwargs['loftq_config'] = lora_dict['loftq_config']
    
    return LoRAConfig(**config_kwargs)


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
    # 可选字段：如果YAML中有值则使用，否则使用dataclass中的默认值（通过不传递参数实现）
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
        'save_total_limit', 'max_samples', 'shuffle_seed'
    ]
    for field in optional_fields:
        if field in train_dict:
            config_kwargs[field] = train_dict[field]
    
    return TrainingConfig(**config_kwargs)


def main():
    """
    主函数：从YAML配置文件读取参数并执行训练流程
    """
    parser = argparse.ArgumentParser(description='Unsloth LoRA 训练脚本')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/unsloth/qwen3_8b_lora.yaml',
        help='YAML配置文件路径'
    )
    args = parser.parse_args()
    
    # 从YAML文件加载配置
    config_dict = load_config_from_yaml(args.config)
    
    # 提取基础配置（所有参数都必须从YAML中读取）
    required_base_fields = ['model_path', 'output_path', 'max_seq_length', 'load_in_4bit', 'load_in_8bit', 'device_map']
    for field in required_base_fields:
        if field not in config_dict:
            raise ValueError(f"YAML配置文件中缺少必需的基础配置参数: {field}")
    
    model_path = config_dict['model_path']
    output_path = config_dict['output_path']
    max_seq_length = config_dict['max_seq_length']
    load_in_4bit = config_dict['load_in_4bit']
    load_in_8bit = config_dict['load_in_8bit']
    device_map = config_dict['device_map']
    dtype = config_dict.get('dtype')  # dtype是可选的
    
    # 创建LoRA配置和训练配置
    lora_config = create_lora_config_from_dict(config_dict)
    training_config = create_training_config_from_dict(config_dict)
    
    # 创建训练器并执行训练
    trainer = UnslothTrainer(
        model_path=model_path,
        output_path=output_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        device_map=device_map,
        dtype=dtype,
        lora_config=lora_config,
        training_config=training_config,
    )
    
    # 运行完整训练流程
    metrics = trainer.run_full_pipeline()
    
    logger.info("训练流程完成！")


if __name__ == "__main__":
    main()


