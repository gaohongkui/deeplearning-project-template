"""配置类定义
核心原则：
1. 每个模块有自己的 Config dataclass
2. 使用类型注解
3. 提供合理的默认值
4. 在 __post_init__ 中验证
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Literal

# ============ 类型定义 ============
OptimizerType = Literal["adam", "adamw", "sgd"]
SchedulerType = Literal["linear", "cosine", "constant"]


# ============ 模型配置 ============
@dataclass
class ModelConfig:
    """模型结构配置"""

    # 基础参数
    model_type: str = "bert"
    tokenizer_name: str = "bert-base-uncased"  # 与模型配套的 tokenizer
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072

    # Dropout
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    # 其他
    max_position_embeddings: int = 512
    vocab_size: int = 30522

    def __post_init__(self):
        """参数验证"""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )

        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")

    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)


# ============ 数据配置 ============
@dataclass
class DataConfig:
    """数据处理配置"""

    # 数据路径
    train_file: str = "data/train.jsonl"
    validation_file: str = "data/val.jsonl"
    test_file: Optional[str] = None

    # 数据处理参数
    max_seq_length: int = 512
    batch_size: int = 32
    num_workers: int = 4

    # 预处理
    preprocessing_num_workers: int = 4
    overwrite_cache: bool = False

    def __post_init__(self):
        """参数验证"""
        if self.max_seq_length <= 0:
            raise ValueError(
                f"max_seq_length must be positive, got {self.max_seq_length}"
            )

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


# ============ 训练配置 ============
@dataclass
class TrainingConfig:
    """训练过程配置"""

    # 基础训练参数
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # 优化器和调度器
    optimizer_type: OptimizerType = "adamw"
    scheduler_type: SchedulerType = "linear"

    # 训练控制
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # 评估和保存
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 100

    # 早停
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.0

    # 随机种子
    seed: int = 42

    def __post_init__(self):
        """参数验证"""
        import os

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 验证优化器类型
        valid_optimizers = ["adam", "adamw", "sgd"]
        if self.optimizer_type not in valid_optimizers:
            raise ValueError(
                f"Invalid optimizer_type: {self.optimizer_type}. Must be one of {valid_optimizers}"
            )

        # 验证调度器类型
        valid_schedulers = ["linear", "cosine", "constant"]
        if self.scheduler_type not in valid_schedulers:
            raise ValueError(
                f"Invalid scheduler_type: {self.scheduler_type}. Must be one of {valid_schedulers}"
            )


# ============ 实验配置（顶层配置）============
@dataclass
class Config:
    """实验总配置 - 包含所有子配置"""

    # 子配置
    model_config: ModelConfig = field(default_factory=ModelConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)

    # 实验元信息
    experiment_name: str = "default_experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # 设备配置
    device: str = "cuda"
    distributed: bool = False
