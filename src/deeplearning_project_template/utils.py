"""工具函数"""

import json
import yaml
import torch
from pathlib import Path
from typing import Tuple, Optional
from loguru import logger
from transformers import AutoTokenizer

from deeplearning_project_template.config import ModelConfig
from deeplearning_project_template.model import TransformerModel


def load_checkpoint(
    checkpoint_dir: str, device: str = "cpu", load_optimizer: bool = False
) -> Tuple[TransformerModel, AutoTokenizer, Optional[dict], dict]:
    """从检查点加载模型和 tokenizer

    Args:
        checkpoint_dir: 检查点目录路径
        device: 设备（'cuda', 'cpu' 等）
        load_optimizer: 是否加载优化器状态

    Returns:
        model: 加载的模型
        tokenizer: 加载的分词器
        optimizer_state: 优化器状态字典（如果 load_optimizer=True）
        trainer_state: 训练状态字典

    Example:
        >>> model, tokenizer, _, state = load_checkpoint("outputs/best_model")
        >>> print(f"Best accuracy: {state['best_metric']:.4f}")
    """
    checkpoint_path = Path(checkpoint_dir)

    # 加载模型配置
    config_path = checkpoint_path / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    model_config = ModelConfig(**config_dict)

    # 创建模型
    model = TransformerModel(model_config)

    # 加载模型权重
    model_path = checkpoint_path / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 加载 tokenizer
    tokenizer_path = checkpoint_path / "tokenizer"
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path}\n"
            "Please retrain the model to save tokenizer with the checkpoint."
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 加载优化器状态（可选）
    optimizer_state = None
    if load_optimizer:
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer_path.exists():
            optimizer_state = torch.load(optimizer_path, map_location=device)

    # 加载训练状态
    trainer_state_path = checkpoint_path / "trainer_state.pt"
    trainer_state = {}
    if trainer_state_path.exists():
        trainer_state = torch.load(trainer_state_path, map_location=device)

    return model, tokenizer, optimizer_state, trainer_state


def load_config_from_file(config_path: str) -> dict:
    """从 YAML 文件加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典

    Example:
        >>> config = load_config_from_file("configs/eval.yaml")
        >>> print(config['batch_size'])
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def setup_device(device: str = "cuda") -> str:
    """设置并验证设备

    Args:
        device: 期望的设备类型（'cuda', 'cpu', 'mps' 等）

    Returns:
        实际可用的设备类型

    Example:
        >>> device = setup_device("cuda")
        >>> print(device)  # 'cuda' 或 'cpu'（如果CUDA不可用）
    """
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS not available, falling back to CPU")
        device = "cpu"

    return device
