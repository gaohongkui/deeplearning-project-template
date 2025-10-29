"""
Author: gaohongkui gaohongkui1021@163.com
Date: 2025-10-29 18:21:52
LastEditors: gaohongkui gaohongkui1021@163.com
FilePath: /deeplearning-project-template/train.py
Description:

Copyright (c) 2025 by gaohongkui, All Rights Reserved.
"""

"""训练入口
关键点：使用 Hydra 自动管理配置
"""

import os

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from transformers import AutoTokenizer
import torch

from deeplearning_project_template.config import (
    Config,
    ModelConfig,
    DataConfig,
    TrainingConfig,
)
from deeplearning_project_template.model import TransformerModel
from deeplearning_project_template.data import DataModule
from deeplearning_project_template.trainer import Trainer


def set_seed(seed: int):
    """设置随机种子"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """主函数

    Args:
        cfg: Hydra 自动注入的配置对象
    """

    # 配置日志记录
    logger.add(
        hydra.core.hydra_config.HydraConfig.get().job_logging.handlers.file.filename
    )

    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    cfg = OmegaConf.to_container(cfg, resolve=True)
    config = Config(
        model_config=ModelConfig(**cfg["model"]),
        data_config=DataConfig(**cfg["data"]),
        training_config=TrainingConfig(**cfg["training"]),
        **{k: v for k, v in cfg.items() if k not in ["model", "data", "training"]},
    )

    # 设置随机种子
    set_seed(config.training_config.seed)

    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 创建模型（传入 ModelConfig 对象）
    model = TransformerModel(config.model_config)
    logger.info(
        f"Model created with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # 创建数据模块（传入 DataConfig 对象）
    data_module = DataModule(config.data_config, tokenizer)
    logger.info(
        f"Data loaded: {len(data_module.train_dataset)} train, "
        f"{len(data_module.val_dataset)} val samples"
    )

    # 创建训练器（传入 TrainingConfig 对象）
    trainer = Trainer(
        model=model,
        data_module=data_module,
        config=config.training_config,
    )

    # 开始训练
    logger.info("Starting training...")
    trainer.train()

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
