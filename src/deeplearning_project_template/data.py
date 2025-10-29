"""数据处理
关键点：DataModule 接收 DataConfig 对象
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
from deeplearning_project_template.config import DataConfig


class TextDataset(Dataset):
    """文本分类数据集"""

    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer,
        max_length: int,
    ):
        """
        Args:
            file_path: 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 加载数据
        with open(file_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 分词
        encoding = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


class DataModule:
    """数据模块

    封装所有数据相关的逻辑
    接收 DataConfig 对象
    """

    def __init__(self, config: DataConfig, tokenizer: AutoTokenizer):
        """
        Args:
            config: DataConfig 对象
            tokenizer: 分词器
        """
        self.config = config
        self.tokenizer = tokenizer

        # 创建数据集
        self.train_dataset = TextDataset(
            file_path=config.train_file,
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
        )

        self.val_dataset = TextDataset(
            file_path=config.validation_file,
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
        )

        if config.test_file:
            self.test_dataset = TextDataset(
                file_path=config.test_file,
                tokenizer=tokenizer,
                max_length=config.max_seq_length,
            )
        else:
            self.test_dataset = None

    def train_dataloader(self) -> DataLoader:
        """返回训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """返回验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """返回测试数据加载器"""
        if self.test_dataset is None:
            raise ValueError("No test dataset available")

        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
