"""训练器
关键点：Trainer 接收多个 config 对象
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from loguru import logger
from pathlib import Path

from deeplearning_project_template.config import TrainingConfig
from deeplearning_project_template.model import TransformerModel
from deeplearning_project_template.data import DataModule


class Trainer:
    """训练器

    负责模型训练的所有逻辑
    接收 TrainingConfig 对象
    """

    def __init__(
        self,
        model: TransformerModel,
        data_module: DataModule,
        config: TrainingConfig,
    ):
        """
        Args:
            model: 模型实例
            data_module: 数据模块实例
            config: TrainingConfig 对象
        """
        self.model = model
        self.data_module = data_module
        self.config = config

        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 设置优化器
        self.optimizer = self._create_optimizer()

        # 设置学习率调度器
        self.scheduler = self._create_scheduler()

        # 训练状态
        self.global_step = 0
        self.best_metric = float("-inf")
        self.patience_counter = 0

    def _create_optimizer(self):
        """根据配置创建优化器"""

        # 分离需要 weight decay 的参数
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        # 根据配置选择优化器
        if self.config.optimizer_type == "adamw":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
            )
        elif self.config.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
            )
        elif self.config.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer_type}")

        return optimizer

    def _create_scheduler(self):
        """根据配置创建学习率调度器"""

        train_dataloader = self.data_module.train_dataloader()
        num_training_steps = (
            len(train_dataloader)
            * self.config.num_train_epochs
            // self.config.gradient_accumulation_steps
        )
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        if self.config.scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif self.config.scheduler_type == "constant":
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0,
            )
        elif self.config.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps,
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler_type}")

        return scheduler

    def train(self):
        """训练循环"""
        logger.info("***** Running training *****")
        logger.info(f"  Num epochs = {self.config.num_train_epochs}")
        logger.info(f"  Batch size = {self.data_module.config.batch_size}")
        logger.info(f"  Learning rate = {self.config.learning_rate}")

        for epoch in range(self.config.num_train_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_train_epochs}")

            # 训练一个 epoch
            train_loss = self._train_epoch()
            logger.info(f"Training loss: {train_loss:.4f}")

            # 评估
            eval_metrics = self.evaluate()
            logger.info(f"Evaluation metrics: {eval_metrics}")

            # 保存检查点
            self._save_checkpoint(epoch, eval_metrics)

            # 早停检查
            if self._check_early_stopping(eval_metrics):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    def _train_epoch(self) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0

        train_dataloader = self.data_module.train_dataloader()
        progress_bar = tqdm(train_dataloader, desc="Training")

        for step, batch in enumerate(progress_bar):
            # 将数据移到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 前向传播
            logits = self.model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )

            # 计算损失
            loss = nn.CrossEntropyLoss()(logits, batch["labels"])

            # 反向传播
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            # 梯度累积
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )

                # 优化器步进
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

            # 定期评估
            if self.global_step % self.config.eval_steps == 0:
                eval_metrics = self.evaluate()
                logger.info(f"Step {self.global_step}: {eval_metrics}")
                self.model.train()

        return total_loss / len(train_dataloader)

    @torch.no_grad()
    def evaluate(self) -> dict:
        """评估模型"""
        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        val_dataloader = self.data_module.val_dataloader()

        for batch in tqdm(val_dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            logits = self.model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )

            loss = nn.CrossEntropyLoss()(logits, batch["labels"])
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

        accuracy = correct / total
        avg_loss = total_loss / len(val_dataloader)

        return {
            "eval_loss": avg_loss,
            "eval_accuracy": accuracy,
        }

    def _save_checkpoint(self, epoch: int, metrics: dict):
        """保存检查点"""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型
        torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")

        # 保存优化器状态
        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

        # 保存训练状态
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "best_metric": self.best_metric,
                "metrics": metrics,
            },
            checkpoint_dir / "trainer_state.pt",
        )

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    def _check_early_stopping(self, metrics: dict) -> bool:
        """检查是否应该早停"""
        if self.config.early_stopping_patience is None:
            return False

        current_metric = metrics.get("eval_accuracy", 0)

        if current_metric > self.best_metric + self.config.early_stopping_threshold:
            self.best_metric = current_metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.config.early_stopping_patience
