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
import shutil
import json

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
        tokenizer,
        device: str = "cuda",
    ):
        """
        Args:
            model: 模型实例
            data_module: 数据模块实例
            config: TrainingConfig 对象
            tokenizer: 分词器实例
            device: 设备配置（'cuda', 'cpu', 'mps' 等）
        """
        self.model = model
        self.data_module = data_module
        self.config = config
        self.tokenizer = tokenizer

        # 设置设备
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        # 缓存 DataLoader，避免重复创建
        self.train_dataloader = self.data_module.train_dataloader()
        self.val_dataloader = self.data_module.val_dataloader()

        # 设置优化器
        self.optimizer = self._create_optimizer()

        # 设置学习率调度器
        self.scheduler = self._create_scheduler()

        # 训练状态
        self.global_step = 0
        self.best_metric = float("-inf")
        self.patience_counter = 0
        self.checkpoints = []  # 保存所有检查点路径，用于管理数量限制

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

        num_training_steps = (
            len(self.train_dataloader)
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

        progress_bar = tqdm(self.train_dataloader, desc="Training")

        for step, batch in enumerate(progress_bar):
            # 将数据移到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 前向传播
            logits = self.model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
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

            # 定期日志记录
            if (
                self.global_step > 0
                and self.global_step % self.config.logging_steps == 0
            ):
                logger.info(
                    f"Step {self.global_step}: loss={loss.item():.4f}, "
                    f"lr={self.scheduler.get_last_lr()[0]:.2e}"
                )

            # 定期保存
            if self.global_step > 0 and self.global_step % self.config.save_steps == 0:
                # 临时评估以保存检查点
                eval_metrics = self.evaluate()
                logger.info(f"Step {self.global_step} evaluation: {eval_metrics}")
                self._save_checkpoint(
                    epoch=f"step-{self.global_step}", metrics=eval_metrics
                )
                self.model.train()

            # 定期评估
            if self.global_step > 0 and self.global_step % self.config.eval_steps == 0:
                eval_metrics = self.evaluate()
                logger.info(f"Step {self.global_step}: {eval_metrics}")
                self.model.train()

        return total_loss / len(self.train_dataloader)

    @torch.no_grad()
    def evaluate(self) -> dict:
        """评估模型"""
        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(self.val_dataloader, desc="Evaluating"):
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
        avg_loss = total_loss / len(self.val_dataloader)

        return {
            "eval_loss": avg_loss,
            "eval_accuracy": accuracy,
        }

    def _save_checkpoint(self, epoch, metrics: dict):
        """保存检查点（包含模型、优化器、训练状态和模型配置）

        Args:
            epoch: epoch 编号或步数标识（如 "step-1000"）
            metrics: 评估指标
        """
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型权重
        torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")

        # 保存模型配置（用于重建模型）
        model_config_dict = self.model.config.to_dict()
        with open(checkpoint_dir / "model_config.json", "w", encoding="utf-8") as f:
            json.dump(model_config_dict, f, indent=2, ensure_ascii=False)

        # 保存 tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir / "tokenizer")

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

        # 记录检查点路径
        self.checkpoints.append(checkpoint_dir)

        # 检查并保存最佳模型
        current_metric = metrics.get("eval_accuracy", 0)
        if current_metric > self.best_metric:
            self._save_best_model(epoch, current_metric, metrics, model_config_dict)

        # 清理旧检查点
        self._cleanup_old_checkpoints()

    def _save_best_model(
        self, epoch, metric: float, metrics: dict, model_config_dict: dict
    ):
        """保存最佳模型

        Args:
            epoch: epoch 编号或步数标识
            metric: 当前最佳指标值
            metrics: 完整的评估指标
            model_config_dict: 模型配置字典
        """
        best_model_dir = Path(self.config.output_dir) / "best_model"
        best_model_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型权重
        torch.save(self.model.state_dict(), best_model_dir / "model.pt")

        # 保存模型配置
        with open(best_model_dir / "model_config.json", "w", encoding="utf-8") as f:
            json.dump(model_config_dict, f, indent=2, ensure_ascii=False)

        # 保存 tokenizer
        self.tokenizer.save_pretrained(best_model_dir / "tokenizer")

        # 保存训练状态
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "best_metric": metric,
                "metrics": metrics,
            },
            best_model_dir / "trainer_state.pt",
        )

        logger.info(f"Best model saved to {best_model_dir} with accuracy: {metric:.4f}")

    def _cleanup_old_checkpoints(self):
        """清理超出限制的旧检查点"""
        if self.config.save_total_limit is None:
            return

        if len(self.checkpoints) > self.config.save_total_limit:
            # 保留最近的 save_total_limit 个检查点
            old_checkpoints = self.checkpoints[: -self.config.save_total_limit]
            self.checkpoints = self.checkpoints[-self.config.save_total_limit :]

            # 删除旧检查点
            for old_checkpoint in old_checkpoints:
                if old_checkpoint.exists():
                    shutil.rmtree(old_checkpoint)
                    logger.info(f"Removed old checkpoint: {old_checkpoint}")

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
