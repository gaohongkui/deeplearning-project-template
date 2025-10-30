"""评估脚本
用于加载训练好的模型并在测试集上进行评估
支持命令行参数和配置文件两种方式
"""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from transformers import AutoTokenizer

from deeplearning_project_template.utils import (
    load_checkpoint,
    load_config_from_file,
    setup_device,
)
from deeplearning_project_template.data import TextDataset
from torch.utils.data import DataLoader


def evaluate(
    checkpoint_path: str,
    test_file: str,
    batch_size: int = 32,
    max_seq_length: int = 512,
    device: str = "cuda",
    save_predictions: bool = False,
):
    """评估模型

    Args:
        checkpoint_path: 检查点路径（如 "outputs/.../best_model"）
        test_file: 测试数据文件路径
        batch_size: 批次大小
        max_seq_length: 最大序列长度
        device: 设备
        save_predictions: 是否保存预测结果
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # 加载模型和 tokenizer
    model, tokenizer, _, trainer_state = load_checkpoint(checkpoint_path, device=device)
    logger.info("Model and tokenizer loaded successfully")

    if trainer_state:
        logger.info(
            f"Checkpoint info: epoch={trainer_state.get('epoch', 'unknown')}, "
            f"best_metric={trainer_state.get('best_metric', 0):.4f}"
        )

    # 创建测试数据集
    logger.info(f"Loading test data from {test_file}")
    test_dataset = TextDataset(
        file_path=test_file,
        tokenizer=tokenizer,
        max_length=max_seq_length,
    )
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # 创建数据加载器
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 评估时使用单进程避免问题
        pin_memory=(device == "cuda"),
    )

    # 评估模型
    logger.info("Starting evaluation...")
    model.eval()
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            # 将数据移到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 前向传播
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # 计算预测
            predictions = torch.argmax(logits, dim=-1)

            # 统计准确率
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            # 保存预测结果
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # 计算指标
    accuracy = total_correct / total_samples

    # 输出结果
    logger.info(f"\n{'=' * 50}")
    logger.info("Evaluation Results:")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Test file: {test_file}")
    logger.info(f"  Total samples: {total_samples}")
    logger.info(f"  Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")
    logger.info(f"{'=' * 50}\n")

    # 可选：保存预测结果
    if save_predictions:
        output_file = Path(checkpoint_path).parent / "predictions.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("prediction\tlabel\n")
            for pred, label in zip(all_predictions, all_labels):
                f.write(f"{pred}\t{label}\n")
        logger.info(f"Predictions saved to {output_file}")

    return {
        "accuracy": accuracy,
        "total_samples": total_samples,
        "predictions": all_predictions,
        "labels": all_labels,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 配置文件参数
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (YAML). Command line args override config file.",
    )

    # 必需参数
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory (e.g., outputs/.../best_model)",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Path to test data file (JSONL format)",
    )

    # 可选参数
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save predictions to file",
    )

    args = parser.parse_args()

    # 如果提供了配置文件，完全从配置文件读取
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = load_config_from_file(args.config)

        # 从配置文件读取所有参数
        args.checkpoint = config.get("checkpoint_path")
        args.test_file = config.get("test_file")
        args.batch_size = config.get("batch_size", 32)
        args.max_seq_length = config.get("max_seq_length", 512)
        args.device = config.get("device", "cuda")
        args.save_predictions = config.get("save_predictions", False)

        logger.info("Using configuration from file (command line args ignored)")

    # 验证必需参数
    if args.checkpoint is None:
        parser.error(
            "--checkpoint is required (or provide config file with checkpoint_path)"
        )
    if args.test_file is None:
        parser.error("--test_file is required (or provide config file with test_file)")

    # 检查设备可用性
    args.device = setup_device(args.device)

    # 运行评估
    logger.info("Starting evaluation with following parameters:")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Test file: {args.test_file}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Max seq length: {args.max_seq_length}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Save predictions: {args.save_predictions}")

    results = evaluate(
        checkpoint_path=args.checkpoint,
        test_file=args.test_file,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        device=args.device,
        save_predictions=args.save_predictions,
    )

    return results


if __name__ == "__main__":
    main()
