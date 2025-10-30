"""推理脚本
用于加载训练好的模型并对新文本进行预测
"""

import argparse
import torch
from loguru import logger
from transformers import AutoTokenizer
from typing import List, Union

from deeplearning_project_template.utils import (
    load_checkpoint,
    load_config_from_file,
    setup_device,
)


def predict(
    checkpoint_path: str,
    texts: Union[str, List[str]],
    max_seq_length: int = 512,
    device: str = "cuda",
) -> List[dict]:
    """对文本进行预测

    Args:
        checkpoint_path: 检查点路径
        texts: 单个文本或文本列表
        max_seq_length: 最大序列长度
        device: 设备

    Returns:
        预测结果列表，每个元素包含 text, label, confidence
    """
    # 确保 texts 是列表
    if isinstance(texts, str):
        texts = [texts]

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # 加载模型和 tokenizer
    model, tokenizer, _, trainer_state = load_checkpoint(checkpoint_path, device=device)
    model.eval()

    logger.info("Model and tokenizer loaded successfully")
    if trainer_state:
        logger.info(
            f"Checkpoint info: epoch={trainer_state.get('epoch', 'unknown')}, "
            f"best_metric={trainer_state.get('best_metric', 0):.4f}"
        )

    # 批量预测
    results = []
    logger.info(f"Predicting {len(texts)} samples...")

    with torch.no_grad():
        for text in texts:
            # 分词
            encoding = tokenizer(
                text,
                max_length=max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # 移到设备
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            # 前向传播
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # 计算预测和置信度
            probs = torch.softmax(logits, dim=-1)
            predicted_label = torch.argmax(logits, dim=-1).item()
            confidence = probs[0, predicted_label].item()

            results.append(
                {
                    "text": text,
                    "label": predicted_label,
                    "confidence": confidence,
                    "probabilities": probs[0].cpu().numpy().tolist(),
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Predict with trained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 配置文件参数
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (YAML). If provided, all params read from config.",
    )

    # 必需参数
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory (e.g., outputs/.../best_model)",
    )

    # 输入参数
    parser.add_argument(
        "--text",
        type=str,
        help="Single text to predict",
    )
    parser.add_argument(
        "--text_file",
        type=str,
        help="File containing texts (one per line)",
    )

    # 可选参数
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
        "--output_file",
        type=str,
        default=None,
        help="Optional: save predictions to file",
    )

    args = parser.parse_args()

    # 如果提供了配置文件，完全从配置文件读取
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = load_config_from_file(args.config)

        # 从配置文件读取所有参数
        args.checkpoint = config.get("checkpoint_path")
        args.text = config.get("text")
        args.text_file = config.get("text_file")
        args.max_seq_length = config.get("max_seq_length", 512)
        args.device = config.get("device", "cuda")
        args.output_file = config.get("output_file")

        logger.info("Using configuration from file (command line args ignored)")

    # 验证必需参数
    if args.checkpoint is None:
        parser.error(
            "--checkpoint is required (or provide config file with checkpoint_path)"
        )

    # 检查输入
    if not args.text and not args.text_file:
        parser.error(
            "Either --text or --text_file must be provided (or in config file)"
        )

    # 检查设备可用性
    args.device = setup_device(args.device)

    # 准备文本
    texts = []
    if args.text:
        texts = [args.text]
    elif args.text_file:
        logger.info(f"Reading texts from {args.text_file}")
        with open(args.text_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(texts)} texts")

    # 运行预测
    logger.info("Starting prediction with following parameters:")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Max seq length: {args.max_seq_length}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Number of texts: {len(texts)}")

    results = predict(
        checkpoint_path=args.checkpoint,
        texts=texts,
        max_seq_length=args.max_seq_length,
        device=args.device,
    )

    # 打印结果
    logger.info(f"\n{'=' * 60}")
    logger.info("Prediction Results:")
    logger.info(f"{'=' * 60}")
    for i, result in enumerate(results, 1):
        text_preview = (
            result["text"][:80] + "..." if len(result["text"]) > 80 else result["text"]
        )
        logger.info(f"\n[{i}] Text: {text_preview}")
        logger.info(f"    Label: {result['label']}")
        logger.info(f"    Confidence: {result['confidence']:.4f}")
        logger.info(
            f"    Probabilities: {[f'{p:.4f}' for p in result['probabilities']]}"
        )
    logger.info(f"\n{'=' * 60}\n")

    # 可选：保存到文件
    if args.output_file:
        import json

        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Predictions saved to {args.output_file}")

    return results


if __name__ == "__main__":
    main()
