# Deep Learning Project Template

一个基于PyTorch和Hydra的深度学习项目模板，专注于文本分类任务。本项目提供了完整的项目结构、配置管理和训练流程，便于快速开始深度学习项目开发。

## 项目特性

- 🚀 **模块化设计**: 清晰的模块分离（模型、数据、训练器、配置）
- ⚙️ **配置管理**: 使用Hydra进行灵活的配置管理
- 📊 **实验追踪**: 集成Weights & Biases进行实验追踪
- 🔧 **类型安全**: 使用dataclass进行类型安全的配置管理
- 📈 **训练监控**: 完整的训练日志和评估指标
- 🧪 **可扩展性**: 易于扩展新的模型、数据集和训练策略

## 项目结构

```
deeplearning-project-template/
├── configs/                    # 配置文件目录
│   ├── config.yaml            # 主配置文件
│   ├── data/                  # 数据配置
│   │   └── glue.yaml          # GLUE数据集配置
│   ├── model/                 # 模型配置
│   │   └── bert.yaml          # BERT模型配置
│   └── training/              # 训练配置
│       └── base.yaml          # 基础训练配置
├── data/                      # 数据目录
│   └── glue/                  # GLUE数据集
│       ├── train.jsonl        # 训练数据
│       └── val.jsonl          # 验证数据
├── src/                       # 源代码目录
│   └── deeplearning_project_template/
│       ├── config.py          # 配置类定义
│       ├── data.py            # 数据处理模块
│       ├── model.py           # 模型定义
│       ├── trainer.py         # 训练器
│       └── utils.py           # 工具函数（包含模型加载）
├── train.py                   # 训练入口脚本
├── eval.py                    # 评估脚本
├── predict.py                 # 推理脚本
├── pyproject.toml             # 项目依赖配置
└── uv.lock                    # 依赖锁定文件
```

## 快速开始

### 环境配置

1. **安装依赖**:
   ```bash
   # 使用uv包管理器（推荐）
   uv sync
   
   # 或者使用pip
   pip install -e .
   ```

2. **安装开发依赖**:
   ```bash
   uv sync --group dev
   ```

### 数据准备

项目使用GLUE格式的数据集。数据文件应为JSONL格式，每行包含：

```json
{"text": "Your text here", "label": 0}
```

将训练数据放在 `data/glue/train.jsonl`，验证数据放在 `data/glue/val.jsonl`。

### 训练模型

运行训练脚本：

```bash
python train.py
```

使用自定义配置：

```bash
# 使用不同的模型配置
python train.py model=bert

# 使用不同的训练配置
python train.py training=base

# 覆盖特定参数
python train.py training.learning_rate=1e-5 training.num_train_epochs=5

# 多配置组合
python train.py model=bert data=glue training=base
```

### 评估模型

使用训练好的模型进行评估（支持命令行参数和配置文件）：

**方式1：使用命令行参数**
```bash
# 评估最佳模型
python eval.py \
  --checkpoint outputs/2025-10-31/01-10-48/outputs/best_model \
  --test_file data/glue/val.jsonl

# 评估特定检查点，自定义参数
python eval.py \
  --checkpoint outputs/2025-10-31/01-10-48/outputs/checkpoint-2 \
  --test_file data/glue/test.jsonl \
  --batch_size 16 \
  --max_seq_length 512 \
  --device cpu

# 保存预测结果
python eval.py \
  --checkpoint outputs/.../best_model \
  --test_file data/glue/test.jsonl \
  --save_predictions
```

**方式2：使用配置文件**
```bash
# 使用配置文件（需先编辑 configs/eval.yaml）
python eval.py --config configs/eval.yaml

# 注意：使用配置文件时，所有参数从配置文件读取，命令行参数将被忽略
```

参数说明：
- `--checkpoint`: 检查点目录路径（必需，tokenizer 会自动从检查点加载）
- `--test_file`: 测试数据文件路径（必需）
- `--config`: 配置文件路径（可选，YAML格式）
- `--batch_size`: 批次大小（默认：32）
- `--max_seq_length`: 最大序列长度（默认：512）
- `--device`: 设备类型（默认：cuda）
- `--save_predictions`: 是否保存预测结果到文件

### 模型推理

对新文本进行预测：

**方式1：使用命令行参数**
```bash
# 预测单个文本
python predict.py \
  --checkpoint outputs/2025-10-31/01-10-48/outputs/best_model \
  --text "This is a great movie!"

# 批量预测（从文件读取，每行一个文本）
python predict.py \
  --checkpoint outputs/2025-10-31/01-10-48/outputs/best_model \
  --text_file texts.txt \
  --output_file predictions.json

# 自定义参数
python predict.py \
  --checkpoint outputs/.../best_model \
  --text "Your text here" \
  --tokenizer_name bert-base-uncased \
  --device cpu
```

**方式2：使用配置文件**
```bash
# 使用配置文件
python predict.py --config configs/predict.yaml
```

参数说明：
- `--checkpoint`: 检查点目录路径（必需，tokenizer 会自动从检查点加载）
- `--text`: 单个文本（与 --text_file 二选一）
- `--text_file`: 批量文本文件路径（每行一个文本）
- `--config`: 配置文件路径（可选）
- `--max_seq_length`: 最大序列长度（默认：512）
- `--device`: 设备类型（默认：cuda）
- `--output_file`: 保存预测结果的文件路径（可选，JSON格式）

### 配置说明

项目使用Hydra进行配置管理，主要配置类别：

- **模型配置** (`configs/model/`): 定义模型结构参数
- **数据配置** (`configs/data/`): 定义数据处理参数
- **训练配置** (`configs/training/`): 定义训练过程参数

## 核心模块

### 1. 配置管理 (`src/deeplearning_project_template/config.py`)

使用dataclass定义类型安全的配置类：

- `ModelConfig`: 模型结构配置
- `DataConfig`: 数据处理配置
- `TrainingConfig`: 训练过程配置
- `Config`: 顶层实验配置

### 2. 模型定义 (`src/deeplearning_project_template/model.py`)

实现Transformer模型架构：

- `TransformerModel`: 主模型类
- `TransformerLayer`: Transformer层
- `MultiHeadAttention`: 多头注意力机制

### 3. 数据处理 (`src/deeplearning_project_template/data.py`)

数据加载和预处理：

- `TextDataset`: 文本分类数据集
- `DataModule`: 数据模块，封装数据加载逻辑

### 4. 训练器 (`src/deeplearning_project_template/trainer.py`)

训练流程管理：

- 模型训练和验证
- 指标计算和日志记录
- 模型保存和早停
- 检查点管理和最佳模型保存

### 5. 工具函数 (`src/deeplearning_project_template/utils.py`)

辅助工具：

- `load_checkpoint()`: 从检查点加载模型、优化器状态和训练状态

### 6. 评估脚本 (`eval.py`)

独立评估脚本：

- 加载训练好的模型
- 在测试集上评估性能
- 支持命令行参数和配置文件

### 7. 推理脚本 (`predict.py`)

独立推理脚本：

- 对新文本进行预测
- 支持单个文本或批量文本
- 返回预测标签和置信度
- 支持命令行参数和配置文件

## 使用示例

### 在 Python 代码中使用

在 Python 代码中加载和使用模型：

```python
from deeplearning_project_template.utils import load_checkpoint
import torch

# 加载模型和 tokenizer（tokenizer 会从检查点自动加载）
model, tokenizer, _, state = load_checkpoint("outputs/.../best_model", device="cuda")

# 预测
text = "This is a test sentence."
encoding = tokenizer(text, max_length=512, padding="max_length", 
                     truncation=True, return_tensors="pt")
input_ids = encoding["input_ids"].to("cuda")
attention_mask = encoding["attention_mask"].to("cuda")

model.eval()
with torch.no_grad():
    logits = model(input_ids, attention_mask)
    prediction = torch.argmax(logits, dim=-1).item()
    
print(f"Predicted label: {prediction}")
print(f"Best metric from training: {state['best_metric']:.4f}")
```

### 配置说明

项目使用Hydra进行配置管理，主要配置类别：

- **模型配置** (`configs/model/`): 定义模型结构参数
- **数据配置** (`configs/data/`): 定义数据处理参数
- **训练配置** (`configs/training/`): 定义训练过程参数

## 核心模块

### 1. 配置管理 (`src/deeplearning_project_template/config.py`)

使用dataclass定义类型安全的配置类：

- `ModelConfig`: 模型结构配置
- `DataConfig`: 数据处理配置
- `TrainingConfig`: 训练过程配置
- `Config`: 顶层实验配置

### 2. 模型定义 (`src/deeplearning_project_template/model.py`)

实现Transformer模型架构：

- `TransformerModel`: 主模型类
- `TransformerLayer`: Transformer层
- `MultiHeadAttention`: 多头注意力机制

### 3. 数据处理 (`src/deeplearning_project_template/data.py`)

数据加载和预处理：

- `TextDataset`: 文本分类数据集
- `DataModule`: 数据模块，封装数据加载逻辑

### 4. 训练器 (`src/deeplearning_project_template/trainer.py`)

训练流程管理：

- 模型训练和验证
- 指标计算和日志记录
- 模型保存和早停
- 检查点管理和最佳模型保存

### 5. 工具函数 (`src/deeplearning_project_template/utils.py`)

辅助工具：

- `load_checkpoint()`: 从检查点加载模型、优化器状态和训练状态

### 6. 评估脚本 (`eval.py`)

独立评估脚本：

- 加载训练好的模型
- 在测试集上评估性能
- 支持命令行参数配置

## 配置示例

### 基础配置 (`configs/config.yaml`)

```yaml
defaults:
  - model: bert
  - data: glue
  - training: base
  - _self_

experiment_name: text_classification
description: "Text classification experiment"
tags:
  - nlp
  - classification

device: cuda
distributed: false
```

### BERT模型配置 (`configs/model/bert.yaml`)

```yaml
model_type: bert
tokenizer_name: bert-base-uncased  # 与模型配套的 tokenizer

# 模型结构参数
hidden_size: 768
num_hidden_layers: 12
num_attention_heads: 12
intermediate_size: 3072

# Dropout 参数
hidden_dropout_prob: 0.1
attention_probs_dropout_prob: 0.1

# 其他参数
max_position_embeddings: 512
vocab_size: 30522  # 必须与 tokenizer 的 vocab 大小匹配
```

### 训练配置 (`configs/training/base.yaml`)

```yaml
output_dir: ${hydra:run.dir}/outputs
num_train_epochs: 3
learning_rate: 5e-5
weight_decay: 0.01
warmup_ratio: 0.1

optimizer_type: adamw
scheduler_type: linear

gradient_accumulation_steps: 1
max_grad_norm: 1.0

eval_steps: 500
save_steps: 500
save_total_limit: 3
logging_steps: 100

early_stopping_patience: 3
early_stopping_threshold: 0.001

seed: 42
```

## 扩展指南

### 添加新模型

1. 在 `configs/model/` 中创建新的配置文件

例如，添加 RoBERTa 模型：

```yaml
# configs/model/roberta.yaml
model_type: roberta
tokenizer_name: roberta-base  # 模型配套的 tokenizer

hidden_size: 768
num_hidden_layers: 12
num_attention_heads: 12
intermediate_size: 3072
vocab_size: 50265  # 必须与 tokenizer 的 vocab 大小匹配
```

2. 使用新模型训练：

```bash
python train.py model=roberta
```

**注意**: tokenizer 与模型强绑定，切换模型配置会自动切换对应的 tokenizer

### 添加新数据集

1. 在 `src/deeplearning_project_template/data.py` 中添加新的Dataset类
2. 在 `configs/data/` 中创建对应的配置文件
3. 更新 `DataConfig` 类以支持新数据集的参数

### 自定义训练策略

1. 修改 `src/deeplearning_project_template/trainer.py` 中的训练逻辑
2. 在 `configs/training/` 中创建新的训练配置文件
3. 更新 `TrainingConfig` 类以支持新的训练参数

## 依赖项

主要依赖：

- `torch>=2.0.0`: PyTorch深度学习框架
- `transformers>=4.35.0`: Hugging Face Transformers库
- `hydra-core>=1.3.0`: 配置管理框架
- `omegaconf>=2.3.0`: 配置对象管理
- `wandb>=0.16.0`: 实验追踪
- `scikit-learn>=1.3.0`: 机器学习工具

开发依赖：

- `notebook>=7.4.7`: Jupyter笔记本
- `ruff>=0.14.2`: 代码格式化

## 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 更新日志

### v0.1.0 (2025-10-29)
- 初始版本发布
- 支持BERT模型和文本分类任务
- 集成Hydra配置管理
- 添加Weights & Biases实验追踪