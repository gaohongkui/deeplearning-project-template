"""模型实现
关键点：构造函数接收 ModelConfig 对象，而不是 Dict
"""

import torch
import torch.nn as nn
from typing import Optional
from deeplearning_project_template.config import ModelConfig


class TransformerModel(nn.Module):
    """Transformer 模型

    注意：构造函数接收 ModelConfig 对象
    """

    def __init__(self, config: ModelConfig):
        """
        Args:
            config: ModelConfig 对象，包含所有模型参数
        """
        super().__init__()

        # 保存配置（方便后续使用）
        self.config = config

        # 使用配置初始化模型组件
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        self.encoder_layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播

        Args:
            input_ids: 输入 token IDs，形状 [batch_size, seq_length]
            attention_mask: 注意力掩码，形状 [batch_size, seq_length]

        Returns:
            logits: 分类 logits，形状 [batch_size, num_classes]
        """
        # 实现前向传播
        hidden_states = self.embeddings(input_ids)
        hidden_states = self.dropout(hidden_states)

        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask)

        # 取 [CLS] token 的输出
        pooled_output = hidden_states[:, 0]
        logits = self.classifier(pooled_output)

        return logits


class TransformerLayer(nn.Module):
    """Transformer 层

    子模块也接收同样的 config 对象
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.attention = MultiHeadAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            dropout_prob=config.attention_probs_dropout_prob,
        )

        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)

        self.output = nn.Linear(config.intermediate_size, config.hidden_size)

        self.layernorm1 = nn.LayerNorm(config.hidden_size)
        self.layernorm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播

        Args:
            hidden_states: 隐藏状态，形状 [batch_size, seq_length, hidden_size]
            attention_mask: 注意力掩码，形状 [batch_size, seq_length]

        Returns:
            layer_output: 输出隐藏状态，形状 [batch_size, seq_length, hidden_size]
        """
        # Self-attention
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.layernorm1(hidden_states + attention_output)

        # Feed-forward
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = torch.relu(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.layernorm2(hidden_states + layer_output)

        return layer_output


class MultiHeadAttention(nn.Module):
    """多头注意力

    这里展示另一种方式：接收具体参数而不是 config 对象
    适用于：功能单一、参数少的模块
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout_prob: float,
    ):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """多头注意力前向传播

        Args:
            hidden_states: 隐藏状态，形状 [batch_size, seq_length, hidden_size]
            attention_mask: 注意力掩码，形状 [batch_size, seq_length]

        Returns:
            context_layer: 上下文向量，形状 [batch_size, seq_length, hidden_size]
        """
        # 实现多头注意力
        batch_size, seq_length, hidden_size = hidden_states.size()

        # 计算 Q, K, V
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        # 重塑为多头
        query_layer = query_layer.view(
            batch_size, seq_length, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)

        key_layer = key_layer.view(
            batch_size, seq_length, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)

        value_layer = value_layer.view(
            batch_size, seq_length, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)

        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size**0.5)

        if attention_mask is not None:
            # 重塑注意力掩码以匹配注意力分数的尺寸
            # attention_mask: [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # 将 padding 位置（mask=0）填充为极小值，使得 softmax 后接近 0
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # 应用注意力
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, seq_length, -1)

        return context_layer
