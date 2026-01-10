#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SASRec模型 - 基于Transformer的序列推荐

参考: https://github.com/paddorch/SASRec.paddle
论文: "Self-Attentive Sequential Recommendation", WWW 2019
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


class SASRec(nn.Layer):
    """
    Self-Attentive Sequential Recommendation

    核心思想: 使用Transformer编码器学习用户行为序列中的时序模式
    """

    def __init__(
        self,
        item_num,
        max_len=50,
        hidden_units=64,
        num_heads=2,
        num_blocks=2,
        dropout_rate=0.5,
    ):
        """
        Args:
            item_num: 物品数量 (包括padding 0)
            max_len: 序列最大长度
            hidden_units: 嵌入维度
            num_heads: 注意力头数量
            num_blocks: Transformer块数量
            dropout_rate: dropout比例
        """
        super(SASRec, self).__init__()

        self.item_num = item_num
        self.max_len = max_len
        self.hidden_units = hidden_units

        # 物品嵌入 (包含padding 0)
        self.item_embedding = nn.Embedding(
            item_num + 1, hidden_units, weight_attr=nn.initializer.Normal(std=0.1)
        )

        # 位置嵌入
        self.position_embedding = nn.Embedding(
            max_len, hidden_units, weight_attr=nn.initializer.Normal(std=0.1)
        )

        # Dropout
        self.emb_dropout = nn.Dropout(dropout_rate)

        # Transformer编码器
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_units,
            nhead=num_heads,
            dim_feedforward=hidden_units,
            dropout=dropout_rate,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=num_blocks
        )

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_units)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.initializer.XavierNormal()(param)

    def get_position_encoding(self, seqs):
        """计算位置编码"""
        # 生成位置索引 [0, 1, 2, ..., max_len-1]
        batch_size = seqs.shape[0]
        positions = np.tile(np.array(range(self.max_len)), [batch_size, 1])

        # 转换为Tensor
        positions = paddle.to_tensor(positions, dtype="int64")

        # 计算嵌入
        seqs_embed = self.item_embedding(seqs)
        position_embed = self.position_embedding(positions)

        return seqs_embed + position_embed

    def get_subsequent_mask(self, seqs):
        """生成后续遮罩 (防止看到未来信息)"""
        # 上三角遮罩，对角线也为False
        mask = paddle.tril(paddle.ones([self.max_len, self.max_len]), diagonal=0) == 0
        return mask

    def forward(self, log_seqs, pos_seqs=None, neg_seqs=None, is_training=True):
        """
        前向传播

        Args:
            log_seqs: 用户历史序列 (batch_size, max_len)
            pos_seqs: 正样本序列 (用于训练)
            neg_seqs: 负样本序列 (用于训练)
            is_training: 是否训练模式

        Returns:
            训练模式: (pos_logits, neg_logits)
            推理模式: 序列特征
        """
        # 位置编码
        seqs_embed = self.get_position_encoding(log_seqs)
        seqs_embed = self.emb_dropout(seqs_embed)
        seqs_embed = self.layer_norm(seqs_embed)

        # Transformer编码
        subsequent_mask = self.get_subsequent_mask(log_seqs)
        encoded_seq = self.encoder(seqs_embed, subsequent_mask)
        encoded_seq = self.layer_norm(encoded_seq)

        if is_training:
            # 训练模式: 计算正负样本logits
            pos_embed = self.item_embedding(pos_seqs)
            neg_embed = self.item_embedding(neg_seqs)

            # 元素级乘积求和
            pos_logits = (encoded_seq * pos_embed).sum(axis=-1)
            neg_logits = (encoded_seq * neg_embed).sum(axis=-1)

            return pos_logits, neg_logits
        else:
            # 推理模式: 返回最后一位的特征
            final_feat = encoded_seq[:, -1, :]  # (batch_size, hidden_units)
            return final_feat

    def predict(self, log_seqs, item_indices):
        """
        预测用户对候选物品的评分 (推理模式)

        Args:
            log_seqs: 用户历史序列 (batch_size, max_len)
            item_indices: 候选物品ID列表

        Returns:
            logits: 预测分数 (batch_size, num_items)
        """
        # 获取序列特征
        final_feat = self.forward(log_seqs, is_training=False)

        # 获取候选物品嵌入
        item_embs = self.item_embedding(paddle.to_tensor(item_indices, dtype="int64"))

        # 计算相似度 (作为推荐分数)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits


def sasrec_predict_next_items(model, user_history, item_indices, device="gpu"):
    """
    使用SASRec模型预测用户下一个可能喜欢的物品

    Args:
        model: 训练好的SASRec模型
        user_history: 用户历史交互序列 (list of item IDs)
        item_indices: 候选物品ID列表
        device: 设备 ('gpu' 或 'cpu')

    Returns:
        推荐分数, 排序后的物品ID
    """
    model.eval()

    # 填充序列到固定长度
    max_len = model.max_len
    if len(user_history) > max_len:
        seq = user_history[-max_len:]
    else:
        seq = [0] * (max_len - len(user_history)) + user_history

    seq = paddle.to_tensor([seq], dtype="int64")

    # 预测
    with paddle.no_grad():
        logits = model.predict(seq, item_indices)
        logits = logits.numpy()[0]

    # 排序
    sorted_indices = np.argsort(-logits)

    return logits[sorted_indices], [item_indices[i] for i in sorted_indices]


if __name__ == "__main__":
    # 简单测试
    model = SASRec(item_num=100, max_len=50, hidden_units=64)

    # 随机输入
    batch_size = 2
    log_seqs = paddle.randint(0, 100, [batch_size, 50])
    pos_seqs = paddle.randint(0, 100, [batch_size, 50])
    neg_seqs = paddle.randint(0, 100, [batch_size, 50])

    # 前向传播
    pos_logits, neg_logits = model(log_seqs, pos_seqs, neg_seqs)

    print(f"输入形状: log_seqs={log_seqs.shape}")
    print(f"正样本logits形状: {pos_logits.shape}")
    print(f"负样本logits形状: {neg_logits.shape}")
    print("✓ SASRec模型测试通过")
