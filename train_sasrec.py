#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SASRec训练脚本

使用方法:
    python train_sasrec.py --data_dir ./data --epochs 20 --batch_size 64
"""

import os
import sys
import argparse
import numpy as np
import paddle
import paddle.optimizer as optimizer
import paddle.nn as nn
import paddle.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sasrec_model import SASRec
from data.sequence_dataset import create_sequence_dataset, SASRecDataset


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="SASRec Training")

    parser.add_argument("--data_dir", type=str, default="./data", help="数据目录")
    parser.add_argument("--save_dir", type=str, default="./models", help="模型保存目录")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--max_len", type=int, default=50, help="序列最大长度")
    parser.add_argument("--hidden_units", type=int, default=64, help="嵌入维度")
    parser.add_argument("--num_heads", type=int, default=2, help="注意力头数量")
    parser.add_argument("--num_blocks", type=int, default=2, help="Transformer块数量")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout比例")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--l2_emb", type=float, default=0.0, help="L2正则化系数")
    parser.add_argument("--eval_interval", type=int, default=100, help="评估间隔(批次)")

    return parser.parse_args()


def bce_loss(pos_logits, neg_logits, labels):
    """BCE损失函数"""
    pos_loss = -paddle.log(F.sigmoid(pos_logits) + 1e-24) * labels
    neg_loss = -paddle.log(1 - F.sigmoid(neg_logits) + 1e-24) * labels
    return paddle.mean(pos_loss + neg_loss)


class BCELoss(nn.Layer):
    """BCE损失函数 (Paddle Layer)"""

    def __init__(self):
        super().__init__()

    def forward(self, pos_logits, neg_logits, labels):
        pos_loss = -paddle.log(F.sigmoid(pos_logits) + 1e-24) * labels
        neg_loss = -paddle.log(1 - F.sigmoid(neg_logits) + 1e-24) * labels
        return paddle.mean(pos_loss + neg_loss)


def train(args):
    """训练SASRec模型"""
    import pandas as pd

    print("=" * 60)
    print("SASRec模型训练")
    print("=" * 60)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 创建数据集
    ratings_path = os.path.join(args.data_dir, "processed", "ratings.csv")
    dataset = create_sequence_dataset(
        ratings_path=ratings_path, max_len=args.max_len, batch_size=args.batch_size
    )

    # 获取最大物品ID (用于嵌入层)
    ratings_df = pd.read_csv(ratings_path)
    max_item_id = ratings_df["movie_id"].max()

    print(f"\n数据集信息:")
    print(f"  用户数量: {dataset.num_users}")
    print(f"  物品数量: {dataset.num_items}")
    print(f"  最大物品ID: {max_item_id}")
    print(f"  序列最大长度: {args.max_len}")

    # 创建模型
    model = SASRec(
        item_num=max_item_id,
        max_len=args.max_len,
        hidden_units=args.hidden_units,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        dropout_rate=args.dropout_rate,
    )

    print(f"\n模型信息:")
    print(f"  嵌入维度: {args.hidden_units}")
    print(f"  注意力头: {args.num_heads}")
    print(f"  Transformer块: {args.num_blocks}")

    # 优化器
    optim = optimizer.Adam(parameters=model.parameters(), learning_rate=args.lr)

    # 损失函数
    criterion = BCELoss()

    # 训练
    model.train()
    total_batch = 0
    best_hit, best_ndcg = 0, 0

    print(f"\n开始训练...")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0
        num_batches = 0

        while num_batches < 100:  # 每个epoch固定100个批次
            # 获取批次数据
            users, seqs, pos, neg = dataset.next_batch()

            # 前向传播
            pos_logits, neg_logits = model(seqs, pos, neg)

            # 计算损失
            labels = (pos != 0).astype("float32")
            loss = criterion(pos_logits, neg_logits, labels)

            # L2正则化
            if args.l2_emb > 0:
                for param in model.item_embedding.parameters():
                    loss += args.l2_emb * paddle.linalg.norm(param)

            # 反向传播
            loss.backward()
            optim.step()
            optim.clear_grad()

            epoch_loss += float(loss)
            num_batches += 1
            total_batch += 1

            # 定期评估
            if total_batch % args.eval_interval == 0:
                hit, ndcg = dataset.evaluate(model, k=10)
                print(
                    f"  Epoch {epoch}, Batch {total_batch}: "
                    f"Loss={float(loss):.4f}, "
                    f"Hit@10={hit:.4f}, NDCG@10={ndcg:.4f}"
                )

                if hit > best_hit:
                    best_hit, best_ndcg = hit, ndcg
                    # 保存最佳模型
                    paddle.save(
                        model.state_dict(),
                        os.path.join(args.save_dir, "sasrec_best.pdparams"),
                    )

        print(f"Epoch {epoch}: Avg Loss={epoch_loss / num_batches:.4f}")

    print("-" * 60)
    print(f"训练完成! 最佳性能: Hit@10={best_hit:.4f}, NDCG@10={best_ndcg:.4f}")

    # 保存最终模型
    paddle.save(
        model.state_dict(), os.path.join(args.save_dir, "sasrec_final.pdparams")
    )

    return model, dataset


def evaluate_sasrec(model_path, dataset_path, k=10):
    """评估SASRec模型"""
    # 加载数据
    dataset = create_sequence_dataset(
        ratings_path=os.path.join(dataset_path, "processed", "ratings.csv"),
        max_len=50,
        batch_size=64,
    )

    # 加载模型
    model = SASRec(
        item_num=dataset.num_items,
        max_len=50,
        hidden_units=64,
        num_heads=2,
        num_blocks=2,
        dropout_rate=0.5,
    )
    model.set_state_dict(paddle.load(model_path))
    model.eval()

    # 评估
    hit, ndcg = dataset.evaluate(model, k=k)
    print(f"评估结果: Hit@{k}={hit:.4f}, NDCG@{k}={ndcg:.4f}")

    return hit, ndcg


if __name__ == "__main__":
    args = parse_args()
    train(args)
