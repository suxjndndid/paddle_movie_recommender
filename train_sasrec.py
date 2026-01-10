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
import paddle.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sasrec_model import SASRec, MyBCEWithLogitLoss
from data.sequence_dataset import create_sequence_dataset, SASRecDataset


def parse_args():
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
    parser.add_argument(
        "--optimizer", type=str, default="AdamW", help="优化器: Adam|AdamW|Adagrad"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["cpu", "gpu"],
        help="训练设备: cpu 或 gpu",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU设备ID (当device=gpu时有效)",
    )

    return parser.parse_args()


def train(args):
    import pandas as pd

    print("=" * 60)
    print("SASRec模型训练")
    print("=" * 60)

    if args.device == "gpu" and paddle.is_compiled_with_cuda():
        device = f"gpu:{args.gpu_id}"
        paddle.set_device(device)
        print(f"使用GPU设备: {device}")
    else:
        paddle.set_device("cpu")
        print(f"使用CPU设备")

    os.makedirs(args.save_dir, exist_ok=True)

    ratings_path = os.path.join(args.data_dir, "processed", "ratings.csv")
    dataset = create_sequence_dataset(
        ratings_path=ratings_path, max_len=args.max_len, batch_size=args.batch_size
    )

    ratings_df = pd.read_csv(ratings_path)
    max_item_id = ratings_df["movie_id"].max()

    print(f"\n数据集信息:")
    print(f"  用户数量: {dataset.num_users}")
    print(f"  物品数量: {dataset.num_items}")
    print(f"  最大物品ID: {max_item_id}")
    print(f"  序列最大长度: {args.max_len}")
    print(f"  设备: {args.device}")

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

    clip = None
    if args.optimizer == "Adam":
        optim = optimizer.Adam(
            parameters=model.parameters(), learning_rate=args.lr, grad_clip=clip
        )
    elif args.optimizer == "Adagrad":
        optim = optimizer.Adagrad(
            parameters=model.parameters(), learning_rate=args.lr, grad_clip=clip
        )
    elif args.optimizer == "AdamW":
        optim = optimizer.AdamW(
            parameters=model.parameters(), learning_rate=args.lr, grad_clip=clip
        )
    else:
        optim = optimizer.Adam(
            parameters=model.parameters(), learning_rate=args.lr, grad_clip=clip
        )

    criterion = MyBCEWithLogitLoss()

    model.train()
    total_batch = 0
    best_hit, best_ndcg = 0, 0

    print(f"\n开始训练...")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0
        num_batches = 0

        while num_batches < 100:
            users, seqs, pos, neg = dataset.next_batch()

            pos_logits, neg_logits = model(seqs, pos, neg)

            targets = (pos != 0).astype(dtype="float32")
            loss = criterion(pos_logits, neg_logits, targets)

            if args.l2_emb > 0:
                for param in model.item_emb.parameters():
                    loss += args.l2_emb * paddle.norm(param)

            loss.backward()
            optim.step()
            optim.clear_grad()

            epoch_loss += float(loss)
            num_batches += 1
            total_batch += 1

            if total_batch % args.eval_interval == 0:
                hit, ndcg = dataset.evaluate(model, k=10)
                print(
                    f"  Epoch {epoch}, Batch {total_batch}: "
                    f"Loss={float(loss):.4f}, "
                    f"Hit@10={hit:.4f}, NDCG@10={ndcg:.4f}"
                )

                if hit > best_hit:
                    best_hit, best_ndcg = hit, ndcg
                    paddle.save(
                        model.state_dict(),
                        os.path.join(args.save_dir, "sasrec_best.pdparams"),
                    )

        print(f"Epoch {epoch}: Avg Loss={epoch_loss / num_batches:.4f}")

    print("-" * 60)
    print(f"训练完成! 最佳性能: Hit@10={best_hit:.4f}, NDCG@10={best_ndcg:.4f}")

    paddle.save(
        model.state_dict(), os.path.join(args.save_dir, "sasrec_final.pdparams")
    )

    return model, dataset


def evaluate_sasrec(model_path, dataset_path, k=10):
    dataset = create_sequence_dataset(
        ratings_path=os.path.join(dataset_path, "processed", "ratings.csv"),
        max_len=50,
        batch_size=64,
    )

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

    hit, ndcg = dataset.evaluate(model, k=k)
    print(f"评估结果: Hit@{k}={hit:.4f}, NDCG@{k}={ndcg:.4f}")

    return hit, ndcg


if __name__ == "__main__":
    args = parse_args()
    train(args)
