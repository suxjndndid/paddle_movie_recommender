#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型训练脚本
"""

import os
import sys
import argparse
import paddle
from paddle.optimizer import Adam
from paddle.nn import MSELoss
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ncf_model import NCF
from data.dataset import create_data_loaders
from evaluation.evaluator import evaluate_recommender, print_evaluation_results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="电影推荐系统训练")

    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--use_features", action="store_true", default=True)
    parser.add_argument("--use_poster", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default=None)
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


def train_epoch(
    model, train_loader, optimizer, criterion, epoch, use_features, use_poster
):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    n_samples = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        user_ids = batch["user_id"]
        movie_ids = batch["movie_id"]
        ratings = batch["rating"]

        # 预测
        if use_features and use_poster:
            predictions = model(
                user_ids,
                movie_ids,
                batch["user_feature"],
                batch["movie_feature"],
                batch["poster_feature"],
            )
        elif use_features:
            predictions = model(
                user_ids, movie_ids, batch["user_feature"], batch["movie_feature"]
            )
        else:
            predictions = model(user_ids, movie_ids)

        # 计算损失
        loss = criterion(predictions.squeeze(), ratings)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        total_loss += loss.item()
        n_samples += len(ratings)

    return total_loss / n_samples


def main():
    args = parse_args()

    print("=" * 60)
    print("电影推荐系统 - 模型训练")
    print("=" * 60)

    # 设置设备
    try:
        if (
            args.device == "gpu"
            and hasattr(paddle, "is_compiled_with_cuda")
            and paddle.is_compiled_with_cuda()
        ):
            paddle.set_device(f"gpu:{args.gpu_id}")
            print(f"使用GPU设备: {args.gpu_id}")
        else:
            paddle.set_device("cpu")
            print("使用CPU设备")
    except Exception:
        paddle.set_device("cpu")
        print("使用CPU设备")

    # 创建数据加载器
    train_loader, test_loader, train_dataset, test_dataset = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        use_features=args.use_features,
        use_poster=args.use_poster,
    )

    # 创建模型
    model = NCF(
        num_users=train_dataset.n_users,
        num_items=train_dataset.n_movies,
        use_features=args.use_features,
        use_poster=args.use_poster,
    )

    # 优化器和损失函数
    optimizer = Adam(learning_rate=args.learning_rate, parameters=model.parameters())
    criterion = MSELoss()

    # 保存目录
    if args.save_dir is None:
        args.save_dir = os.path.join(args.data_dir, "models")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n模型配置:")
    print(f"  - 用户数: {train_dataset.n_users}")
    print(f"  - 电影数: {train_dataset.n_movies}")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 训练轮数: {args.epochs}")
    print(f"  - 学习率: {args.learning_rate}")
    print(f"  - 使用特征: {args.use_features}")
    print(f"  - 使用海报: {args.use_poster}")
    print(f"  - 设备: {args.device}")

    # 训练循环
    print("\n开始训练...")
    best_metric = float("inf")

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            epoch,
            args.use_features,
            args.use_poster,
        )

        # 评估
        all_movie_idxs = list(range(1, train_dataset.n_movies))

        if args.use_features:
            movie_features_all = train_dataset.get_all_movie_features()
            poster_features_all = (
                train_dataset.get_all_poster_features() if args.use_poster else None
            )
        else:
            movie_features_all = None
            poster_features_all = None

        metrics = evaluate_recommender(
            model,
            test_loader,
            all_movie_idxs,
            use_features=args.use_features,
            use_poster=args.use_poster,
            movie_features_all=movie_features_all,
            poster_features_all=poster_features_all,
        )

        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test MAE: {metrics['MAE']:.4f}")
        print(f"  Test RMSE: {metrics['RMSE']:.4f}")
        print(f"  Test Accuracy: {metrics['Accuracy']:.4f}")

        # 保存最佳模型
        if metrics["MAE"] < best_metric:
            best_metric = metrics["MAE"]
            save_path = os.path.join(args.save_dir, "ncf_model.pdparams")
            paddle.save(model.state_dict(), save_path)
            print(f"  -> 保存最佳模型: {save_path}")

    # 最终评估
    print("\n" + "=" * 60)
    print("最终评估结果")
    print("=" * 60)
    print_evaluation_results(metrics)

    print(f"\n训练完成！最佳模型保存在: {args.save_dir}")


if __name__ == "__main__":
    main()
