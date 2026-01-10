#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
序列数据处理模块 - 为SASRec准备训练数据

功能:
1. 构建用户行为序列
2. 训练/验证/测试集划分
3. 批次采样器
"""

import random
import numpy as np
import pandas as pd
import paddle


def build_user_sequences(ratings_df, min_seq_length=2, max_seq_length=50):
    """
    构建用户行为序列

    Args:
        ratings_df: 评分数据 (包含user_id, movie_id, timestamp)
        min_seq_length: 用户最小交互数量
        max_seq_length: 序列最大长度

    Returns:
        user_train: 训练集用户序列 {user_id: [item1, item2, ...]}
        user_valid: 验证集 (最后一个物品作为target)
        user_test: 测试集 (最后一个物品作为target)
        all_items: 所有物品集合
    """
    print("构建用户行为序列...")

    # 按时间戳排序
    ratings_df = ratings_df.sort_values(["user_id", "timestamp"])

    # 按用户分组
    user_groups = ratings_df.groupby("user_id")

    user_train = {}
    user_valid = {}
    user_test = {}

    all_items = set()

    for user_id, group in user_groups:
        items = group["movie_id"].tolist()

        if len(items) < min_seq_length:
            continue

        all_items.update(items)

        # 划分: 倒数第二个作为验证, 最后一个作为测试
        train_items = items[:-2]
        valid_item = items[-2]
        test_item = items[-1]

        user_train[user_id] = train_items
        user_valid[user_id] = valid_item
        user_test[user_id] = test_item

    print(f"  用户数量: {len(user_train)}")
    print(f"  物品数量: {len(all_items)}")
    print(f"  平均序列长度: {np.mean([len(v) for v in user_train.values()]):.1f}")

    return user_train, user_valid, user_test, all_items


def random_neq(low, high, excluded_set):
    """随机生成一个不在excluded_set中的整数"""
    while True:
        t = np.random.randint(low, high)
        if t not in excluded_set:
            return t


def sample_sequence(user_train, user_num, item_num, max_len, batch_size):
    """
    采样一个批次的数据

    Returns:
        users: 用户ID列表
        seqs: 正样本序列
        pos: 正样本 (下一个物品)
        neg: 负样本
    """
    users = []
    seqs = []
    pos = []
    neg = []

    for _ in range(batch_size):
        # 随机选择一个用户
        user = np.random.randint(1, user_num + 1)

        # 确保用户有足够的历史记录
        while user not in user_train or len(user_train[user]) < 1:
            user = np.random.randint(1, user_num + 1)

        # 构建序列
        seq = np.zeros([max_len], dtype=np.int32)
        pos_seq = np.zeros([max_len], dtype=np.int32)
        neg_seq = np.zeros([max_len], dtype=np.int32)

        user_items = set(user_train[user])
        next_item = user_items.pop()  # 最后一个物品

        idx = max_len - 1
        for item in reversed(user_train[user][:-1]):
            seq[idx] = item
            pos_seq[idx] = next_item
            # 负样本: 不在用户历史中的物品
            neg_seq[idx] = random_neq(1, item_num + 1, user_items)

            user_items.add(item)
            next_item = item
            idx -= 1
            if idx < 0:
                break

        users.append(user)
        seqs.append(seq)
        pos.append(pos_seq)
        neg.append(neg_seq)

    return (np.array(users), np.array(seqs), np.array(pos), np.array(neg))


class SASRecDataset:
    """
    SASRec数据集类
    """

    def __init__(
        self, user_train, user_valid, user_test, all_items, max_len=50, batch_size=64
    ):
        """
        Args:
            user_train: 训练集用户序列
            user_valid: 验证集
            user_test: 测试集
            all_items: 所有物品集合
            max_len: 序列最大长度
            batch_size: 批次大小
        """
        self.user_train = user_train
        self.user_valid = user_valid
        self.user_test = user_test
        self.all_items = list(all_items)
        self.num_items = len(all_items)
        self.num_users = len(user_train)
        self.max_len = max_len
        self.batch_size = batch_size

        # 计算训练批次数 (使用采样)
        self.num_batches = max(100, self.num_users // self.batch_size)

    def __len__(self):
        """返回训练批次数"""
        return self.num_batches

    def __iter__(self):
        """迭代器，支持 DataLoader 使用"""
        for _ in range(self.num_batches):
            users, seqs, pos, neg = sample_sequence(
                self.user_train,
                self.num_users,
                self.num_items,
                self.max_len,
                self.batch_size,
            )
            yield (
                paddle.to_tensor(users, dtype="int64"),
                paddle.to_tensor(seqs, dtype="int64"),
                paddle.to_tensor(pos, dtype="int64"),
                paddle.to_tensor(neg, dtype="int64"),
            )

    def next_batch(self):
        """获取下一个批次"""
        users, seqs, pos, neg = sample_sequence(
            self.user_train,
            self.num_users,
            self.num_items,
            self.max_len,
            self.batch_size,
        )

        return (
            paddle.to_tensor(users, dtype="int64"),
            paddle.to_tensor(seqs, dtype="int64"),
            paddle.to_tensor(pos, dtype="int64"),
            paddle.to_tensor(neg, dtype="int64"),
        )

    def evaluate(self, model, k=10):
        """
        在验证集/测试集上评估模型

        Args:
            model: 训练好的SASRec模型
            k: Top-K

        Returns:
            Hit@K, NDCG@K
        """
        model.eval()

        hits = []
        ndcgs = []

        for user_id, pos_item in self.user_test.items():
            # 构建用户序列
            if user_id not in self.user_train:
                continue

            seq = self.user_train[user_id]
            if len(seq) > self.max_len:
                seq = seq[-self.max_len :]
            else:
                seq = [0] * (self.max_len - len(seq)) + seq

            seq = paddle.to_tensor([seq], dtype="int64")

            # 预测所有物品的分数
            item_indices = list(range(1, self.num_items + 1))
            with paddle.no_grad():
                logits = model.predict(seq, item_indices)
                logits = logits.numpy()[0]

            # 获取Top-K
            top_k_items = np.argsort(-logits)[:k]

            # 计算Hit@K和NDCG@K
            if pos_item in top_k_items:
                hit_rank = np.where(top_k_items == pos_item)[0][0] + 1
                hits.append(1)
                ndcgs.append(1.0 / np.log2(hit_rank + 1))
            else:
                hits.append(0)
                ndcgs.append(0)

        return np.mean(hits), np.mean(ndcgs)


def create_sequence_dataset(ratings_path, max_len=50, batch_size=64, train_ratio=0.8):
    """
    创建序列数据集

    Args:
        ratings_path: 评分数据路径
        max_len: 序列最大长度
        batch_size: 批次大小
        train_ratio: 训练集比例 (用于划分训练/验证)

    Returns:
        dataset: SASRecDataset对象
    """
    # 加载评分数据
    ratings_df = pd.read_csv(ratings_path)

    # 构建序列
    user_train, user_valid, user_test, all_items = build_user_sequences(
        ratings_df, min_seq_length=2
    )

    # 创建数据集
    dataset = SASRecDataset(
        user_train=user_train,
        user_valid=user_valid,
        user_test=user_test,
        all_items=all_items,
        max_len=max_len,
        batch_size=batch_size,
    )

    return dataset


if __name__ == "__main__":
    # 简单测试
    print("测试序列数据处理...")

    # 模拟数据
    user_train = {
        1: [1, 2, 3, 4, 5],
        2: [2, 3, 4, 5, 6],
        3: [1, 3, 5, 7, 9],
    }
    user_valid = {1: 6, 2: 7, 3: 11}
    user_test = {1: 7, 2: 8, 3: 12}
    all_items = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

    dataset = SASRecDataset(
        user_train=user_train,
        user_valid=user_valid,
        user_test=user_test,
        all_items=all_items,
        max_len=5,
        batch_size=2,
    )

    print(f"用户数量: {dataset.num_users}")
    print(f"物品数量: {dataset.num_items}")

    # 测试采样
    users, seqs, pos, neg = dataset.next_batch()
    print(f"批次形状: users={users.shape}, seqs={seqs.shape}")

    print("✓ 序列数据处理测试通过")
