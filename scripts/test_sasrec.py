#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SASRec快速测试脚本

使用方法:
    python scripts/test_sasrec.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sasrec_model import SASRec
from data.sequence_dataset import create_sequence_dataset, SASRecDataset


def test_sasrec():
    """测试SASRec模型"""
    print("=" * 60)
    print("SASRec模型测试")
    print("=" * 60)

    # 创建数据集
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ratings_path = os.path.join(data_dir, "data", "processed", "ratings.csv")

    print("\n[1/3] 创建数据集...")
    try:
        dataset = create_sequence_dataset(
            ratings_path=ratings_path, max_len=50, batch_size=4
        )
        print(f"  ✓ 用户数量: {dataset.num_users}")
        print(f"  ✓ 物品数量: {dataset.num_items}")
    except Exception as e:
        print(f"  ✗ 数据集创建失败: {e}")
        print("  提示: 请先运行 python main.py 处理数据")
        return False

    # 创建模型
    print("\n[2/3] 创建SASRec模型...")
    try:
        model = SASRec(
            item_num=dataset.num_items,
            max_len=50,
            hidden_units=64,
            num_heads=2,
            num_blocks=2,
            dropout_rate=0.5,
        )
        print(f"  ✓ 模型创建成功")
        print(f"  ✓ 参数数量: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"  ✗ 模型创建失败: {e}")
        return False

    # 测试前向传播
    print("\n[3/3] 测试前向传播...")
    try:
        # 获取批次数据
        users, seqs, pos, neg = dataset.next_batch()
        print(f"  ✓ 数据批次形状: users={users.shape}, seqs={seqs.shape}")

        # 前向传播
        model.eval()
        pos_logits, neg_logits = model(seqs, pos, neg)
        print(f"  ✓ 前向传播成功")
        print(f"    pos_logits形状: {pos_logits.shape}")
        print(f"    neg_logits形状: {neg_logits.shape}")

        # 测试推理模式
        final_feat = model(seqs, is_training=False)
        print(f"  ✓ 推理模式成功, 输出形状: {final_feat.shape}")

    except Exception as e:
        print(f"  ✗ 前向传播失败: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✓ SASRec测试全部通过!")
    print("=" * 60)

    return True


def test_sequence_recommendation():
    """测试序列推荐功能"""
    print("\n" + "=" * 60)
    print("序列推荐功能测试")
    print("=" * 60)

    # 模拟用户历史
    user_history = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    candidate_items = list(range(1, 100))

    # 创建模型
    model = SASRec(item_num=100, max_len=50, hidden_units=64)
    model.eval()

    # 预测
    import paddle

    seq = paddle.to_tensor([user_history], dtype="int64")
    item_indices = list(range(1, 100))

    with paddle.no_grad():
        logits = model.predict(seq, item_indices)
        logits = logits.numpy()[0]

    # 排序
    top_k = 5
    top_k_items = [item_indices[i] for i in np.argsort(-logits)[:top_k]]

    print(f"\n用户历史: {user_history}")
    print(f"Top-{top_k}推荐: {top_k_items}")

    print("\n✓ 序列推荐测试通过")
    return True


if __name__ == "__main__":
    success = test_sasrec()
    if success:
        success = test_sequence_recommendation()

    if success:
        print("\n🎉 所有测试通过!")
    else:
        print("\n❌ 测试失败")
        sys.exit(1)
