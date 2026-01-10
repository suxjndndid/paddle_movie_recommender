#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推荐系统评估模块
包含各种推荐系统评估指标
"""

import math
import numpy as np
from collections import defaultdict
import paddle

from .enhanced_metrics import (
    EnhancedMetrics,
    compute_rmse,
    compute_mae,
    compute_accuracy,
)


class RecommenderEvaluator:
    """推荐系统评估器"""

    def __init__(self):
        pass

    @staticmethod
    def mae(predictions, ground_truth):
        """平均绝对误差 (MAE)"""
        if len(predictions) == 0:
            return 0.0
        error = sum(abs(p - t) for p, t in zip(predictions, ground_truth)) / len(
            predictions
        )
        return error

    @staticmethod
    def rmse(predictions, ground_truth):
        """均方根误差 (RMSE)"""
        if len(predictions) == 0:
            return 0.0
        mse = sum((p - t) ** 2 for p, t in zip(predictions, ground_truth)) / len(
            predictions
        )
        return math.sqrt(mse)

    @staticmethod
    def accuracy(predictions, ground_truth, threshold=3.5):
        """预测精度 (Accuracy) - 预测正确（预测值与真实值在阈值内一致）的比例"""
        if len(predictions) == 0:
            return 0.0
        correct = sum(1 for p, t in zip(predictions, ground_truth) if abs(p - t) < 0.5)
        return correct / len(predictions)

    @staticmethod
    def precision_recall_f1(predictions, ground_truth, k=10, threshold=3.5):
        """精确率、召回率、F1分数"""
        if len(predictions) == 0:
            return 0.0, 0.0, 0.0

        # 推荐的物品（预测评分>=阈值的）
        recommended = set(i for i, p in enumerate(predictions) if p >= threshold)

        # 相关的物品（真实评分>=阈值的）
        relevant = set(i for i, t in enumerate(ground_truth) if t >= threshold)

        if len(recommended) == 0 or len(relevant) == 0:
            return 0.0, 0.0, 0.0

        # 命中数
        hits = recommended & relevant

        precision = len(hits) / len(recommended) if len(recommended) > 0 else 0.0
        recall = len(hits) / len(relevant) if len(relevant) > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return precision, recall, f1

    @staticmethod
    def dcg_at_k(relevances, k):
        """计算DCG@K"""
        relevances = np.array(relevances)[:k]
        if len(relevances) == 0:
            return 0.0
        discounts = np.log2(np.arange(len(relevances)) + 2)
        return np.sum(relevances / discounts)

    @staticmethod
    def ndcg_at_k(predictions, ground_truth, k=10):
        """NDCG@K"""
        if len(predictions) == 0:
            return 0.0

        # 按预测值排序
        sorted_indices = np.argsort(predictions)[::-1][:k]
        relevances = [ground_truth[i] for i in sorted_indices]

        dcg = RecommenderEvaluator.dcg_at_k(relevances, k)

        # 按真实值排序的理想DCG
        ideal_relevances = sorted(ground_truth, reverse=True)[:k]
        idcg = RecommenderEvaluator.dcg_at_k(ideal_relevances, k)

        if idcg == 0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def hit_rate_at_k(predictions, ground_truth, k=10, threshold=3.5):
        """HR@K - 命中率"""
        if len(predictions) == 0:
            return 0.0

        top_k_indices = np.argsort(predictions)[::-1][:k]
        relevant_items = set(i for i, t in enumerate(ground_truth) if t >= threshold)

        hits = len(set(top_k_indices) & relevant_items)
        return hits / len(relevant_items) if len(relevant_items) > 0 else 0.0

    @staticmethod
    def confusion_matrix(predictions, ground_truth, num_classes=5):
        """混淆矩阵"""
        matrix = np.zeros((num_classes, num_classes), dtype=int)

        for p, t in zip(predictions, ground_truth):
            p_class = int(np.clip(round(p) - 1, 0, num_classes - 1))
            t_class = int(np.clip(round(t) - 1, 0, num_classes - 1))
            matrix[t_class, p_class] += 1

        return matrix

    @staticmethod
    def coverage(all_recommended_items, all_items):
        """覆盖率 - 推荐列表中覆盖的物品比例"""
        if len(all_items) == 0:
            return 0.0
        recommended_set = set()
        for items in all_recommended_items:
            recommended_set.update(items)
        return len(recommended_set) / len(all_items)

    @staticmethod
    def diversity(recommended_items_list, similarity_matrix=None):
        """多样性 - 推荐列表中物品之间的差异程度"""
        if len(recommended_items_list) == 0:
            return 0.0

        diversity_scores = []

        for recommended_items in recommended_items_list:
            if len(recommended_items) <= 1:
                diversity_scores.append(0.0)
                continue

            # 计算列表内物品对之间的差异
            pairs = 0
            diversity = 0.0

            for i in range(len(recommended_items)):
                for j in range(i + 1, len(recommended_items)):
                    item_i, item_j = recommended_items[i], recommended_items[j]

                    if (
                        similarity_matrix is not None
                        and item_i < len(similarity_matrix)
                        and item_j < len(similarity_matrix)
                    ):
                        # 使用预计算的相似度矩阵
                        sim = similarity_matrix[item_i][item_j]
                    else:
                        # 假设均匀分布
                        sim = 0.5

                    diversity += 1 - sim
                    pairs += 1

            if pairs > 0:
                diversity_scores.append(diversity / pairs)
            else:
                diversity_scores.append(0.0)

        return np.mean(diversity_scores)

    @staticmethod
    def popularity_bias(all_recommended_items, item_popularity):
        """流行度偏差 - 推荐物品的平均流行度"""
        if len(all_recommended_items) == 0:
            return 0.0

        total_popularity = 0
        count = 0

        for items in all_recommended_items:
            for item in items:
                if item in item_popularity:
                    total_popularity += item_popularity[item]
                    count += 1

        return total_popularity / count if count > 0 else 0.0


def evaluate_recommender(
    model,
    test_data,
    all_items,
    use_features=True,
    use_poster=False,
    movie_features_all=None,
    poster_features_all=None,
):
    """
    评估推荐模型

    Args:
        model: 训练好的模型
        test_data: 测试数据集
        all_items: 所有物品ID列表
        use_features: 是否使用特征
        use_poster: 是否使用海报特征
        movie_features_all: 所有电影特征矩阵 (n_movies, n_features) 或 None
        poster_features_all: 所有电影海报特征矩阵 (n_movies, 2048) 或 None

    Returns:
        metrics: 评估指标字典
    """
    evaluator = RecommenderEvaluator()

    predictions = []
    ground_truth = []
    recommended_items_list = []

    model.eval()

    with paddle.no_grad():
        for batch in test_data:
            user_ids = batch["user_id"]
            movie_ids = batch["movie_id"]
            ratings = batch["rating"]

            # 预测
            if use_features and use_poster:
                preds = model(
                    user_ids,
                    movie_ids,
                    batch["user_feature"],
                    batch["movie_feature"],
                    batch["poster_feature"],
                )
            elif use_features:
                preds = model(
                    user_ids, movie_ids, batch["user_feature"], batch["movie_feature"]
                )
            else:
                preds = model(user_ids, movie_ids)

            preds = preds.numpy().flatten()

            predictions.extend(preds)
            ground_truth.extend(ratings.numpy())

            # 获取推荐列表
            for i in range(len(user_ids)):
                user_idx = user_ids[i].item()
                user_feature = None
                if use_features and "user_feature" in batch:
                    user_feature = batch["user_feature"][i].numpy()
                user_recommended = get_top_k_recommendations(
                    model,
                    user_idx,
                    all_items,
                    k=10,
                    use_features=use_features and movie_features_all is not None,
                    use_poster=use_poster and poster_features_all is not None,
                    user_feature=user_feature,
                    movie_features_all=movie_features_all,
                    poster_features_all=poster_features_all,
                )
                recommended_items_list.append(user_recommended)

    # 计算各项指标
    metrics = {
        "MAE": evaluator.mae(predictions, ground_truth),
        "RMSE": evaluator.rmse(predictions, ground_truth),
        "Accuracy": evaluator.accuracy(predictions, ground_truth),
        "Precision@10": evaluator.precision_recall_f1(predictions, ground_truth, k=10)[
            0
        ],
        "Recall@10": evaluator.precision_recall_f1(predictions, ground_truth, k=10)[1],
        "F1@10": evaluator.precision_recall_f1(predictions, ground_truth, k=10)[2],
        "NDCG@10": evaluator.ndcg_at_k(predictions, ground_truth, k=10),
        "HitRate@10": evaluator.hit_rate_at_k(predictions, ground_truth, k=10),
    }

    # 混淆矩阵
    cm = evaluator.confusion_matrix(predictions, ground_truth)
    metrics["ConfusionMatrix"] = cm

    # 覆盖率
    metrics["Coverage"] = evaluator.coverage(recommended_items_list, all_items)

    return metrics


def get_top_k_recommendations(
    model,
    user_idx,
    all_movie_idxs,
    k=10,
    use_features=True,
    use_poster=False,
    user_feature=None,
    movie_features_all=None,
    poster_features_all=None,
):
    """
    获取用户的Top-K推荐列表

    Args:
        model: 训练好的模型
        user_idx: 用户ID（索引）
        all_movie_idxs: 所有电影ID（索引）
        k: 推荐数量
        use_features: 是否使用特征
        use_poster: 是否使用海报特征
        user_feature: 用户特征 (特征维度,)
        movie_features_all: 所有电影特征 (n_movies, feature_dim) 或 None
        poster_features_all: 所有电影海报特征 (n_movies, 2048) 或 None

    Returns:
        recommended_movie_idxs: 推荐的电影ID列表
    """
    model.eval()

    # 检查特征是否完整（需要所有电影的特征）
    features_available = (
        use_features
        and user_feature is not None
        and movie_features_all is not None
        and (not use_poster or poster_features_all is not None)
    )

    # 批量大小，避免一次性处理所有电影
    batch_size = 256
    all_preds = []

    with paddle.no_grad():
        # 用户特征
        if features_available:
            user_features_base = paddle.to_tensor(user_feature).unsqueeze(0)

        # 分批处理
        for i in range(0, len(all_movie_idxs), batch_size):
            batch_end = min(i + batch_size, len(all_movie_idxs))
            batch_movie_idxs = all_movie_idxs[i:batch_end]
            batch_len = len(batch_movie_idxs)

            batch_user_ids = paddle.full([batch_len], user_idx, dtype="int64")

            if features_available:
                batch_user_features = paddle.tile(user_features_base, [batch_len, 1])
                batch_movie_features = paddle.to_tensor(movie_features_all[i:batch_end])

                if use_poster and poster_features_all is not None:
                    batch_poster_features = paddle.to_tensor(
                        poster_features_all[i:batch_end]
                    )
                    preds = model(
                        batch_user_ids,
                        paddle.to_tensor(batch_movie_idxs),
                        batch_user_features,
                        batch_movie_features,
                        batch_poster_features,
                    )
                else:
                    preds = model(
                        batch_user_ids,
                        paddle.to_tensor(batch_movie_idxs),
                        batch_user_features,
                        batch_movie_features,
                    )
            else:
                preds = model(batch_user_ids, paddle.to_tensor(batch_movie_idxs))

            all_preds.extend(preds.numpy().flatten())

        # 获取top-k
        top_k_indices = np.argsort(np.array(all_preds))[::-1][:k]
        recommended_movie_idxs = [all_movie_idxs[i] for i in top_k_indices]

    return recommended_movie_idxs


def print_evaluation_results(metrics):
    """打印评估结果"""
    print("\n" + "=" * 60)
    print("推荐系统评估结果")
    print("=" * 60)

    for key, value in metrics.items():
        if key == "ConfusionMatrix":
            print(f"\n混淆矩阵:")
            print(value)
        else:
            print(f"{key}: {value:.4f}")

    print("=" * 60)


def save_evaluation_results(metrics, output_file):
    """保存评估结果"""
    import json

    # 处理numpy类型
    serializable_metrics = {}
    for key, value in metrics.items():
        if key == "ConfusionMatrix":
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, (np.floating, float)):
            serializable_metrics[key] = float(value)
        elif isinstance(value, (np.integer, int)):
            serializable_metrics[key] = int(value)
        else:
            serializable_metrics[key] = str(value)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(serializable_metrics, f, ensure_ascii=False, indent=2)

    print(f"评估结果已保存到: {output_file}")
