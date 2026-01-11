#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推荐系统主程序
包含三种推荐路径：热门推荐、新品推荐、个性化推荐
支持相似用户和相似电影推荐
"""

import os
import sys
import random
import pickle
import numpy as np
import pandas as pd
import paddle
import logging

logging.getLogger("paddle").setLevel(logging.WARNING)
logging.getLogger("paddle").propagate = False

from tqdm import tqdm

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.ncf_model import NCF, UserSimilarityModel, MovieSimilarityModel
from models.sasrec_model import SASRec
from data.dataset import (
    MovieLensDataset,
    get_popular_movies,
    get_new_movies,
    get_all_movies,
)


class MovieRecommender:
    """电影推荐系统"""

    def __init__(
        self,
        data_dir,
        model_path=None,
        sasrec_model_path=None,
        use_features=True,
        use_poster=False,
    ):
        """
        初始化推荐系统

        Args:
            data_dir: 数据目录
            model_path: NCF模型文件路径
            sasrec_model_path: SASRec模型文件路径
            use_features: 是否使用特征
            use_poster: 是否使用海报特征
        """
        self.data_dir = data_dir
        self.use_features = use_features
        self.use_poster = use_poster

        # 加载数据集信息
        self._load_dataset_info()

        # 加载NCF模型
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("警告: 未加载NCF预训练模型")

        # 加载SASRec模型
        self.sasrec_model = None
        self.sasrec_max_len = 50
        if sasrec_model_path and os.path.exists(sasrec_model_path):
            self.load_sasrec_model(sasrec_model_path)
        else:
            print("警告: 未加载SASRec预训练模型")

        # 构建用户交互序列（用于SASRec）
        self.user_sequences = {}
        self._build_user_sequences()

        # 加载相似度矩阵
        self.user_similarity_matrix = None
        self.movie_similarity_matrix = None
        self._load_similarity_matrices()

    def _load_dataset_info(self):
        """加载数据集信息"""
        print("加载数据集信息...")

        # 加载用户和电影信息
        self.users = pd.read_csv(os.path.join(self.data_dir, "processed", "users.csv"))
        self.movies = pd.read_csv(
            os.path.join(self.data_dir, "processed", "movies.csv")
        )

        # 构建ID映射
        self.user_id_map = {
            uid: idx + 1
            for idx, uid in enumerate(sorted(self.users["user_id"].unique()))
        }
        self.user_id_reverse = {
            idx + 1: uid
            for idx, uid in enumerate(sorted(self.users["user_id"].unique()))
        }

        self.movie_id_map = {
            mid: idx + 1
            for idx, mid in enumerate(sorted(self.movies["movie_id"].unique()))
        }
        self.movie_id_reverse = {
            idx + 1: mid
            for idx, mid in enumerate(sorted(self.movies["movie_id"].unique()))
        }

        # 统计信息
        self.n_users = len(self.user_id_map) + 1
        self.n_movies = len(self.movie_id_map) + 1

        # 加载用户特征
        self.user_features = {}
        for _, row in self.users.iterrows():
            uid = row["user_id"]
            self.user_features[uid] = {
                "gender": row["gender_encoded"],
                "age": row["age"],
                "occupation": row["occupation"],
                "zipcode_prefix": int(row["zipcode"][:3])
                if pd.notna(row["zipcode"])
                else 0,
            }

        # 加载电影特征
        self.movie_features = {}
        genre_cols = [
            c
            for c in self.movies.columns
            if c.startswith("genre_") and c != "genre_list"
        ]
        self.n_genres = len(genre_cols)

        for _, row in self.movies.iterrows():
            mid = row["movie_id"]
            self.movie_features[mid] = {
                "release_year": row["release_year"]
                if pd.notna(row["release_year"])
                else 1990,
                "genres": [row[c] for c in genre_cols],
                "title": row["title"],
            }

        # 加载海报特征
        poster_features_file = os.path.join(
            self.data_dir, "processed", "poster_features.pkl"
        )
        if self.use_poster and os.path.exists(poster_features_file):
            with open(poster_features_file, "rb") as f:
                self.poster_features = pickle.load(f)
        else:
            self.poster_features = {}

        # 加载评分数据
        self.ratings = pd.read_csv(
            os.path.join(self.data_dir, "processed", "ratings.csv")
        )

        # 计算电影流行度（用于热门推荐）
        self.movie_popularity = self.ratings.groupby("movie_id").size().to_dict()

        # 准备候选电影列表
        self.popular_movies = get_popular_movies(self.data_dir, top_n=100)
        self.new_movies = get_new_movies(self.data_dir, top_n=100)
        self.all_movies = get_all_movies(self.data_dir)

        print(f"数据集信息加载完成: {len(self.users)} 用户, {len(self.movies)} 电影")

    def _load_similarity_matrices(self):
        """加载相似度矩阵"""
        similarity_dir = os.path.join(self.data_dir, "processed")

        # 加载用户相似度矩阵
        user_sim_file = os.path.join(similarity_dir, "user_similarity.pkl")
        if os.path.exists(user_sim_file):
            with open(user_sim_file, "rb") as f:
                self.user_similarity_matrix = pickle.load(f)
        else:
            self.user_similarity_matrix = {}

        # 加载电影相似度矩阵
        movie_sim_file = os.path.join(similarity_dir, "movie_similarity.pkl")
        if os.path.exists(movie_sim_file):
            with open(movie_sim_file, "rb") as f:
                self.movie_similarity_matrix = pickle.load(f)
        else:
            self.movie_similarity_matrix = {}

    def load_model(self, model_path):
        """加载训练好的NCF模型（自适应超参）"""
        print(f"加载NCF模型: {model_path}")

        model_state = paddle.load(model_path)

        gmf_weight = model_state.get("gmf.user_embed.weight")
        gmf_embed_dim = (
            gmf_weight.shape[1]
            if gmf_weight is not None and len(gmf_weight.shape) > 1
            else 32
        )

        mlp_weight = model_state.get("mlp.user_embed.weight")
        mlp_embed_dim = (
            mlp_weight.shape[1]
            if mlp_weight is not None and len(mlp_weight.shape) > 1
            else 32
        )

        fusion_weight = model_state.get("fusion.0.weight")
        fusion_input_dim = fusion_weight.shape[0] if fusion_weight is not None else 40

        mlp_output_dim = 16  # 最后一层输出是16
        gmf_output_dim = 1

        base_dim = gmf_output_dim + mlp_output_dim

        if self.use_features and self.use_poster:
            num_features = fusion_input_dim - base_dim - 2048
        elif self.use_features:
            num_features = fusion_input_dim - base_dim
        elif self.use_poster:
            num_features = fusion_input_dim - base_dim - 2048
        else:
            num_features = 0

        num_movie_features = num_features - 4

        print(
            f"  自适应维度: gmf_embed={gmf_embed_dim}, mlp_embed={mlp_embed_dim}, movie_features={num_movie_features}"
        )

        self.model = NCF(
            num_users=self.n_users,
            num_items=self.n_movies,
            gmf_embed_dim=gmf_embed_dim,
            mlp_embed_dim=mlp_embed_dim,
            use_features=self.use_features,
            use_poster=self.use_poster,
            num_user_features=4,
            num_movie_features=max(num_movie_features, 19),
        )

        self.model.set_state_dict(model_state)
        self.model.eval()

        print("NCF模型加载完成")

    def load_sasrec_model(self, model_path):
        """加载训练好的SASRec模型（自适应超参）"""
        print(f"加载SASRec模型: {model_path}")

        checkpoint = paddle.load(model_path)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model_state = checkpoint["state_dict"]
            print(f"  从checkpoint加载，epoch: {checkpoint.get('epoch', 'N/A')}")
        else:
            model_state = checkpoint

        hidden_units = model_state["item_emb.weight"].shape[1]
        max_len = model_state["pos_emb.weight"].shape[0]
        num_heads = (
            model_state["encoder_layer.self_attn.q_proj.weight"].shape[0]
            // hidden_units
        )
        dropout_rate = 0.5
        trained_item_num = model_state["item_emb.weight"].shape[0] - 1  # 减去padding

        print(
            f"  自适应超参: hidden_units={hidden_units}, max_len={max_len}, num_heads={num_heads}, item_num={trained_item_num}"
        )

        self.sasrec_model = SASRec(
            item_num=trained_item_num,
            max_len=max_len,
            hidden_units=hidden_units,
            num_heads=num_heads,
            num_blocks=2,
            dropout_rate=dropout_rate,
        )

        self.sasrec_max_len = max_len
        self.sasrec_model.set_state_dict(model_state)
        self.sasrec_model.eval()

        print("SASRec模型加载完成")

    def _build_user_sequences(self):
        """构建用户交互序列（用于SASRec）"""
        print("构建用户交互序列...")

        ratings = self.ratings.sort_values("timestamp")

        for user_id in self.ratings["user_id"].unique():
            user_ratings = self.ratings[self.ratings["user_id"] == user_id]
            sequence = user_ratings["movie_id"].tolist()
            self.user_sequences[user_id] = sequence

        print(f"用户交互序列构建完成: {len(self.user_sequences)} 用户")

    def compute_user_similarity(self, method="features"):
        """
        计算用户相似度

        Args:
            method: 'features' 基于用户属性, 'ratings' 基于评分行为
        """
        # 检查缓存文件是否存在
        output_file = os.path.join(self.data_dir, "processed", "user_similarity.pkl")
        if os.path.exists(output_file):
            print(f"用户相似度矩阵已存在，跳过计算: {output_file}")
            with open(output_file, "rb") as f:
                self.user_similarity_matrix = pickle.load(f)
            return

        print(f"计算用户相似度 (method={method})...")

        if method == "features":
            # 基于用户属性的相似度
            user_ids = list(self.user_features.keys())
            n_users = len(user_ids)

            similarity_matrix = np.zeros((n_users, n_users))

            for i, uid1 in enumerate(user_ids):
                for j, uid2 in enumerate(user_ids):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    elif i < j:
                        feat1 = self.user_features[uid1]
                        feat2 = self.user_features[uid2]

                        # 特征向量
                        vec1 = np.array(
                            [
                                feat1["gender"],
                                feat1["age"] / 56,
                                feat1["occupation"] / 20,
                                feat1["zipcode_prefix"] / 999,
                            ]
                        )
                        vec2 = np.array(
                            [
                                feat2["gender"],
                                feat2["age"] / 56,
                                feat2["occupation"] / 20,
                                feat2["zipcode_prefix"] / 999,
                            ]
                        )

                        # 余弦相似度
                        sim = np.dot(vec1, vec2) / (
                            np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8
                        )
                        similarity_matrix[i, j] = sim
                        similarity_matrix[j, i] = sim

            # 转换为字典格式
            self.user_similarity_matrix = {
                user_ids[i]: {
                    user_ids[j]: similarity_matrix[i, j] for j in range(n_users)
                }
                for i in range(n_users)
            }

        # 保存相似度矩阵
        output_file = os.path.join(self.data_dir, "processed", "user_similarity.pkl")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "wb") as f:
            pickle.dump(self.user_similarity_matrix, f)

        print(f"用户相似度矩阵已保存: {output_file}")

    def compute_movie_similarity(self, method="features"):
        """
        计算电影相似度

        Args:
            method: 'features' 基于内容特征, 'poster' 基于海报特征
        """
        # 检查缓存文件是否存在
        output_file = os.path.join(self.data_dir, "processed", "movie_similarity.pkl")
        if os.path.exists(output_file):
            print(f"电影相似度矩阵已存在，跳过计算: {output_file}")
            with open(output_file, "rb") as f:
                self.movie_similarity_matrix = pickle.load(f)
            return

        print(f"计算电影相似度 (method={method})...")

        movie_ids = list(self.movie_features.keys())
        n_movies = len(movie_ids)
        similarity_matrix = np.zeros((n_movies, n_movies))

        for i, mid1 in enumerate(movie_ids):
            for j, mid2 in enumerate(movie_ids):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                elif i < j:
                    if method == "features":
                        # 基于内容特征的相似度
                        feat1 = self.movie_features[mid1]
                        feat2 = self.movie_features[mid2]

                        # 年份差异（归一化）
                        year_diff = (
                            abs(feat1["release_year"] - feat2["release_year"]) / 50
                        )

                        # 类型Jaccard相似度
                        set1 = set(
                            idx for idx, v in enumerate(feat1["genres"]) if v > 0
                        )
                        set2 = set(
                            idx for idx, v in enumerate(feat2["genres"]) if v > 0
                        )

                        if len(set1 | set2) > 0:
                            genre_sim = len(set1 & set2) / len(set1 | set2)
                        else:
                            genre_sim = 0

                        # 综合相似度
                        sim = 0.7 * genre_sim + 0.3 * (1 - year_diff)

                    elif method == "poster" and self.use_poster:
                        # 基于海报特征的相似度
                        if (
                            mid1 in self.poster_features
                            and mid2 in self.poster_features
                        ):
                            vec1 = self.poster_features[mid1]
                            vec2 = self.poster_features[mid2]
                            sim = np.dot(vec1, vec2) / (
                                np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8
                            )
                        else:
                            sim = 0
                    else:
                        sim = 0

                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim

        # 转换为字典格式
        self.movie_similarity_matrix = {
            movie_ids[i]: {
                movie_ids[j]: similarity_matrix[i, j] for j in range(n_movies)
            }
            for i in range(n_movies)
        }

        # 保存相似度矩阵
        output_file = os.path.join(self.data_dir, "processed", "movie_similarity.pkl")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "wb") as f:
            pickle.dump(self.movie_similarity_matrix, f)

        print(f"电影相似度矩阵已保存: {output_file}")

    def recommend_popular(self, n=2, exclude_rated=None):
        """
        热门推荐

        Args:
            n: 推荐数量
            exclude_rated: 需要排除的已评分电影ID列表

        Returns:
            推荐电影ID列表
        """
        if exclude_rated is None:
            exclude_rated = set()

        # 选择评分次数最多的电影（热门）
        candidates = [m for m in self.popular_movies if m not in exclude_rated]

        return candidates[:n]

    def recommend_new(self, n=3, exclude_rated=None):
        """
        新品推荐

        Args:
            n: 推荐数量
            exclude_rated: 需要排除的已评分电影ID列表

        Returns:
            推荐电影ID列表
        """
        if exclude_rated is None:
            exclude_rated = set()

        # 选择最近上映的电影（新品）
        candidates = [m for m in self.new_movies if m not in exclude_rated]

        return candidates[:n]

    def recommend_personalized(self, user_id, n=5, method="model"):
        """
        个性化推荐

        Args:
            user_id: 用户ID
            n: 推荐数量
            method: 'model' 使用NCF模型, 'user_sim' 基于相似用户, 'movie_sim' 基于相似电影, 'sasrec' 使用SASRec序列推荐

        Returns:
            推荐电影ID列表
        """
        if method == "model" and self.model is not None:
            return self._recommend_by_model(user_id, n)
        elif method == "sasrec" and self.sasrec_model is not None:
            return self._recommend_by_sasrec(user_id, n)
        elif method == "user_sim":
            return self._recommend_by_similar_users(user_id, n)
        elif method == "movie_sim":
            return self._recommend_by_similar_movies(user_id, n)
        else:
            return self._recommend_hybrid(user_id, n)

    def _recommend_by_model(self, user_id, n=5):
        """使用NCF模型进行个性化推荐"""
        if user_id not in self.user_id_map:
            return []

        user_idx = self.user_id_map[user_id]

        # 排除已评分的电影
        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        rated_movies = set(user_ratings["movie_id"].tolist())

        # 候选电影
        candidates = [m for m in self.all_movies if m not in rated_movies]

        if len(candidates) == 0:
            return []

        # 批量预测
        self.model.eval()

        with paddle.no_grad():
            user_ids = paddle.full([len(candidates)], user_idx, dtype="int64")
            movie_idxs = [self.movie_id_map.get(m, 0) for m in candidates]
            movie_ids = paddle.to_tensor(movie_idxs, dtype="int64")

            # 特征
            if self.use_features and user_id in self.user_features:
                ufeat = self.user_features[user_id]
                user_feature = np.array(
                    [
                        ufeat["gender"],
                        ufeat["age"] / 56,
                        ufeat["occupation"] / 20,
                        ufeat["zipcode_prefix"] / 999,
                    ],
                    dtype="float32",
                )
                user_feature_tensor = paddle.to_tensor(user_feature)
                user_features = paddle.tile(
                    user_feature_tensor.unsqueeze(0), [len(candidates), 1]
                )
            else:
                user_features = None

            # 预测
            if self.use_features and self.use_poster:
                movie_features_list = []
                poster_features_list = []
                for m in candidates:
                    if m in self.movie_features:
                        mfeat = self.movie_features[m]
                        year_norm = (mfeat["release_year"] - 1920) / (2000 - 1920)
                        movie_features_list.append([year_norm] + mfeat["genres"])
                    else:
                        movie_features_list.append([0.5] + [0] * self.n_genres)

                    if m in self.poster_features:
                        poster_features_list.append(self.poster_features[m])
                    else:
                        poster_features_list.append(np.zeros(2048, dtype="float32"))

                movie_features = paddle.to_tensor(
                    np.stack(movie_features_list), dtype="float32"
                )
                poster_features = paddle.to_tensor(
                    np.stack(poster_features_list), dtype="float32"
                )

                predictions = self.model(
                    user_ids, movie_ids, user_features, movie_features, poster_features
                )
            elif self.use_features:
                movie_features_list = []
                for m in candidates:
                    if m in self.movie_features:
                        mfeat = self.movie_features[m]
                        year_norm = (mfeat["release_year"] - 1920) / (2000 - 1920)
                        movie_features_list.append([year_norm] + mfeat["genres"])
                    else:
                        movie_features_list.append([0.5] + [0] * self.n_genres)

                movie_features = paddle.to_tensor(
                    np.stack(movie_features_list), dtype="float32"
                )
                predictions = self.model(
                    user_ids, movie_ids, user_features, movie_features
                )
            else:
                predictions = self.model(user_ids, movie_ids)

            predictions = predictions.numpy().flatten()

            # 获取top-n
            top_indices = np.argsort(predictions)[::-1][:n]
            recommendations = [candidates[i] for i in top_indices]

        return recommendations

    def _recommend_by_similar_users(self, user_id, n=5):
        """基于相似用户的推荐"""
        if user_id not in self.user_similarity_matrix:
            return []

        # 获取相似用户
        user_sims = self.user_similarity_matrix[user_id]
        similar_users = sorted(user_sims.items(), key=lambda x: x[1], reverse=True)[:20]

        # 获取相似用户喜欢的电影
        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        rated_movies = set(user_ratings["movie_id"].tolist())

        movie_scores = {}

        for sim_user, sim_score in similar_users:
            if sim_score < 0.5:  # 只考虑相似度大于0.5的用户
                continue

            sim_user_ratings = self.ratings[self.ratings["user_id"] == sim_user]
            highly_rated = sim_user_ratings[sim_user_ratings["rating"] >= 4]

            for _, row in highly_rated.iterrows():
                if row["movie_id"] not in rated_movies:
                    if row["movie_id"] not in movie_scores:
                        movie_scores[row["movie_id"]] = 0
                    movie_scores[row["movie_id"]] += sim_score * row["rating"]

        # 排序返回top-n
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        return [m[0] for m in sorted_movies[:n]]

    def _recommend_by_similar_movies(self, user_id, n=5):
        """基于相似电影的推荐"""
        if user_id not in self.user_similarity_matrix:
            return []

        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        highly_rated = user_ratings[user_ratings["rating"] >= 4]["movie_id"].tolist()

        # 获取这些电影相似的电影
        movie_scores = {}

        for movie_id in highly_rated:
            if movie_id in self.movie_similarity_matrix:
                similar_movies = self.movie_similarity_matrix[movie_id]
                for sim_movie, sim_score in similar_movies.items():
                    if sim_score < 0.5:  # 只考虑相似度大于0.5的电影
                        continue
                    if sim_movie not in movie_scores:
                        movie_scores[sim_movie] = 0
                    movie_scores[sim_movie] += sim_score

        # 排序返回top-n
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        return [m[0] for m in sorted_movies[:n]]

    def _recommend_by_sasrec(self, user_id, n=5):
        """使用SASRec模型进行序列推荐"""
        if self.sasrec_model is None:
            print("警告: SASRec模型未加载")
            return []

        if user_id not in self.user_sequences:
            return []

        user_history = self.user_sequences[user_id]

        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        rated_movies = set(user_ratings["movie_id"].tolist())

        candidates = [m for m in self.all_movies if m not in rated_movies]

        if len(candidates) == 0:
            return []

        self.sasrec_model.eval()

        with paddle.no_grad():
            max_len = self.sasrec_max_len
            if len(user_history) > max_len:
                seq = user_history[-max_len:]
            else:
                seq = [0] * (max_len - len(user_history)) + user_history

            seq_tensor = paddle.to_tensor([seq], dtype="int64")
            item_indices = [self.movie_id_map.get(m, 0) for m in candidates]

            logits = self.sasrec_model.predict(seq_tensor, item_indices)
            logits = logits.numpy()[0]

            sorted_indices = np.argsort(-logits)
            recommendations = [candidates[i] for i in sorted_indices[:n]]

        return recommendations

    def _recommend_hybrid(self, user_id, n=5):
        """混合推荐策略 - 综合NCF、SASRec、相似用户、相似电影"""
        model_recs = []
        sasrec_recs = []
        user_sim_recs = []
        movie_sim_recs = []

        candidate_multiplier = 3  # 候选倍数，确保去重后有足够结果
        candidate_n = n * candidate_multiplier

        if self.model is not None:
            model_recs = self._recommend_by_model(user_id, candidate_n)
        else:
            print("  提示: NCF模型未加载")

        if self.sasrec_model is not None:
            sasrec_recs = self._recommend_by_sasrec(user_id, candidate_n)
            print(f"  SASRec候选: {len(sasrec_recs)} 条")
        else:
            print("  提示: SASRec模型未加载")

        if self.user_similarity_matrix:
            user_sim_recs = self._recommend_by_similar_users(user_id, candidate_n)
            print(f"  相似用户候选: {len(user_sim_recs)} 条")
        else:
            print("  提示: 用户相似度矩阵未加载")

        if self.movie_similarity_matrix:
            movie_sim_recs = self._recommend_by_similar_movies(user_id, candidate_n)
            print(f"  相似电影候选: {len(movie_sim_recs)} 条")
        else:
            print("  提示: 电影相似度矩阵未加载")

        print(
            f"  合并前: NCF={len(model_recs)}, SASRec={len(sasrec_recs)}, 相似用户={len(user_sim_recs)}, 相似电影={len(movie_sim_recs)}"
        )

        # 合并去重（交替混合：NCF 1个, SASRec 1个, NCF 1个, SASRec 1个... 确保各方法均衡）
        all_recs = []
        seen = set()
        methods = [
            (model_recs, "NCF"),
            (sasrec_recs, "SASRec"),
            (user_sim_recs, "相似用户"),
            (movie_sim_recs, "相似电影"),
        ]

        # 循环遍历所有方法，轮流添加
        max_len = max(len(recs) for recs, _ in methods)
        for i in range(max_len):
            for recs, name in methods:
                if i < len(recs) and recs[i] not in seen:
                    all_recs.append(recs[i])
                    seen.add(recs[i])
                if len(all_recs) >= n:
                    break
            if len(all_recs) >= n:
                break

        return all_recs

    def recommend_cold_start(self, user_features, n=5):
        """
        冷启动用户推荐

        Args:
            user_features: 用户特征字典 {'gender':, 'age':, 'occupation':, 'zipcode_prefix':}
            n: 推荐数量

        Returns:
            推荐电影ID列表（混合热门和新品）
        """
        # 冷启动用户没有历史记录，返回热门+新品推荐
        popular_recs = self.recommend_popular(n=2)
        new_recs = self.recommend_new(n=3)

        return popular_recs + new_recs

    def recommend(self, user_id, n=10, method="hybrid"):
        """
        综合推荐接口

        Args:
            user_id: 用户ID（如果是新用户，使用None或'new_user'）
            n: 总推荐数量 (默认10条 = 热门2 + 新品3 + 个性化5)
            method: 'hybrid' 混合推荐, 'popular' 仅热门, 'new' 仅新品, 'personalized' 仅个性化

        Returns:
            dict: 推荐结果 {
                'popular': [...],  # 热门推荐 (2条)
                'new': [...],      # 新品推荐 (3条)
                'personalized': [...]  # 个性化推荐 (5条)
            }
        """
        # 明确各类型数量: 热门2 + 新品3 + 个性化5 = 10条
        n_popular = 2
        n_new = 3
        n_personalized = n - n_popular - n_new  # 剩余部分

        if user_id is None or user_id == "new_user" or str(user_id).startswith("new_"):
            # 新用户
            result = {
                "popular": self.recommend_popular(n=n_popular),
                "new": self.recommend_new(n=n_new),
                "personalized": self.recommend_cold_start({}, n=n_personalized),
            }
        else:
            # 老用户
            user_id = int(user_id)

            # 获取用户已评分的电影
            user_ratings = self.ratings[self.ratings["user_id"] == user_id]
            rated_movies = set(user_ratings["movie_id"].tolist())

            if method == "popular":
                result = {
                    "popular": self.recommend_popular(
                        n=n_popular, exclude_rated=rated_movies
                    ),
                    "new": [],
                    "personalized": [],
                }
            elif method == "new":
                result = {
                    "popular": [],
                    "new": self.recommend_new(n=n_new, exclude_rated=rated_movies),
                    "personalized": [],
                }
            elif method == "personalized":
                result = {
                    "popular": [],
                    "new": [],
                    "personalized": self.recommend_personalized(
                        user_id, n=n_personalized, method="hybrid"
                    ),
                }
            else:  # hybrid
                result = {
                    "popular": self.recommend_popular(
                        n=n_popular, exclude_rated=rated_movies
                    ),
                    "new": self.recommend_new(n=n_new, exclude_rated=rated_movies),
                    "personalized": self.recommend_personalized(
                        user_id, n=n_personalized, method="hybrid"
                    ),
                }

        return result

    def get_recommendation_details(self, recommendations):
        """
        获取推荐的详细信息

        Args:
            recommendations: 推荐结果字典

        Returns:
            详细信息字典
        """
        details = {}

        for rec_type, movie_ids in recommendations.items():
            details[rec_type] = []
            for mid in movie_ids:
                if mid in self.movie_features:
                    info = self.movie_features[mid].copy()
                    info["movie_id"] = mid
                    info["popularity"] = self.movie_popularity.get(mid, 0)
                    details[rec_type].append(info)

        return details


def main():
    """主函数 - 测试推荐系统"""
    import argparse

    parser = argparse.ArgumentParser(description="电影推荐系统")
    parser.add_argument("--data_dir", type=str, default=None, help="数据目录")
    parser.add_argument("--model_path", type=str, default=None, help="NCF模型文件路径")
    parser.add_argument(
        "--sasrec_model_path", type=str, default=None, help="SASRec模型文件路径"
    )
    parser.add_argument("--use_poster", action="store_true", help="是否使用海报特征")
    parser.add_argument("--device", type=str, default="gpu", help="运行设备 (cpu/gpu)")
    args = parser.parse_args()

    paddle.set_device(args.device)

    # 如果未指定data_dir，使用脚本所在目录
    if args.data_dir is None:
        data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    else:
        data_dir = args.data_dir

    ncf_model_path = (
        args.model_path
        if args.model_path
        else os.path.join(data_dir, "models", "ncf_model.pdparams")
    )
    sasrec_model_path = (
        args.sasrec_model_path
        if args.sasrec_model_path
        else os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "models",
            "SASRec_best.pth.tar",
        )
    )

    print("=" * 60)
    print("电影推荐系统 - SASRec集成测试")
    print("=" * 60)

    print("\n[1/4] 初始化推荐系统...")
    recommender = MovieRecommender(
        data_dir=data_dir,
        model_path=ncf_model_path,
        sasrec_model_path=sasrec_model_path,
        use_features=True,
        use_poster=args.use_poster,
    )
    print(f"  用户数: {recommender.n_users}")
    print(f"  电影数: {recommender.n_movies}")
    print(f"  SASRec模型: {'已加载' if recommender.sasrec_model else '未加载'}")

    test_user_id = 1

    print("\n[2/4] 测试不同推荐方法 (用户 {test_user_id})...")

    movies_df = pd.read_csv(os.path.join(data_dir, "processed", "movies.csv"))

    def movie_id_to_title(movie_ids):
        titles = []
        for mid in movie_ids:
            movie_info = movies_df[movies_df["movie_id"] == mid]
            if not movie_info.empty:
                titles.append(movie_info.iloc[0]["title"])
            else:
                titles.append(f"Unknown (ID:{mid})")
        return titles

    print("\nNCF模型推荐:")
    ncf_recs = recommender._recommend_by_model(test_user_id, 5)
    ncf_titles = movie_id_to_title(ncf_recs)
    for i, title in enumerate(ncf_titles, 1):
        print(f"  {i}. {title}")

    print("\nSASRec序列推荐:")
    sasrec_recs = recommender._recommend_by_sasrec(test_user_id, 5)
    sasrec_titles = movie_id_to_title(sasrec_recs)
    for i, title in enumerate(sasrec_titles, 1):
        print(f"  {i}. {title}")

    print("\n[3/4] 测试综合推荐...")
    recommendations = recommender.recommend(test_user_id, n=10, method="hybrid")
    print(f"\n综合推荐结果:")
    print(f"  热门推荐: {len(recommendations['popular'])} 条")
    print(f"  新品推荐: {len(recommendations['new'])} 条")
    print(f"  个性化推荐: {len(recommendations['personalized'])} 条")

    print("\n[4/4] 显示推荐详情...")

    print("\n热门推荐:")
    for i, mid in enumerate(recommendations["popular"], 1):
        movie_info = movies_df[movies_df["movie_id"] == mid]
        if not movie_info.empty:
            title = movie_info.iloc[0]["title"]
            print(f"  {i}. {title}")

    print("\n新品推荐:")
    for i, mid in enumerate(recommendations["new"], 1):
        movie_info = movies_df[movies_df["movie_id"] == mid]
        if not movie_info.empty:
            title = movie_info.iloc[0]["title"]
            print(f"  {i}. {title}")

    print("\n个性化推荐:")
    for i, mid in enumerate(recommendations["personalized"], 1):
        movie_info = movies_df[movies_df["movie_id"] == mid]
        if not movie_info.empty:
            title = movie_info.iloc[0]["title"]
            print(f"  {i}. {title}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
