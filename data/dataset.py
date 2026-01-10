#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据加载模块
负责加载和处理训练/测试数据
"""

import os
import pickle
import numpy as np
import pandas as pd
import paddle
from paddle.io import Dataset, DataLoader


class MovieLensDataset(Dataset):
    """MovieLens数据集"""

    def __init__(self, data_dir, mode="train", use_features=True, use_poster=False):
        """
        初始化数据集

        Args:
            data_dir: 数据目录
            mode: 'train' 或 'test'
            use_features: 是否使用特征
            use_poster: 是否使用海报特征
        """
        self.data_dir = data_dir
        self.mode = mode
        self.use_features = use_features
        self.use_poster = use_poster

        # 加载数据
        self.users = pd.read_csv(os.path.join(data_dir, "processed", "users.csv"))
        self.movies = pd.read_csv(os.path.join(data_dir, "processed", "movies.csv"))

        if mode == "train":
            self.ratings = pd.read_csv(
                os.path.join(data_dir, "processed", "train_ratings.csv")
            )
        else:
            self.ratings = pd.read_csv(
                os.path.join(data_dir, "processed", "test_ratings.csv")
            )

        # 构建ID映射（从1开始，便于paddle embedding）
        self._build_mappings()

        # 加载特征
        self._load_features()

        print(f"数据集加载完成: {mode} 模式, {len(self.ratings)} 条记录")

    def _build_mappings(self):
        """构建ID映射"""
        # 用户ID映射
        self.user_id_map = {
            uid: idx + 1
            for idx, uid in enumerate(sorted(self.users["user_id"].unique()))
        }
        self.user_id_reverse = {
            idx + 1: uid
            for idx, uid in enumerate(sorted(self.users["user_id"].unique()))
        }

        # 电影ID映射
        self.movie_id_map = {
            mid: idx + 1
            for idx, mid in enumerate(sorted(self.movies["movie_id"].unique()))
        }
        self.movie_id_reverse = {
            idx + 1: mid
            for idx, mid in enumerate(sorted(self.movies["movie_id"].unique()))
        }

        # 统计信息
        self.n_users = len(self.user_id_map) + 1  # +1 用于padding
        self.n_movies = len(self.movie_id_map) + 1
        self.n_genres = len([c for c in self.movies.columns if c.startswith("genre_")])

    def _load_features(self):
        """加载特征数据"""
        # 用户特征
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

        # 电影特征
        self.movie_features = {}
        genre_cols = [
            c
            for c in self.movies.columns
            if c.startswith("genre_") and c != "genre_list"
        ]

        for _, row in self.movies.iterrows():
            mid = row["movie_id"]
            self.movie_features[mid] = {
                "release_year": row["release_year"]
                if pd.notna(row["release_year"])
                else 1990,
                "genres": [row[c] for c in genre_cols],
            }

        # 海报特征
        poster_features_file = os.path.join(
            self.data_dir, "processed", "poster_features.pkl"
        )
        if self.use_poster and os.path.exists(poster_features_file):
            with open(poster_features_file, "rb") as f:
                self.poster_features = pickle.load(f)
        else:
            self.poster_features = {}

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        """获取单个样本"""
        row = self.ratings.iloc[idx]

        user_id = int(row["user_id"])
        movie_id = int(row["movie_id"])
        rating = float(row["rating"])

        # 映射到embedding索引
        user_idx = self.user_id_map.get(user_id, 0)
        movie_idx = self.movie_id_map.get(movie_id, 0)

        # 用户特征
        if self.use_features and user_id in self.user_features:
            ufeat = self.user_features[user_id]
            user_feature = np.array(
                [
                    ufeat["gender"],
                    ufeat["age"] / 56,  # 归一化
                    ufeat["occupation"] / 20,
                    ufeat["zipcode_prefix"] / 999,
                ],
                dtype="float32",
            )
        else:
            user_feature = np.zeros(4, dtype="float32")

        # 电影特征
        if self.use_features and movie_id in self.movie_features:
            mfeat = self.movie_features[movie_id]
            year_norm = (mfeat["release_year"] - 1920) / (2000 - 1920)
            movie_feature = np.array([year_norm] + mfeat["genres"], dtype="float32")
        else:
            movie_feature = np.zeros(self.n_genres + 1, dtype="float32")

        # 海报特征
        if self.use_poster and movie_id in self.poster_features:
            poster_feat = self.poster_features[movie_id]
        else:
            poster_feat = np.zeros(2048, dtype="float32")  # ResNet50特征维度

        return {
            "user_id": user_idx,
            "movie_id": movie_idx,
            "rating": rating,
            "user_feature": user_feature,
            "movie_feature": movie_feature,
            "poster_feature": poster_feat,
            "original_user_id": user_id,
            "original_movie_id": movie_id,
        }

    def get_all_movie_features(self):
        """获取所有电影的特征矩阵，用于推荐评估"""
        n_movies = len(self.movie_id_map) + 1  # +1 for padding
        n_features = self.n_genres + 1  # year + genres

        features = np.zeros((n_movies, n_features), dtype="float32")

        for movie_id, idx in self.movie_id_map.items():
            if movie_id in self.movie_features:
                mfeat = self.movie_features[movie_id]
                year_norm = (mfeat["release_year"] - 1920) / (2000 - 1920)
                features[idx] = np.array([year_norm] + mfeat["genres"], dtype="float32")

        return features

    def get_all_poster_features(self):
        """获取所有电影的海报特征矩阵，用于推荐评估"""
        n_movies = len(self.movie_id_map) + 1  # +1 for padding
        poster_dim = 2048  # ResNet50 feature dimension

        features = np.zeros((n_movies, poster_dim), dtype="float32")

        for movie_id, idx in self.movie_id_map.items():
            if movie_id in self.poster_features:
                features[idx] = np.array(
                    self.poster_features[movie_id], dtype="float32"
                )

        return features


class InferenceDataset(Dataset):
    """推理数据集（用于生成推荐）"""

    def __init__(
        self, users, movies, user_features, movie_features, poster_features=None
    ):
        """
        初始化推理数据集

        Args:
            users: 用户ID列表
            movies: 电影ID列表
            user_features: 用户特征字典
            movie_features: 电影特征字典
            poster_features: 海报特征字典
        """
        self.users = users
        self.movies = movies
        self.user_features = user_features
        self.movie_features = movie_features
        self.poster_features = poster_features or {}

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user_id = self.users[idx]

        # 用户特征
        if user_id in self.user_features:
            ufeat = self.user_features[user_id]
            user_feature = np.array(
                [
                    ufeat.get("gender", 0),
                    ufeat.get("age", 25) / 56,
                    ufeat.get("occupation", 0) / 20,
                    ufeat.get("zipcode_prefix", 0) / 999,
                ],
                dtype="float32",
            )
        else:
            user_feature = np.zeros(4, dtype="float32")

        # 电影特征
        movie_features_list = []
        for movie_id in self.movies:
            if movie_id in self.movie_features:
                mfeat = self.movie_features[movie_id]
                year_norm = (mfeat.get("release_year", 1990) - 1920) / (2000 - 1920)
                movie_feat = np.array(
                    [year_norm] + mfeat.get("genres", []), dtype="float32"
                )
            else:
                n_genres = len(
                    self.movie_features[list(self.movie_features.keys())[0]]["genres"]
                )
                movie_feat = np.zeros(n_genres + 1, dtype="float32")
            movie_features_list.append(movie_feat)

        # 海报特征
        poster_features_list = []
        for movie_id in self.movies:
            if movie_id in self.poster_features:
                poster_features_list.append(self.poster_features[movie_id])
            else:
                poster_features_list.append(np.zeros(2048, dtype="float32"))

        return {
            "user_id": user_id,
            "user_feature": user_feature,
            "movie_features": np.stack(movie_features_list),
            "poster_features": np.stack(poster_features_list),
        }


def create_data_loaders(data_dir, batch_size=256, use_features=True, use_poster=False):
    """创建训练和测试数据加载器"""
    train_dataset = MovieLensDataset(
        data_dir, mode="train", use_features=use_features, use_poster=use_poster
    )
    test_dataset = MovieLensDataset(
        data_dir, mode="test", use_features=use_features, use_poster=use_poster
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset


def get_popular_movies(data_dir, top_n=100):
    """获取热门电影列表（按评分数量）"""
    ratings = pd.read_csv(os.path.join(data_dir, "processed", "ratings.csv"))
    movie_counts = ratings.groupby("movie_id").size().sort_values(ascending=False)
    return movie_counts.head(top_n).index.tolist()


def get_new_movies(data_dir, top_n=100, years=10):
    """获取新电影列表（按首映时间）"""
    movies = pd.read_csv(os.path.join(data_dir, "processed", "movies.csv"))
    recent_year = movies["release_year"].max() - years
    new_movies = movies[movies["release_year"] >= recent_year].sort_values(
        "release_year", ascending=False
    )
    return new_movies.head(top_n)["movie_id"].tolist()


def get_all_movies(data_dir):
    """获取所有电影ID列表"""
    movies = pd.read_csv(os.path.join(data_dir, "processed", "movies.csv"))
    return movies["movie_id"].tolist()
