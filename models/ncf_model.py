#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NCF模型定义
Neural Collaborative Filtering with Poster Features
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class GMF(nn.Layer):
    """Generalized Matrix Factorization (GMF)"""

    def __init__(self, num_users, num_items, embed_dim=32):
        super(GMF, self).__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.embed_dim = embed_dim

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.initializer.Normal(mean=0.0, std=0.01)(self.user_embed.weight)
        nn.initializer.Normal(mean=0.0, std=0.01)(self.item_embed.weight)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embed(user_ids)
        item_emb = self.item_embed(item_ids)

        # 逐元素相乘
        element_product = user_emb * item_emb

        # 输出层
        output = paddle.sum(element_product, axis=1, keepdim=True)
        return output  # [batch_size, 1]


class MLP(nn.Layer):
    """Multi-Layer Perceptron"""

    def __init__(self, num_users, num_items, embed_dim=32, layers=[64, 32, 16]):
        super(MLP, self).__init__()

        # 嵌入层
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)

        # MLP层
        mlp_layers = []
        input_dim = embed_dim * 2
        for output_dim in layers:
            mlp_layers.append(nn.Linear(input_dim, output_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.2))
            input_dim = output_dim

        self.mlp = nn.LayerList(mlp_layers)
        self.output_dim = layers[-1]

        self._init_weights()

    def _init_weights(self):
        nn.initializer.Normal(mean=0.0, std=0.01)(self.user_embed.weight)
        nn.initializer.Normal(mean=0.0, std=0.01)(self.item_embed.weight)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embed(user_ids)
        item_emb = self.item_embed(item_ids)

        # 拼接
        concat = paddle.concat([user_emb, item_emb], axis=-1)

        # 通过MLP
        for layer in self.mlp:
            concat = layer(concat)

        return concat


class NCF(nn.Layer):
    """Neural Collaborative Filtering (GMF + MLP)"""

    def __init__(
        self,
        num_users,
        num_items,
        gmf_embed_dim=32,
        mlp_embed_dim=32,
        mlp_layers=[64, 32, 16],
        use_features=False,
        use_poster=False,
        num_user_features=4,
        num_movie_features=20,
        poster_feature_dim=2048,
    ):
        super(NCF, self).__init__()

        self.use_features = use_features
        self.use_poster = use_poster

        # GMF部分
        self.gmf = GMF(num_users, num_items, gmf_embed_dim)
        self.gmf_output_dim = 1  # GMF输出是逐元素乘积的和，维度为1

        # MLP部分
        self.mlp = MLP(num_users, num_items, mlp_embed_dim, mlp_layers)
        self.mlp_output_dim = mlp_layers[-1]

        # 特征融合层
        fusion_input_dim = self.gmf_output_dim + self.mlp_output_dim

        if use_features:
            fusion_input_dim += num_user_features + num_movie_features

        if use_poster:
            fusion_input_dim += poster_feature_dim

        # 最终预测层
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        user_ids,
        item_ids,
        user_features=None,
        movie_features=None,
        poster_features=None,
    ):
        # GMF路径 - 输出 [batch_size, 1]
        gmf_out = self.gmf(user_ids, item_ids)

        # MLP路径 - 输出 [batch_size, 16]
        mlp_out = self.mlp(user_ids, item_ids)

        # 融合 GMF [batch_size, 1] + MLP [batch_size, 16] = [batch_size, 17]
        fusion_input = paddle.concat([gmf_out, mlp_out], axis=1)

        # 添加特征
        if self.use_features:
            tensors_to_concat = [fusion_input]
            if user_features is not None:
                tensors_to_concat.append(user_features)
            if movie_features is not None:
                tensors_to_concat.append(movie_features)
            fusion_input = paddle.concat(tensors_to_concat, axis=1)

        # 添加海报特征
        if self.use_poster and poster_features is not None:
            # 对poster_features进行池化（如果输入是3D）
            if len(poster_features.shape) == 3:
                poster_features = paddle.mean(poster_features, axis=1)
            fusion_input = paddle.concat([fusion_input, poster_features], axis=1)

        # 最终预测
        prediction = self.fusion(fusion_input)
        prediction = F.sigmoid(prediction) * 4 + 1  # 缩放到[1, 5]

        return prediction


class UserSimilarityModel(nn.Layer):
    """基于用户相似度的推荐模型"""

    def __init__(self, num_users, num_items, embed_dim=32):
        super(UserSimilarityModel, self).__init__()

        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)

        # 用户特征编码
        self.user_feature_net = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, embed_dim)
        )

        # 电影特征编码
        self.movie_feature_net = nn.Sequential(
            nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, embed_dim)
        )

        nn.initializer.Normal(mean=0.0, std=0.01)(self.user_embed.weight)
        nn.initializer.Normal(mean=0.0, std=0.01)(self.item_embed.weight)

    def forward(self, user_ids, item_ids, user_features):
        # 用户ID嵌入
        user_id_embed = self.user_embed(user_ids)

        # 用户特征嵌入
        user_feat_embed = self.user_feature_net(user_features)

        # 融合用户表示
        user_repr = user_id_embed + user_feat_embed

        # 电影嵌入
        movie_embed = self.item_embed(item_ids)

        # 计算相似度
        similarity = F.cosine_similarity(user_repr, movie_embed, axis=1)

        return similarity.unsqueeze(1)


class MovieSimilarityModel(nn.Layer):
    """基于电影相似度的推荐模型"""

    def __init__(self, num_users, num_items, embed_dim=32):
        super(MovieSimilarityModel, self).__init__()

        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.movie_embed = nn.Embedding(num_items, embed_dim)

        nn.initializer.Normal(mean=0.0, std=0.01)(self.user_embed.weight)
        nn.initializer.Normal(mean=0.0, std=0.01)(self.movie_embed.weight)

    def forward(self, user_ids, item_ids):
        user_embed = self.user_embed(user_ids)
        movie_embed = self.item_embed(item_ids)

        # 计算相似度
        similarity = F.cosine_similarity(user_embed, movie_embed, axis=1)

        return similarity.unsqueeze(1)
