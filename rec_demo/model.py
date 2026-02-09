"""推荐模型：基于隐式反馈的矩阵分解（Matrix Factorization）。

算法思路：
- 将用户-物品交互矩阵分解为两个低维矩阵：用户隐向量矩阵 U 和物品隐向量矩阵 V
- 预测分数 = U[user] · V[item]
- 使用梯度下降优化，最小化预测误差 + 正则项
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .config import TrainConfig


class MatrixFactorizationModel:
    """简单的矩阵分解模型（用于隐式反馈推荐）。

    这是推荐系统中最经典的算法之一，适合作为入门学习。
    实际工业场景中可能会用更复杂的模型（如 Neural CF、DeepFM 等）。
    """

    def __init__(self, n_factors: int = 32, reg: float = 0.01, lr: float = 0.05):
        """
        参数:
            n_factors: 隐向量维度（越大表达能力越强，但容易过拟合）
            reg: 正则化系数（防止过拟合）
            lr: 学习率
        """
        self.n_factors = n_factors
        self.reg = reg
        self.lr = lr
        self.user_factors: np.ndarray | None = None  # 用户隐向量矩阵 [n_users, n_factors]
        self.item_factors: np.ndarray | None = None  # 物品隐向量矩阵 [n_items, n_factors]

    def fit(
        self,
        user_item_matrix: sparse.csr_matrix,
        n_epochs: int = 30,
        verbose: bool = True,
    ) -> MatrixFactorizationModel:
        """训练模型。

        参数:
            user_item_matrix: 用户-物品交互矩阵（稀疏矩阵，shape: [n_users, n_items]）
            n_epochs: 训练轮数
            verbose: 是否打印训练进度
        """
        n_users, n_items = user_item_matrix.shape

        # 随机初始化用户和物品的隐向量
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # 获取所有非零交互（用于训练）
        user_indices, item_indices = user_item_matrix.nonzero()
        n_interactions = len(user_indices)
        ratings = np.array(user_item_matrix[user_indices, item_indices]).flatten()

        # 梯度下降训练
        for epoch in range(n_epochs):
            total_error = 0.0

            # 随机打乱训练样本（提高训练稳定性）
            indices = np.random.permutation(n_interactions)

            for idx in indices:
                u = user_indices[idx]
                i = item_indices[idx]
                r = ratings[idx]

                # 预测分数
                pred = np.dot(self.user_factors[u], self.item_factors[i])

                # 计算误差
                error = r - pred
                total_error += error**2

                # 更新用户和物品隐向量（梯度下降 + 正则化）
                user_factor_old = self.user_factors[u].copy()
                self.user_factors[u] += self.lr * (
                    error * self.item_factors[i] - self.reg * self.user_factors[u]
                )
                self.item_factors[i] += self.lr * (
                    error * user_factor_old - self.reg * self.item_factors[i]
                )

            # 计算平均误差
            mse = total_error / n_interactions
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, MSE: {mse:.4f}")

        return self

    def predict(self, user_idx: int, item_idx: int) -> float:
        """预测用户对物品的评分。

        参数:
            user_idx: 用户在矩阵中的索引（不是原始 user_id）
            item_idx: 物品在矩阵中的索引（不是原始 item_id）

        返回:
            预测分数
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("模型尚未训练，请先调用 fit()")
        return float(np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))

    def recommend(
        self, user_idx: int, item_indices: np.ndarray | list[int], top_k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """为用户推荐 top-k 物品。

        参数:
            user_idx: 用户在矩阵中的索引
            item_indices: 候选物品索引列表（例如：排除用户已交互过的物品）
            top_k: 返回前 k 个推荐

        返回:
            (推荐的物品索引数组, 对应的预测分数数组)
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("模型尚未训练，请先调用 fit()")

        # 计算用户对所有候选物品的预测分数
        user_vector = self.user_factors[user_idx]
        item_vectors = self.item_factors[item_indices]
        scores = np.dot(item_vectors, user_vector)

        # 取 top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_item_indices = item_indices[top_indices]
        top_scores = scores[top_indices]

        return top_item_indices, top_scores
