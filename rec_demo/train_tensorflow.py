"""TensorFlow 版训练脚本：使用 TensorFlow 实现矩阵分解训练。

与 train.py 使用相同的数据和保存格式，训练完成后可直接被 service.py 加载。
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from .config import MODEL_PATH, MAPPING_PATH, train_config
from .data_loader import load_interaction_data


def build_mf_model(n_users: int, n_items: int, n_factors: int) -> tuple[tf.Variable, tf.Variable]:
    """构建用户、物品隐向量矩阵（TensorFlow Variable）。"""
    user_factors = tf.Variable(
        tf.random.normal([n_users, n_factors], stddev=0.1), name="user_factors"
    )
    item_factors = tf.Variable(
        tf.random.normal([n_items, n_factors], stddev=0.1), name="item_factors"
    )
    return user_factors, item_factors


def train_step(
    user_idx: tf.Tensor,
    item_idx: tf.Tensor,
    ratings: tf.Tensor,
    user_factors: tf.Variable,
    item_factors: tf.Variable,
    lr: float,
    reg: float,
) -> float:
    """单步梯度下降：MSE + L2 正则。"""
    with tf.GradientTape() as tape:
        u = tf.gather(user_factors, user_idx)  # (batch, n_factors)
        v = tf.gather(item_factors, item_idx)  # (batch, n_factors)
        pred = tf.reduce_sum(u * v, axis=1)   # (batch,)
        mse = tf.reduce_mean(tf.square(pred - ratings))
        l2 = reg * (
            tf.reduce_sum(tf.square(user_factors)) + tf.reduce_sum(tf.square(item_factors))
        )
        loss = mse + l2
    grads = tape.gradient(loss, [user_factors, item_factors])
    for g, v in zip(grads, [user_factors, item_factors]):
        v.assign_sub(lr * g)
    return float(loss.numpy())


def train_model() -> None:
    """加载数据 → TensorFlow 训练 → 保存为与 train.py 相同的 .npz 格式。"""
    print("=" * 60)
    print("TensorFlow 矩阵分解训练")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/3] 加载数据...")
    interaction_data = load_interaction_data()
    matrix = interaction_data.user_item_matrix
    n_users, n_items = matrix.shape
    user_indices, item_indices = matrix.nonzero()
    ratings = np.array(matrix[user_indices, item_indices]).flatten().astype(np.float32)
    n_interactions = len(ratings)
    print(f"  用户数: {n_users}, 物品数: {n_items}, 交互数: {n_interactions}")

    # 2. 构建模型（变量）
    user_factors, item_factors = build_mf_model(
        n_users, n_items, train_config.n_factors
    )
    n_epochs = train_config.n_epochs
    lr = train_config.lr
    reg = train_config.reg

    print(f"\n[2/3] 训练模型 (n_factors={train_config.n_factors}, epochs={n_epochs})...")
    batch_size = 256
    for epoch in range(n_epochs):
        perm = np.random.permutation(n_interactions)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_interactions, batch_size):
            end = min(start + batch_size, n_interactions)
            idx = perm[start:end]
            u = tf.constant(user_indices[idx], dtype=tf.int32)
            i = tf.constant(item_indices[idx], dtype=tf.int32)
            r = tf.constant(ratings[idx], dtype=tf.float32)
            loss = train_step(u, i, r, user_factors, item_factors, lr, reg)
            total_loss += loss
            n_batches += 1
        avg_loss = total_loss / n_batches
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

    # 3. 保存为与 train.py 相同的 npz 格式，供 service 加载
    print("\n[3/3] 保存模型...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        MODEL_PATH,
        user_factors=user_factors.numpy(),
        item_factors=item_factors.numpy(),
        n_factors=train_config.n_factors,
    )
    np.savez(
        MAPPING_PATH,
        user_id_to_index=interaction_data.user_id_to_index,
        item_id_to_index=interaction_data.item_id_to_index,
        index_to_user_id=interaction_data.index_to_user_id,
        index_to_item_id=interaction_data.index_to_item_id,
    )
    print(f"  模型已保存到: {MODEL_PATH}")
    print(f"  ID 映射已保存到: {MAPPING_PATH}")
    print("\nTensorFlow 训练完成！")


if __name__ == "__main__":
    train_model()
