"""训练脚本：加载数据 → 训练模型 → 保存模型。

这是算法工程中的核心环节之一。在实际生产环境中，这个脚本可能会：
1. 从 Hive/Spark 读取大规模数据
2. 使用分布式训练框架（如 TensorFlow/PyTorch）
3. 将模型保存到模型仓库（如 MLflow）
4. 触发模型部署流程
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from .config import MODEL_PATH, MAPPING_PATH, train_config
from .data_loader import load_interaction_data
from .model import MatrixFactorizationModel


def train_model() -> None:
    """训练推荐模型并保存。

    流程：
    1. 加载用户-物品交互数据
    2. 训练矩阵分解模型
    3. 保存模型参数和 ID 映射关系
    """
    print("=" * 60)
    print("开始训练推荐模型")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/3] 加载数据...")
    interaction_data = load_interaction_data()
    matrix = interaction_data.user_item_matrix
    print(
        f"  用户数: {matrix.shape[0]}, "
        f"物品数: {matrix.shape[1]}, "
        f"交互数: {matrix.nnz}"
    )

    # 2. 训练模型
    print(f"\n[2/3] 训练模型 (n_factors={train_config.n_factors}, epochs={train_config.n_epochs})...")
    model = MatrixFactorizationModel(
        n_factors=train_config.n_factors,
        reg=train_config.reg,
        lr=train_config.lr,
    )
    model.fit(matrix, n_epochs=train_config.n_epochs, verbose=True)

    # 3. 保存模型
    print("\n[3/3] 保存模型...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 保存模型参数（用户和物品隐向量）
    np.savez(
        MODEL_PATH,
        user_factors=model.user_factors,
        item_factors=model.item_factors,
        n_factors=train_config.n_factors,
    )

    # 保存 ID 映射关系（用于在线服务时转换 user_id <-> user_idx）
    np.savez(
        MAPPING_PATH,
        user_id_to_index=interaction_data.user_id_to_index,
        item_id_to_index=interaction_data.item_id_to_index,
        index_to_user_id=interaction_data.index_to_user_id,
        index_to_item_id=interaction_data.index_to_item_id,
    )

    print(f"  模型已保存到: {MODEL_PATH}")
    print(f"  ID 映射已保存到: {MAPPING_PATH}")
    print("\n训练完成！")


if __name__ == "__main__":
    train_model()
