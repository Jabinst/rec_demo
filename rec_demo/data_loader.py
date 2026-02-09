from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from .config import USERS_PATH, ITEMS_PATH, INTERACTIONS_PATH


@dataclass
class InteractionData:
    """封装训练所需的核心数据结构。"""

    user_item_matrix: sparse.csr_matrix
    user_id_to_index: Dict[int, int]
    item_id_to_index: Dict[int, int]
    index_to_user_id: Dict[int, int]
    index_to_item_id: Dict[int, int]


def load_users(path: Path = USERS_PATH) -> pd.DataFrame:
    """加载用户表（示例字段：user_id, user_name）。"""
    if not path.exists():
        # 提供一个最小示例，方便第一次运行
        df = pd.DataFrame({"user_id": [1, 2, 3], "user_name": ["Alice", "Bob", "Carol"]})
        path.parent.mkdir(exist_ok=True)
        df.to_csv(path, index=False)
        return df
    return pd.read_csv(path)


def load_items(path: Path = ITEMS_PATH) -> pd.DataFrame:
    """加载文章表（示例字段：item_id, title, tags）。"""
    if not path.exists():
        df = pd.DataFrame(
            {
                "item_id": [101, 102, 103, 104],
                "title": ["Python 入门", "推荐系统概览", "深度学习基础", "工程实践指南"],
                "tags": ["python,基础", "推荐,算法", "深度学习,基础", "工程,实践"],
            }
        )
        path.parent.mkdir(exist_ok=True)
        df.to_csv(path, index=False)
        return df
    return pd.read_csv(path)


def load_interactions(path: Path = INTERACTIONS_PATH) -> pd.DataFrame:
    """加载用户-文章交互日志。

    示例字段：user_id, item_id, clicks, likes
    会根据 clicks/likes 计算一个隐式反馈强度。
    """
    if not path.exists():
        # 构造一个简单的示例交互表
        df = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 2, 3],
                "item_id": [101, 102, 101, 103, 104, 104],
                "clicks": [3, 1, 5, 2, 1, 4],
                "likes": [1, 0, 1, 0, 1, 1],
            }
        )
        path.parent.mkdir(exist_ok=True)
        df.to_csv(path, index=False)
        return df
    return pd.read_csv(path)


def build_interaction_matrix(
    users: pd.DataFrame,
    items: pd.DataFrame,
    interactions: pd.DataFrame,
    click_weight: float = 1.0,
    like_weight: float = 3.0,
) -> InteractionData:
    """将原始交互日志转成 user-item 稀疏矩阵。

    参数:
        click_weight: 点击行为的权重
        like_weight: 点赞行为的权重
    """
    # 只保留存在于用户表和物品表中的记录
    interactions = interactions.merge(users[["user_id"]], on="user_id", how="inner")
    interactions = interactions.merge(items[["item_id"]], on="item_id", how="inner")

    # 计算隐式反馈强度分数
    interactions["score"] = (
        interactions.get("clicks", 0) * click_weight
        + interactions.get("likes", 0) * like_weight
    )

    # 过滤掉 score <= 0 的记录
    interactions = interactions[interactions["score"] > 0]

    unique_user_ids = interactions["user_id"].unique()
    unique_item_ids = interactions["item_id"].unique()

    user_id_to_index = {uid: i for i, uid in enumerate(unique_user_ids)}
    item_id_to_index = {iid: i for i, iid in enumerate(unique_item_ids)}
    index_to_user_id = {i: uid for uid, i in user_id_to_index.items()}
    index_to_item_id = {i: iid for iid, i in item_id_to_index.items()}

    user_indices = interactions["user_id"].map(user_id_to_index).values
    item_indices = interactions["item_id"].map(item_id_to_index).values
    scores = interactions["score"].astype(float).values

    user_item_matrix = sparse.coo_matrix(
        (scores, (user_indices, item_indices)),
        shape=(len(unique_user_ids), len(unique_item_ids)),
    ).tocsr()

    return InteractionData(
        user_item_matrix=user_item_matrix,
        user_id_to_index=user_id_to_index,
        item_id_to_index=item_id_to_index,
        index_to_user_id=index_to_user_id,
        index_to_item_id=index_to_item_id,
    )


def load_interaction_data() -> InteractionData:
    """对外暴露的一站式加载函数：读 CSV → 构造稀疏矩阵。"""
    users = load_users()
    items = load_items()
    interactions = load_interactions()
    return build_interaction_matrix(users, items, interactions)

