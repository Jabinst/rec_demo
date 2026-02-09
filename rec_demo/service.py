"""在线推荐服务：加载模型 → 提供 HTTP API。

这是算法工程中的在线服务环节。在实际生产环境中，这个服务可能会：
1. 使用更高效的推理框架（如 TensorFlow Serving、TorchServe）
2. 集成特征服务（Feature Store）获取实时特征
3. 使用缓存（Redis）加速推荐
4. 部署到 Kubernetes，支持水平扩展
5. 集成监控和日志系统
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import MODEL_PATH, MAPPING_PATH, serve_config
from .data_loader import load_items, load_users
from .model import MatrixFactorizationModel


# 全局变量：模型和映射关系（在实际生产环境中，这些可能会通过依赖注入或单例模式管理）
_model: MatrixFactorizationModel | None = None
_id_mappings: Dict | None = None
_items_df = None
_users_df = None


def load_model() -> None:
    """加载训练好的模型和 ID 映射关系。

    在实际生产环境中，这个函数可能会：
    - 从模型仓库（如 MLflow、S3）下载最新模型
    - 使用模型版本管理
    - 支持热更新（不重启服务就能切换模型）
    """
    global _model, _id_mappings, _items_df, _users_df

    if not MODEL_PATH.exists() or not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"模型文件不存在，请先运行训练脚本：\n"
            f"  python -m rec_demo.train\n"
            f"  或: python rec_demo/train.py"
        )

    # 加载模型参数
    model_data = np.load(MODEL_PATH, allow_pickle=True)
    _model = MatrixFactorizationModel(n_factors=int(model_data["n_factors"]))
    _model.user_factors = model_data["user_factors"]
    _model.item_factors = model_data["item_factors"]

    # 加载 ID 映射
    mapping_data = np.load(MAPPING_PATH, allow_pickle=True)
    _id_mappings = {
        "user_id_to_index": mapping_data["user_id_to_index"].item(),
        "item_id_to_index": mapping_data["item_id_to_index"].item(),
        "index_to_user_id": mapping_data["index_to_user_id"].item(),
        "index_to_item_id": mapping_data["index_to_item_id"].item(),
    }

    # 加载物品和用户元数据（用于返回推荐结果时展示标题等信息）
    _items_df = load_items()
    _users_df = load_users()

    print(f"✓ 模型加载成功 (n_factors={_model.n_factors})")
    print(f"✓ 用户数: {len(_id_mappings['user_id_to_index'])}, 物品数: {len(_id_mappings['item_id_to_index'])}")


# 创建 FastAPI 应用
app = FastAPI(
    title="博客文章推荐系统 API",
    description="提供个性化文章推荐服务",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型。"""
    load_model()


# ========== API 接口定义 ==========


class RecommendationRequest(BaseModel):
    """推荐请求参数。"""

    user_id: int
    top_k: int = serve_config.default_top_k


class RecommendationItem(BaseModel):
    """推荐结果中的单个物品。"""

    item_id: int
    title: str
    tags: str
    score: float


class RecommendationResponse(BaseModel):
    """推荐响应。"""

    user_id: int
    recommendations: List[RecommendationItem]


@app.get("/")
async def root():
    """健康检查接口。"""
    return {
        "message": "博客文章推荐系统 API",
        "status": "running",
        "endpoints": {
            "推荐接口": "/api/recommend",
            "文档": "/docs",
        },
    }


@app.post("/api/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest) -> RecommendationResponse:
    """为用户推荐文章。

    这是推荐系统的核心接口。在实际生产环境中，这个接口可能会：
    1. 先做召回（从海量物品中筛选出候选集，如使用向量检索）
    2. 再做排序（使用更复杂的模型，如深度学习模型）
    3. 最后做重排（考虑多样性、新鲜度等业务规则）
    """
    if _model is None or _id_mappings is None:
        raise HTTPException(status_code=500, detail="模型未加载，请检查服务状态")

    user_id = request.user_id
    top_k = request.top_k

    # 检查用户是否存在
    if user_id not in _id_mappings["user_id_to_index"]:
        raise HTTPException(status_code=404, detail=f"用户 {user_id} 不存在")

    user_idx = _id_mappings["user_id_to_index"][user_id]

    # 获取所有物品索引（在实际场景中，这里应该是召回后的候选集）
    all_item_indices = np.array(list(_id_mappings["item_id_to_index"].values()))

    # 获取用户已交互过的物品（排除这些，避免重复推荐）
    # 注意：这里简化处理，实际场景中需要从数据库或缓存中获取用户历史
    # 为了演示，我们假设所有物品都是候选

    # 调用模型推荐
    recommended_item_indices, scores = _model.recommend(
        user_idx=user_idx, item_indices=all_item_indices, top_k=top_k
    )

    # 将物品索引转换为物品 ID，并获取物品元数据
    recommendations = []
    for item_idx, score in zip(recommended_item_indices, scores):
        item_id = _id_mappings["index_to_item_id"][int(item_idx)]
        item_info = _items_df[_items_df["item_id"] == item_id].iloc[0]

        recommendations.append(
            RecommendationItem(
                item_id=int(item_id),
                title=item_info["title"],
                tags=item_info.get("tags", ""),
                score=float(score),
            )
        )

    return RecommendationResponse(user_id=user_id, recommendations=recommendations)


@app.get("/api/user/{user_id}/info")
async def get_user_info(user_id: int):
    """获取用户信息（用于调试和展示）。"""
    if _users_df is None:
        raise HTTPException(status_code=500, detail="用户数据未加载")

    user_info = _users_df[_users_df["user_id"] == user_id]
    if user_info.empty:
        raise HTTPException(status_code=404, detail=f"用户 {user_id} 不存在")

    return user_info.iloc[0].to_dict()


@app.get("/api/item/{item_id}/info")
async def get_item_info(item_id: int):
    """获取物品信息（用于调试和展示）。"""
    if _items_df is None:
        raise HTTPException(status_code=500, detail="物品数据未加载")

    item_info = _items_df[_items_df["item_id"] == item_id]
    if item_info.empty:
        raise HTTPException(status_code=404, detail=f"物品 {item_id} 不存在")

    return item_info.iloc[0].to_dict()


if __name__ == "__main__":
    import uvicorn

    # 启动服务（默认端口 8000）
    uvicorn.run(app, host="0.0.0.0", port=8000)
