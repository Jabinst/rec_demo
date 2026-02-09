import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

# 数据目录
DATA_DIR = BASE_DIR / "data"
USERS_PATH = DATA_DIR / "users.csv"
ITEMS_PATH = DATA_DIR / "items.csv"
INTERACTIONS_PATH = DATA_DIR / "interactions.csv"

# 模型目录
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "mf_model.npz"
MAPPING_PATH = MODELS_DIR / "id_mappings.npz"


class TrainConfig:
    """训练相关配置"""

    # 矩阵分解维度
    n_factors: int = 32
    # 正则项
    reg: float = 0.01
    # 学习率
    lr: float = 0.05
    # 训练轮数
    n_epochs: int = 30
    # 每个用户保留的最大推荐数（用于离线评估）
    top_k_eval: int = 20


class ServeConfig:
    """在线服务相关配置"""

    # 默认推荐数量
    default_top_k: int = 10


train_config = TrainConfig()
serve_config = ServeConfig()

