"""PyTorch 版训练脚本：使用 PyTorch 实现矩阵分解训练。

与 train.py 使用相同的数据和保存格式，训练完成后可直接被 service.py 加载。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .config import MODEL_PATH, MAPPING_PATH, train_config
from .data_loader import load_interaction_data


class MFDataset(Dataset):
    """用户-物品-分数 三元组数据集。"""

    def __init__(self, user_indices: np.ndarray, item_indices: np.ndarray, ratings: np.ndarray):
        self.user_indices = torch.LongTensor(user_indices)
        self.item_indices = torch.LongTensor(item_indices)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.user_indices[idx], self.item_indices[idx], self.ratings[idx]


class MatrixFactorizationPyTorch(nn.Module):
    """PyTorch 矩阵分解模型：用户/物品隐向量用 Embedding 表示，预测分数为内积。"""

    def __init__(self, n_users: int, n_items: int, n_factors: int):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        nn.init.normal_(self.user_emb.weight, 0, 0.1)
        nn.init.normal_(self.item_emb.weight, 0, 0.1)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_idx)   # (batch, n_factors)
        v = self.item_emb(item_idx)   # (batch, n_factors)
        return (u * v).sum(dim=1)     # (batch,)


def train_model() -> None:
    """加载数据 → PyTorch 训练 → 保存为与 train.py 相同的 .npz 格式。"""
    print("=" * 60)
    print("PyTorch 矩阵分解训练")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/3] 加载数据...")
    interaction_data = load_interaction_data()
    matrix = interaction_data.user_item_matrix
    n_users, n_items = matrix.shape
    user_indices, item_indices = matrix.nonzero()
    ratings = np.array(matrix[user_indices, item_indices]).flatten().astype(np.float32)
    print(f"  用户数: {n_users}, 物品数: {n_items}, 交互数: {len(ratings)}")

    # 2. 构建 DataLoader
    dataset = MFDataset(user_indices, item_indices, ratings)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # 3. 模型与优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MatrixFactorizationPyTorch(n_users, n_items, train_config.n_factors).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=train_config.lr)
    n_epochs = train_config.n_epochs

    print(f"\n[2/3] 训练模型 (n_factors={train_config.n_factors}, epochs={n_epochs}, device={device})...")
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0
        for u, i, r in dataloader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            optimizer.zero_grad()
            pred = model(u, i)
            mse = ((pred - r) ** 2).mean()
            # L2 正则
            reg = train_config.reg * (
                model.user_emb.weight.pow(2).sum() + model.item_emb.weight.pow(2).sum()
            )
            loss = mse + reg
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

    # 4. 保存为与 train.py 相同的 npz 格式，供 service 加载
    print("\n[3/3] 保存模型...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    user_factors = model.user_emb.weight.detach().cpu().numpy()
    item_factors = model.item_emb.weight.detach().cpu().numpy()
    np.savez(
        MODEL_PATH,
        user_factors=user_factors,
        item_factors=item_factors,
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
    print("\nPyTorch 训练完成！")


if __name__ == "__main__":
    train_model()
