## 博客文章个性化推荐 Demo

这个项目是一个**极简的推荐系统工程示例**，围绕「博客文章推荐」场景，展示从**数据 → 训练 → 模型保存 → 在线服务推荐**的完整闭环，方便你理解大厂里推荐工程的基本形态。

### 场景设定（可类比真实业务）

- 有一批博客文章：标题、标签、作者、发布时间等。
- 有一批用户，对文章产生过：
  - 点击（clicks）
  - 点赞（likes）
- 我们把这些行为当作**隐式反馈**，利用协同过滤/矩阵分解，学习用户和文章的向量表示（embedding），为每个用户推荐他们可能喜欢的文章。

### 算法工程分工说明

这个 Demo 展示了推荐系统中**算法团队**和**工程团队**的典型分工：

- **算法团队的工作**（体现在 `model.py` 和 `train.py`）：
  - 设计推荐算法（矩阵分解）
  - 特征工程（将点击/点赞转为隐式反馈分数）
  - 模型训练与调参
  - 离线评估模型效果

- **工程团队的工作**（体现在 `data_loader.py` 和 `service.py`）：
  - 数据加载与预处理（从 CSV/数据库读取数据）
  - 模型服务化（将训练好的模型包装成 HTTP API）
  - 在线推理优化（加载模型、缓存、性能优化）

在实际大厂中，这个流程会更加复杂：
- 数据可能来自 Hive/Spark，需要大规模 ETL
- 训练可能使用 TensorFlow/PyTorch，需要分布式训练
- 在线服务可能使用 C++/Java，需要毫秒级响应
- 需要特征平台、模型平台、实验平台等基础设施

### 工程结构概览

```text
rec-demo/
├── README.md                # 项目说明（你现在在看这个）
├── requirements.txt         # Python 依赖
├── data/
│   ├── users.csv            # 示例用户数据（可扩展）
│   ├── items.csv            # 示例文章数据（可扩展）
│   └── interactions.csv     # 用户-文章行为日志（点击/点赞等）
└── rec_demo/
    ├── __init__.py
    ├── config.py            # 配置，例如文件路径、模型保存路径等
    ├── data_loader.py       # 数据加载与预处理
    ├── model.py             # 推荐模型实现（矩阵分解 / 协同过滤）
    ├── train.py             # 离线训练脚本：读数据 → 训练模型 → 保存
    └── serve.py             # 在线服务：加载模型 → 提供 HTTP 推荐接口
```

> 设计思路：尽量贴近「算法工程」最常见的分层——**数据层（data_loader） + 算法层（model + train） + 服务层（serve）**，方便你今后扩展到更复杂的场景。

---

## 1. 安装与环境准备

```bash
cd /Users/jabin/projects/python_projects/rec-demo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Python 版本建议：**3.9+**。

---

## 2. 示例数据说明

我们在 `data/` 目录下放了三类 CSV（格式简单直观，方便你自己改造）：

- `users.csv`
  - 字段示例：`user_id, user_name`
- `items.csv`
  - 字段示例：`item_id, title, tags`
- `interactions.csv`
  - 字段示例：`user_id, item_id, clicks, likes`
  - clicks / likes 会被转成一个**隐式反馈强度**，作为矩阵分解的目标值。

你可以直接替换成自己更加真实的日志数据，只要字段含义一致即可。

---

## 3. 训练：离线构建推荐模型

训练入口在 `rec_demo/train.py`，核心流程：

1. 使用 `data_loader.py` 读取并预处理三类 CSV。
2. 将 (user_id, item_id, 行为强度) 转成稀疏交互矩阵。
3. 用 `model.py` 中实现的**矩阵分解（MF，使用隐式反馈）**训练用户/物品 embedding。
4. 将训练好的模型参数（embedding 矩阵、索引映射等）保存到磁盘。

执行示例：

```bash
cd /Users/jabin/projects/python_projects/rec-demo
source venv/bin/activate
python -m rec_demo.train
```

训练完成后，会在 `models/` 目录下看到序列化好的模型文件。

---

## 4. 在线服务：提供推荐接口

在线服务入口在 `rec_demo/serve.py`，使用 **FastAPI** 搭建一个简单 HTTP 服务：

- 启动命令：

```bash
cd /Users/jabin/projects/python_projects/rec-demo
source venv/bin/activate
uvicorn rec_demo.serve:app --reload --port 8000
```

- 主要接口：
  - `GET /`：健康检查
  - `POST /api/recommend`：获取个性化推荐
  - `GET /api/user/{user_id}/info`：获取用户信息
  - `GET /api/item/{item_id}/info`：获取文章信息
  - `GET /docs`：自动生成的 API 文档（Swagger UI）

服务启动后，可以用以下方式测试：

**方式1：使用 curl**
```bash
# 获取用户 1 的 Top-5 推荐
curl -X POST "http://127.0.0.1:8000/api/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "top_k": 5}'
```

**方式2：使用浏览器**
- 访问 `http://127.0.0.1:8000/docs` 查看交互式 API 文档
- 在文档页面直接测试接口

**方式3：使用 Python**
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/api/recommend",
    json={"user_id": 1, "top_k": 5}
)
print(response.json())
```

---

## 5. 快速开始：完整流程演示

### 步骤 1：安装依赖
```bash
cd /Users/jabin/projects/python_projects/rec-demo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 步骤 2：训练模型
```bash
python -m rec_demo.train
```

你会看到类似输出：
```
============================================================
开始训练推荐模型
============================================================

[1/3] 加载数据...
  用户数: 3, 物品数: 4, 交互数: 6

[2/3] 训练模型 (n_factors=32, epochs=30)...
Epoch 5/30, MSE: 0.1234
Epoch 10/30, MSE: 0.0567
...

[3/3] 保存模型...
  模型已保存到: models/mf_model.npz
  ID 映射已保存到: models/id_mappings.npz

训练完成！
```

### 步骤 3：启动推荐服务
```bash
uvicorn rec_demo.service:app --reload --port 8000
```

### 步骤 4：测试推荐接口
打开浏览器访问 `http://127.0.0.1:8000/docs`，在交互式文档中测试推荐接口。

或者使用 curl：
```bash
curl -X POST "http://127.0.0.1:8000/api/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "top_k": 3}'
```

---

## 6. 学习建议：从这个 Demo 学什么？

你可以重点从这几个角度来理解「算法工程」：

### 6.1 数据到特征
- `data_loader.py` 如何把原始 CSV 转成训练所需的稀疏矩阵
- 如何将多种行为（点击、点赞）融合成隐式反馈分数
- 如何处理用户-物品 ID 与矩阵索引的映射关系

### 6.2 算法到工程接口
- `model.py` 里只关心「怎么根据交互数据拟合出 user/item embedding」
- `train.py` 做了「算法脚本工程化」：参数配置、日志、模型保存
- 这是算法同学常写的代码风格

### 6.3 离线到在线
- `service.py` 不再做训练，只负责「加载已有模型 + 提供查询接口」
- 这就是大厂里常见的 **离线训练 / 在线推理** 分离的工程模式
- 在线服务需要关注：性能、稳定性、可扩展性

### 6.4 扩展方向

等你熟悉了之后，可以自然延伸到：

- **更复杂的模型**：Wide&Deep、DeepFM、DIN 等深度学习模型
- **多路召回**：基于内容的召回、热门召回、协同过滤召回
- **实时特征**：使用 Flink/Kafka 计算实时特征（如最近 10 分钟点击次数）
- **特征平台**：统一管理特征定义和生产流程
- **A/B 实验**：集成实验平台，支持多模型对比
- **性能优化**：模型量化、向量检索（Faiss）、缓存策略

---

---

## 7. 项目结构详解

```
rec-demo/
├── README.md                # 项目说明文档
├── requirements.txt         # Python 依赖包
├── data/                    # 数据目录（首次运行会自动生成示例数据）
│   ├── users.csv            # 用户表
│   ├── items.csv            # 文章表
│   └── interactions.csv     # 用户-文章交互日志
├── models/                  # 模型目录（训练后生成）
│   ├── mf_model.npz         # 训练好的模型参数
│   └── id_mappings.npz      # ID 映射关系
└── rec_demo/                # 核心代码包
    ├── __init__.py
    ├── config.py            # 配置管理（路径、超参数等）
    ├── data_loader.py       # 数据加载与预处理
    ├── model.py             # 矩阵分解模型实现
    ├── train.py             # 离线训练脚本
    └── service.py           # 在线推荐服务（FastAPI）
```

### 各模块职责

| 模块 | 职责 | 对应团队 |
|------|------|----------|
| `data_loader.py` | 数据加载、特征工程、构建交互矩阵 | 算法 + 工程 |
| `model.py` | 推荐算法实现（矩阵分解） | 算法 |
| `train.py` | 模型训练流程、模型保存 | 算法 |
| `service.py` | 在线服务、API 接口、模型加载 | 工程 |

---

## 8. 下一步：如何扩展

1. **先跑通当前 Demo**：确保训练和服务都能正常运行，能拿到推荐结果

2. **尝试修改参数**：
   - 调整 `config.py` 中的 `n_factors`、`n_epochs` 等超参数
   - 调整 `data_loader.py` 中的 `click_weight`、`like_weight`，看推荐结果变化

3. **增加业务规则**：
   - 在 `service.py` 中过滤用户已看过的文章
   - 按时间衰减（新文章优先）
   - 增加多样性（避免推荐相似文章）

4. **扩展数据**：
   - 在 `items.csv` 中增加更多特征（作者、分类、发布时间等）
   - 在 `interactions.csv` 中增加更多行为（评论、收藏、分享等）
   - 修改 `data_loader.py` 使用这些新特征

5. **升级模型**：
   - 替换为更复杂的模型（如使用 TensorFlow/PyTorch 实现 Wide&Deep）
   - 增加内容特征（基于标签的召回）
   - 实现多路召回 + 排序的完整流程

---

## 9. 常见问题

**Q: 为什么使用矩阵分解而不是深度学习？**  
A: 这个 Demo 的目的是展示算法工程的基本流程，矩阵分解简单易懂，便于理解。实际生产环境会根据场景选择更复杂的模型。

**Q: 如何接入真实数据？**  
A: 将你的数据按照 `data/` 目录下的 CSV 格式整理，或者修改 `data_loader.py` 从数据库/API 读取数据。

**Q: 如何提高推荐效果？**  
A: 可以从多个角度优化：
- 增加更多特征（用户画像、文章内容特征）
- 使用更复杂的模型（深度学习模型）
- 优化负采样策略
- 增加实时特征
- 多路召回 + 精排

**Q: 如何部署到生产环境？**  
A: 生产环境需要考虑：
- 使用 Docker 容器化
- 部署到 Kubernetes
- 集成监控和日志系统
- 使用模型服务框架（如 TensorFlow Serving）
- 增加缓存层（Redis）
- 支持水平扩展

---

## 10. 参考资料

- [推荐系统实践](https://book.douban.com/subject/10769749/) - 项亮
- [深度学习推荐系统](https://book.douban.com/subject/35013197/) - 王喆
- [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) - Netflix 经典论文

---

**祝你学习愉快！** 如果遇到问题，可以查看代码注释或修改代码来理解每个模块的作用。 

