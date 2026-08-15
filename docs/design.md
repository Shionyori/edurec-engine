# edurec-engine 推荐模型设计文档

> **状态**：V1.0（待评审）
> **更新**：2026-08-15
> **接入目标**：edurec-platform（独立微服务，REST 调用，详见 `docs/external/edurec-platform-design.md`）
> **修订记录**：初稿 → 数据策略与架构定稿 → 补充技术选型评估 → 格式润色

---

## 目录

- §1 项目概述
- §2 推荐流水线全景（含省略决策、流水线架构选型）
- §3 工程结构蓝图（含模块化选型）
- §4 数据层（含数据层技术选型）
- §5 特征工程（含特征工程选型）
- §6 召回模型：双塔 DSSM（含召回选型）
- §7 排序模型：多任务 DeepFM（含排序选型）
- §8 重排（含重排选型）
- §9 训练方案
- §10 评估方案
- §11 批量推理与接入
- §12 开发顺序
- §13 决策记录

---

## 1. 项目概述

### 1.1 定位与目标

edurec-engine 是一个教育资源**推荐模型引擎**，作为 edurec-platform 的独立服务被调用，用于平台的首页推荐流场景（`GET /api/v1/recommendations`）。

**一期交付边界**：离线训练 + 批量推理。推荐结果写回平台 `recommendations` 缓存表。
**二期（后置，本期不展开）**：在线 REST 推理服务。

项目**学习导向**：每个阶段结构划分清晰、可独立理解和测试；关键决策以**技术选型评估**形式给出——统一遵循「评估维度 → 候选方案对比 → 决策 → 依据 → 风险与缓解 → 演进」结构。

### 1.2 核心决策（完整决策记录见 §13）

| # | 决策项 | 选择 | 理由 |
|---|--------|------|------|
| 1 | 技术栈 | Python + PyTorch | ML 生态成熟，参考项目同栈 |
| 2 | 模型架构 | 两阶段：召回（双塔 DSSM）+ 排序（多任务 DeepFM） | 工业标准，召回为资源量增长预留扩展性 |
| 3 | 数据策略 | 行为模拟器打通流水线 + MovieLens-1M 真实数据验证 | 平台尚未上线，先模拟后真实，避免纯生成数据失真 |
| 4 | 交付边界 | 离线训练 + 批量推理 | 契合平台 recommendations 缓存表，务实 |
| 5 | 接入方式 | 独立服务，REST 调用；批量写库 | 平台设计既定；在线服务二期 |

### 1.3 平台数据上下文（engine 需兼容的 schema）

平台 MySQL 已有表（与 engine 相关的部分）：

- `users(id, username, display_name, ...)` —— **无人口属性字段**，画像必须从行为反推
- `resources(id, title, type[course|article|video], category_id, tags JSON, metadata JSON, avg_rating, view_count, ...)`
- `categories(id, name)`
- `user_behaviors(id, user_id, resource_id, action[view|click|favorite], created_at)`
- `ratings(id, user_id, resource_id, score 1-5, ...)`
- `recommendations(id, user_id, resource_ids JSON, created_at)` —— 推荐结果缓存表

---

## 2. 推荐流水线全景

```
原始数据 ──► ① 数据层/预处理 ──► ② 特征工程 ──► ③ 召回 ──► ④ 粗筛(规则)
 模拟器 / MovieLens           用户画像+资源特征    双塔DSSM      过滤已看/低质
                                        + 热门兜底
                                                      │
                                                      ▼
 输出 top-N ◄── ⑦ 批量推理 ◄── ⑥ 重排 ◄── ⑤ 精排 ◄───┘
   写回平台    全链路跑活跃用户  多样性/去重/冷启动  多任务DeepFM打分
```

| # | 阶段 | 一句话职责 | 输入 → 输出 | 关键指标 |
|---|------|-----------|------------|---------|
| ① | 数据层/预处理 | 把脏数据变成能训的样本 | 原始记录 → 训练/验证/测试集 | — |
| ② | 特征工程 | 把「用户/资源/上下文」变成数字 | 样本 → 特征向量 | — |
| ③ | 召回 | 从全量资源捞出一两百个候选 | 用户+全量资源 → 候选集 | Recall@K, HitRate@K |
| ④ | 粗筛 | 规则过滤（替代粗排模型） | 候选集 → 精简候选 | — |
| ⑤ | 精排 | 对候选精确打分排序 | 候选+画像特征 → 排序序列 | AUC, GAUC, RMSE |
| ⑥ | 重排 | 多样性/去重/冷启动干预 | 排序序列 → 最终 top-N | 离线可度量多样性 |
| ⑦ | 批量推理 | 所有活跃用户走全链路并写库 | 模型+全用户 → recommendations 表 | — |

### 2.1 有意的省略与替代决策

1. **粗排用规则粗筛代替独立模型**。粗排模型（浅层网络给候选打分压缩规模）只有在候选集巨大（上万）且精排昂贵时才值得。一期资源量小、候选仅一两百，DeepFM 单次打分毫秒级，独立粗排模型是浪费。用「过滤已看/低质量/不合规」的规则粗筛即可，资源量上来后再补。
2. **召回直接全量点积，不建 ANN 索引**。资源量小（几百~几千）时全量计算更简单精确；双塔产出的是 embedding，资源量大后无缝切换 faiss ANN。

### 2.2 技术选型：流水线架构（多阶段漏斗 vs 端到端）

**评估维度**：目标对齐性、算力效率、可调试性

**候选方案评估**：

| 方案 | 目标对齐 | 算力效率 | 可调试性 | 结论 |
|------|---------|---------|---------|------|
| 端到端单模型 | 单一目标，无法兼顾召回/精确/多样性三个冲突目标 | 全量精算，无规模压缩 | 黑盒，难定位问题阶段 | 不采用 |
| 多阶段漏斗（召回→粗筛→精排→重排） | 每级一个目标，各自可优化 | 逐级压缩规模，精算只作用于少量候选 | 每级独立验证、可定位 | ✅ 采用 |

**决策**：采用多阶段漏斗架构。
**依据**：召回率/精确率/多样性是冲突目标，单一模型难以同时优化；漏斗结构把「全量→top-N」的规模压缩前置到便宜操作，精确算力只花在最值得的候选上。
**风险与缓解**：阶段间误差累积——召回漏掉候选，精排再准也无用 → 用 Recall@K 单独把关召回质量。
**演进**：各阶段独立升级（召回切 ANN、补粗排模型），不影响其他阶段。

---

## 3. 工程结构蓝图

```
edurec-engine/
  dataset/                  # 数据落盘（原始 / 处理后 parquet/csv）
  model/                    # 模型工件（checkpoint、编码器、ANN索引、评估报告）
  src/engine/
    config.py               # 统一配置（dataclass + yaml，全流程共享）
    data/
      schema.py             # 统一数据 schema（平台兼容）：User/Resource/Behavior/Rating
      simulator.py          # 行为模拟器：生成平台 schema 兼容样本
      movielens.py          # MovieLens-1M 加载 → 统一 schema
      preprocess.py         # 清洗、负采样、ID 编码、切分
    features/
      feature_config.py     # 声明式特征定义（数值/类目/多值类目）
      user_features.py      # 用户画像特征工程
      item_features.py      # 资源特征工程
    models/
      common/               # 共享基础模块：FM 层、MLP、embedding 工具
      recall/
        dssm.py             # 双塔召回模型
        trainer.py          # 召回训练（负采样 / in-batch softmax）
      rank/
        multitask_deepfm.py # 多任务排序模型
        trainer.py          # 排序训练
    pipeline/
      train_recall.py       # 召回训练入口
      train_rank.py         # 排序训练入口
      evaluate.py           # 离线评估（召回/排序/端到端）
      infer_batch.py        # 批量推理：所有活跃用户 → top-N
    serving/
      api.py                # 二期预留：REST 接口骨架
  scripts/
    gen_sim_data.py         # ① 生成模拟数据
    load_movielens.py       # ② 下载并转换公开集
    train_all.py            # ③ 一键训练（召回+排序）
    run_batch_infer.py      # ④ 批量推理输出
  tests/                    # 每个模块对应单测
  pyproject.toml
  README.md
```

**设计原则：**

- **单向依赖**：`data → features → models → pipeline`，下层不依赖上层，可独立测试
- **配置驱动**：数据源（sim / movielens）、超参、特征开关全走 config，换数据源/模型不改流水线代码
- **可插拔**：召回和排序是独立模块，替换模型只动 `models/` 内部
- **统一 schema 是枢纽**：平台 / 模拟器 / 公开集三路数据全部转成一套 schema，下游只看 schema

### 3.1 技术选型：工程模块化与依赖治理

**评估维度**：独立可测性、变更成本、数据源扩展性

**候选方案评估**：

| 方案 | 独立可测 | 变更成本 | 数据源扩展 | 结论 |
|------|---------|---------|-----------|------|
| 单体混合（数据/特征/模型同层） | 差，改动互相牵连 | 高，一处变更联动多处 | 差，加数据源要改下游所有模块 | 不采用 |
| 分层单向依赖（data→features→models→pipeline）+ 配置驱动 + 统一 schema | 每层可独立单测 | 低，替换只动层内部 | 高，加数据源只加 loader | ✅ 采用 |

**决策**：分层单向依赖 + 配置驱动 + 统一 schema 枢纽。
**依据**：三路数据（模拟/公开/平台）格式迥异，在入口归一后下游永远只写一套代码；配置驱动保证「换数据源/超参」不动流水线；分层单向依赖让每个模块可独立验证。
**风险与缓解**：分层过细产生样板代码 → 以「能独立测试的最小单元」为粒度，不追求极致拆分。

---

## 4. 数据层（阶段 ①）

### 4.1 统一 schema（平台兼容）

```python
# data/schema.py
@dataclass
class User:        user_id: int
@dataclass
class Resource:    resource_id: int
                   title: str
                   type: str                    # course | article | video
                   category_id: int
                   tags: list[str]              # 平台 tags JSON
                   metadata: dict               # 时长/字数等扩展字段
@dataclass
class Behavior:    user_id: int
                   resource_id: int
                   action: str                  # view | click | favorite
                   ts: int                      # 时间戳
@dataclass
class Rating:      user_id: int
                   resource_id: int
                   score: int                   # 1-5
                   ts: int
```

下游（特征/训练）**只看这套 schema**，这是「模拟 → 平台真实数据无缝切换」的关键。

### 4.2 行为模拟器（打通流水线用）

目的：平台尚未上线、无真实数据，先用模拟器生成带**真实结构**（而非纯随机）的数据，验证整条流水线能跑通、模型能收敛。

参数与机制：

- 规模参数：用户数 N、资源数 M、类型/类目/标签分布、行为量级（交互总数）
- **注入隐藏结构**，让数据「可学」：
  - 每个用户：类目偏好向量 `u_cat`、内容口味参数、活跃度
  - 每个资源：质量分 `q_i`、热度 `pop_i`、冷门度
  - 交互概率：`P(行为) ∝ 偏好匹配度(u_cat · cat_i) × q_i × pop_i + 高斯噪声`
- 产出：`user_behaviors`（view/click/favorite 分层）+ `ratings` + 刻意留出**冷启动用户/新资源子集**（验证冷启动策略）
- **刻意不理想化**：加噪声、加偏置（热门更易被交互），避免模拟数据「太干净」掩盖流水线缺陷

### 4.3 MovieLens-1M（真实数据验证）

目的：证明模型在**真实行为数据**上真正收敛、指标合理，避免「只在模拟数据上自嗨」。

- 下载：公开 ML-1M（约 6MB），`ratings.dat` / `movies.dat` / `users.dat`
- 映射规则：
  - 评分 **4-5 分 → 正向交互**（`click`，隐式信号）
  - 原始 1-5 分保留 → **评分任务监督**
  - `genres` → 类目/标签
  - 时间戳保留 → 时间切分可用

### 4.4 预处理（preprocess.py）

| 步骤 | 内容 | 学习点 |
|------|------|--------|
| 清洗 | 过滤交互数 <5 的用户/资源、去重、删异常高频行为 | 噪声对训练的影响 |
| 负采样 | 召回训练：in-batch 随机负样本 + 热门负采样混合 | 负样本是召回的关键 |
| ID 编码 | 用户/资源/类目/标签 → 连续索引，vocab 落盘复用 | 编码与模型解耦 |
| 切分 | 按用户**时间序**切 80/10/10，防泄漏 | 时序泄漏是常见坑 |
| 样本格式 | 排序：正样本(行为/评分)+负样本；召回：交互对 | 两阶段消费不同数据形态 |

### 4.5 数据层技术选型

#### 4.5.1 数据接入方案（统一 schema 抽象 vs 各源直接适配）

**评估维度**：下游复用性、数据源扩展性、特征口径一致性

| 方案 | 复用性 | 扩展性 | 口径一致性 | 结论 |
|------|--------|--------|-----------|------|
| 各源直接适配（下游分别处理） | 差，每加一个源写一套下游 | 差，改所有下游 | 特征口径难统一 | 不采用 |
| 统一 schema 抽象（入口归一） | 下游一套代码 | 加源只加 loader | 入口统一口径 | ✅ 采用 |

**决策**：统一 schema 抽象。
**依据**：模拟器/公开集/平台三路格式迥异，入口归一后特征、训练、推理只写一份代码，是「模拟→真实无缝切换」的前提。
**风险与缓解**：schema 过度抽象丢失各源特有信息 → 用 `metadata: dict` 兜底扩展字段。

#### 4.5.2 模拟器生成策略（隐藏结构 vs 纯随机 vs 真实采样）

**评估维度**：可学性（是否含可学习结构）、失真度（是否过度理想）、数据可得性

| 方案 | 可学性 | 失真度 | 可得性 | 结论 |
|------|--------|--------|--------|------|
| 纯随机生成 | 无可学结构，流水线验证无意义 | 失真 | 无需外部数据 | 不采用 |
| 隐藏结构 + 噪声（偏好×质量×热度 + 高斯噪声） | 可学，能验证收敛 | 带噪声，不过度理想 | 无需外部数据 | ✅ 采用 |
| 真实分布采样 | 高 | 低 | 一期无真实数据 | 不采用（二期） |

**决策**：注入隐藏结构 + 加噪声。
**依据**：生成数据需「可学」以验证流水线正确性，但过度理想会掩盖缺陷；结构+噪声是两者的平衡点。
**风险与缓解**：模拟分布与真实分布存在偏差 → 模拟数据仅承担「流水线正确性」验证，「有效性」由公开集（§4.5.3）负责。
**演进**：二期有真实数据后，模拟器降级为回归测试数据源。

#### 4.5.3 公开数据集选型（MovieLens-1M vs 教育域数据集）

**评估维度**：信号完备性（隐式交互/显式评分/metadata）、规模与算力需求、社区资料与排错成本、领域契合度

| 方案 | 信号完备 | 规模/算力 | 资料/排错 | 领域契合 | 结论 |
|------|---------|-----------|-----------|---------|------|
| MovieLens-1M | 隐式+显式+metadata 齐全，对应平台三套信号 | 100 万评分，CPU 可跑 | 资料最全，易排错 | 非教育域 | ✅ 采用 |
| EdNet / XuetangX | 教育域真实行为 | 规模大（EdNet 7000 万+），预处理成本高 | 资料少 | 契合 | 备选（二期） |

**决策**：采用 MovieLens-1M。
**依据**：信号完备性恰好覆盖平台三套信号（行为/评分/metadata）；规模适中可 CPU 训练；社区资料全、问题易排查。领域差异由统一 schema 吸收——换数据集只影响 loader，不影响下游。
**演进**：二期引入教育域数据集做领域内进阶验证。

#### 4.5.4 负采样策略（in-batch vs 热门负采样 vs 混合）

**评估维度**：构造开销、抗热门偏置、训练稳定性

| 方案 | 构造开销 | 抗热门偏置 | 稳定性 | 结论 |
|------|---------|-----------|--------|------|
| 仅 in-batch 随机 | 零开销 | 弱 | 稳定基线 | 基线 |
| 仅热门负采样 | 需构建热门集 | 强 | 可能过抑制长尾 | 可选 |
| in-batch 随机 + 热门负采样 | 略增 | 均衡 | 稳 | ✅ 采用 |

**决策**：in-batch 随机为主 + 热门负采样补充。
**依据**：行为日志只有正样本，无负样本则模型无法区分「不喜欢」和「没见过」，会退化为全给高分；in-batch 零构造开销，热门负采样抑制「全推热门」偏置。
**风险与缓解**：负样本过严压低召回 → 用验证集 Recall@K 调整负采样比例。

---

## 5. 特征工程（阶段 ②）

### 5.1 声明式特征配置（feature_config.py）

特征**声明式定义**，模型只消费配置、不关心特征怎么算。这是「结构清晰」的落点：

```python
# features/feature_config.py
@dataclass
class FeatureSpec:
    name: str
    kind: "numeric" | "categorical" | "multi_hot"
    input_dim: int     # categorical=类别数 / multi_hot=词典大小
    embed_dim: int     # embedding 维度（numeric 忽略）
    used_by: "recall" | "rank" | "both"   # 喂给哪个模型

USER_FEATURES = [
    FeatureSpec("user_id",      "categorical", n_users, 64),
    FeatureSpec("active_days",  "numeric",     1),
    FeatureSpec("n_views",      "numeric",     1),
    FeatureSpec("n_favs",       "numeric",     1),
    FeatureSpec("gap_days",     "numeric",     1),           # 距上次行为天数
    FeatureSpec("cat_interest", "multi_hot",   n_cats, 8),  # 类目兴趣向量
    FeatureSpec("tag_interest", "multi_hot",   n_tags, 8),  # 标签兴趣向量
    FeatureSpec("type_pref",    "multi_hot",   3,      4),  # course/article/video
]

ITEM_FEATURES = [
    FeatureSpec("resource_id", "categorical", n_res,  64),
    FeatureSpec("category_id", "categorical", n_cats, 8),
    FeatureSpec("tags",        "multi_hot",   n_tags, 8),
    FeatureSpec("type",        "categorical", 3,      4),
    FeatureSpec("log_view",    "numeric",     1),           # log(热度)
    FeatureSpec("avg_rating",  "numeric",     1),
    FeatureSpec("age_days",    "numeric",     1),           # 上线天数（时效）
]
```

### 5.2 用户画像特征（user_features.py）

平台 `users` 表无人口属性 → 画像全部从**训练集行为统计反推**：

| 特征 | 计算方式 | 信号 |
|------|---------|------|
| `cat_interest` | 用户各**类目**交互次数 → 归一化概率向量 | 类目兴趣 |
| `tag_interest` | 各**标签**交互次数归一化 | 细粒度兴趣 |
| `type_pref` | 三种类型交互占比（3维） | 内容口味 |
| `active_days` | 活跃天数 | 粘性 |
| `n_views` / `n_favs` | 各行为次数 | 活跃度 |
| `gap_days` | 距最近一次行为间隔天数 | 召回意愿/时效 |

**防泄漏规则（学习重点）**：画像统计**只能从训练集计算**；验证/测试集的特征用训练集统计结果。否则评估指标虚高。

### 5.3 资源特征（item_features.py）

类目 / 标签 / 类型 + 三个冷启动友好信号：`log_view`（热度）、`avg_rating`（质量）、`age_days`（时效，给新资源曝光机会）。

### 5.4 上下文特征

请求时刻：`hour_of_day`、`day_of_week`；用户 `gap_days`。一期在排序模型使用。

### 5.5 特征工程设计与选型

#### 5.5.1 特征定义方式（声明式 FeatureSpec vs 硬编码）

**评估维度**：模型/特征解耦、增改成本、跨模型复用

| 方案 | 解耦 | 增改成本 | 复用 | 结论 |
|------|------|---------|------|------|
| 硬编码在模型内 | 特征与模型耦合 | 加特征改模型代码 | 换模型重写 | 不采用 |
| 声明式 FeatureSpec 配置 | 模型只消费配置 | 加特征加一行 | 召回/排序共享 | ✅ 采用 |

**决策**：声明式定义。
**依据**：换模型（如升级 ESMM）不动特征代码；`used_by` 显式标注特征归属，避免召回/排序特征混用。

#### 5.5.2 用户画像特征来源（行为反推）

**约束**：平台 `users` 表无人口属性字段（§1.3），画像特征的**唯一可得来源**是行为统计，因此不做多方案选型，直接采用行为反推。

**依据**：六项特征覆盖四个维度——**兴趣**（cat/tag/type）、**活跃**（active_days/n_views/n_favs）、**时效**（gap_days）。

#### 5.5.3 多值类目表示（加权 embedding vs argmax）

**评估维度**：信息量保留、平滑度

| 方案 | 信息量 | 平滑度 | 结论 |
|------|--------|--------|------|
| argmax（仅取最可能类目） | 丢弃次要兴趣 | 硬切换 | 不采用 |
| multi_hot 概率向量 × embedding 矩阵 | 保留全部兴趣强度 | 平滑加权 | ✅ 采用 |

**决策**：加权求和 embedding。
**依据**：类目兴趣是「多个类目同时可能感兴趣」的概率分布，`兴趣向量 × embedding矩阵` 按强度加权求和，信息量大于 argmax。

#### 5.5.4 防泄漏规则（正确性约束）

**性质**：硬性约束，非可选项。画像统计**只能从训练集计算**，验证/测试集复用训练集统计结果。若用全量数据统计，等于把「测试答案」泄漏给模型，评估指标虚高、上线即翻车。这是离线评估可信度的底线。

---

## 6. 召回模型：双塔 DSSM（阶段 ③）

### 6.1 网络结构

```
   用户塔 (User Tower)                  资源塔 (Item Tower)
 ┌──────────────────────┐          ┌──────────────────────┐
 │ user_id ──► emb(64)  │          │ resource_id ─► emb(64)│
 │ cat_interest ─► emb  │          │ category ──► emb(8)   │
 │ tag_interest ─► emb  │          │ tags ──► mean-emb     │
 │ type_pref ──► emb    │          │ type ────► emb(4)     │
 │        concat        │          │        concat         │
 │   MLP(→128→64)       │          │   MLP(→128→64)        │
 │        ▼             │          │        ▼              │
 │  user_emb (64维)     │          │  item_emb (64维)      │
 └──────────────────────┘          └──────────────────────┘
        │                              │
        └────────── 相似度 = 内积 ◄─────┘
```

### 6.2 训练目标

- **正样本**：有交互的行为（view/click/favorite 均视为正）
- **负样本**：**in-batch 随机负采样**——一个 batch 内其他成员对应的 item 即负样本，配 temperature `τ`
- **损失**：InfoNCE / in-batch softmax

```
L = -log( exp(score(u, i⁺)/τ) / Σⱼ exp(score(u, iⱼ)/τ) )   # j 含正样本 + batch 内全部负样本
```

- 可选补充：**热门负采样**——「很热但用户没交互」的资源作显式负样本，抑制「全推热门」的偏置

### 6.3 双塔结构要点（学习点）

- 用户塔/资源塔**独立编码** → 资源 embedding 可**离线预计算**，推理只剩用户塔在线，快
- 内积/余弦相似度 → 适配 ANN 检索，资源量大后直接挂 faiss

#### 6.3.1 技术选型：召回模型结构（双塔 vs SVD vs Item-CF）

**评估维度**：特征融合能力、冷启动、检索可扩展性

| 方案 | 特征融合 | 冷启动 | 检索扩展 | 结论 |
|------|---------|--------|---------|------|
| 矩阵分解（SVD） | 无法融入用户/资源特征 | 冷启动用户/物品无 latent 向量 | 需在线矩阵计算 | 不采用 |
| Item-CF | 无，纯物品相似度 | 无历史用户无法服务 | 相似度表，规模受限 | 不采用 |
| 双塔（DSSM） | 特征可进塔 | 画像特征可兜底 | 用户/资源独立编码，资源 embedding 离线预计算，直连 ANN | ✅ 采用 |

**决策**：双塔结构。
**依据**：用户/资源独立编码使资源 embedding 可离线预计算，在线只剩用户塔；产出向量直接对接 ANN 检索。

#### 6.3.2 技术选型：召回负样本获取（in-batch softmax vs 显式负采样）

**评估维度**：构造开销、效果稳定性、无偏性

| 方案 | 构造开销 | 稳定性 | 无偏性 | 结论 |
|------|---------|--------|--------|------|
| 显式负采样（独立构造负样本对） | 高，需额外采样逻辑 | 可控 | 有偏（依赖采样分布） | 备选 |
| in-batch softmax | 零开销，batch 内互为负样本 | 稳定，工业默认 | 有偏（batch 内热度不均） | ✅ 一期采用 |
| sampled softmax（采样校正） | 高 | 稳 | 近似无偏 | 二期 |

**决策**：in-batch softmax。
**依据**：正样本对确定，负样本直接取 batch 内其他 item，零构造开销、实现简单、效果稳定。

### 6.4 代码骨架

```python
# models/recall/dssm.py
class TwoTower(nn.Module):
    def __init__(self, n_users, n_res, n_cats, n_tags):
        super().__init__()
        self.user_id_emb = nn.Embedding(n_users, 64)
        self.cat_emb  = nn.Embedding(n_cats, 8)
        self.tag_emb  = nn.Embedding(n_tags, 8)
        self.type_emb = nn.Embedding(3, 4)
        self.item_id_emb = nn.Embedding(n_res, 64)
        self.user_mlp = nn.Sequential(nn.Linear(64 + 8 + 8 + 4, 128), nn.ReLU(),
                                      nn.Linear(128, 64))
        self.item_mlp = nn.Sequential(nn.Linear(64 + 8 + 8 + 4, 128), nn.ReLU(),
                                      nn.Linear(128, 64))

    def user_emb(self, u):               # u: dict of user features (tensor)
        x = torch.cat([
            self.user_id_emb(u["user_id"]),
            u["cat_interest"] @ self.cat_emb.weight,   # multi_hot 加权求和
            u["tag_interest"] @ self.tag_emb.weight,
            u["type_pref"] @ self.type_emb.weight,     # type_pref: multi_hot(3, embed 4)
        ], dim=-1)
        return self.user_mlp(x)

    def item_emb(self, i):               # 结构同 user_emb
        ...
        return self.item_mlp(x)

    def score(self, u, i):
        return (self.user_emb(u) * self.item_emb(i)).sum(-1)   # 内积

    def forward(self, users, pos_items, neg_items):
        # batch 内所有 item 的 embedding 拼成矩阵，算 InfoNCE loss
        ...
```

---

## 7. 排序模型：多任务 DeepFM（阶段 ⑤）

### 7.1 网络结构

```
         输入特征（稀疏 categorical/multi_hot + 数值）
                       │
        ┌──────────────┼─────────────────────┐
        ▼              ▼                     ▼
   一阶线性(embedding和)  二阶FM交互(内积对)    数值特征 concat
        │              │                     │
        └──► FM 部分 ◄──┘                     │
                 │                           │
              concat ────────────► Deep 部分 (MLP 128→64)
                 │                           │
                 └──────► Shared Bottom (拼接) ◄────┘
                                  │
                 ┌────────────────┼────────────────┐
                 ▼                ▼                ▼
            CTR 任务头         CVR 任务头       Rating 任务头
          (点击/浏览概率)      (收藏概率)       (1-5分回归)
```

采用**共享底层（shared-bottom）+ 独立任务头**的最简多任务结构；更进阶的级联建模（如 ESMM）留作扩展。

### 7.2 任务定义（对应平台行为数据）

| 任务头 | 监督信号 | 损失 | 样本 |
|--------|---------|------|------|
| CTR | 是否 view/click | BCE | 全部样本（正+负） |
| CVR | 是否 favorite | BCE | 全部样本 |
| Rating | 评分 1-5 | MSE | **仅打过分的样本** |

```
L_total = w₁·BCE(ctr) + w₂·BCE(cvr) + w₃·MSE(rating)     # w 可配置
```

### 7.3 FM 二阶交互（DeepFM 核心）

不用做特征两两组合，FM 用隐向量技巧把二阶交互算成 O(k)：

```
二阶交互项 = 0.5 × ( ||Σᵢ vᵢ||²  −  Σᵢ ||vᵢ||² )
```

每个特征一个 embedding `vᵢ`，两两交互 ≈ 隐向量内积。Deep 部分再补非线性高阶交互。

### 7.4 最终排序分（衔接重排）

```
final_score = α·pCTR + β·pCVR + γ·(pred_rating/5)     # α+β+γ=1，可配置
```

重排阶段再叠加：MMR 多样性、去重、新资源加权，取 top-N。

### 7.5 代码骨架

```python
# models/common/fm.py
class FM(nn.Module):
    def forward(self, feat_embs):            # feat_embs: 各特征 embedding 列表
        first  = sum(e.sum(1) for e in feat_embs)        # 一阶
        sum_sq = sum(e.sum(1) ** 2 for e in feat_embs)
        sq_sum = sum((e ** 2).sum(1) for e in feat_embs)
        second = 0.5 * (sum_sq - sq_sum)                 # 二阶
        return first + second

# models/rank/multitask_deepfm.py
class MultiTaskDeepFM(nn.Module):
    def __init__(self, feature_spec):
        super().__init__()
        self.embeddings = FeatureEmbedding(feature_spec)   # 所有稀疏特征 embedding
        self.fm   = FM()
        self.deep = nn.Sequential(nn.Linear(emb_dim_total, 128), nn.ReLU(),
                                  nn.Linear(128, 64))
        self.ctr_head    = nn.Linear(64, 1)     # + sigmoid
        self.cvr_head    = nn.Linear(64, 1)
        self.rating_head = nn.Linear(64, 1)     # 回归

    def forward(self, features):
        embs = self.embeddings(features)
        shared = torch.cat([self.fm(embs), self.deep(concat_all(embs))], -1)
        pctr   = torch.sigmoid(self.ctr_head(shared))
        pcvr   = torch.sigmoid(self.cvr_head(shared))
        rating = self.rating_head(shared)
        return pctr, pcvr, rating
```

### 7.6 排序模型技术选型

#### 7.6.1 精排模型结构（DeepFM vs LR/FM/DNN）

**评估维度**：特征交互建模能力、稀疏高维收敛性

| 方案 | 二阶交互 | 高阶交互 | 稀疏高维收敛 | 结论 |
|------|---------|---------|-------------|------|
| LR | 无 | 无 | 好 | 不采用 |
| FM | 显式 | 无 | 好 | 不采用（缺高阶） |
| DNN | 隐式 | 隐式 | 稀疏下难收敛 | 不采用 |
| DeepFM | 显式（FM 部分） | 显式（Deep 部分） | 好 | ✅ 采用 |

**决策**：DeepFM。
**依据**：显式二阶（FM）与隐式高阶（Deep）互补，兼顾交互建模与稀疏收敛，是 CTR 排序的主流 baseline。

#### 7.6.2 任务组织（多任务共享底层 vs 独立多模型）

**评估维度**：信号相关性利用、推理成本

| 方案 | 相关性利用 | 推理成本 | 结论 |
|------|-----------|---------|------|
| 三个独立模型 | 无共享 | 3 次前向 | 不采用 |
| 多任务共享底层 | 表示共享 + 数据增强 | 1 次前向 | ✅ 采用 |

**决策**：多任务共享底层。
**依据**：平台点击/收藏/评分三信号高度相关；一次前向同时出三任务分；三任务对应平台三张行为表，是标准多任务场景。

#### 7.6.3 多任务结构（shared-bottom vs ESMM）

**评估维度**：目标匹配度、复杂度、一期必要性

| 方案 | 目标匹配 | 复杂度 | 结论 |
|------|---------|--------|------|
| ESMM 级联 | 为「曝光→转化漏斗」设计，与本期无漏斗场景不符 | 高 | 二期 |
| shared-bottom + 独立任务头 | 简单、可解释、够用 | 低 | ✅ 一期 |

**决策**：shared-bottom。
**依据**：一期以「可解释、结构清晰、先跑通」为目标，级联建模收益有限。
**演进**：出现转化漏斗场景时升级 ESMM。

#### 7.6.4 任务样本策略（稠密全样本 vs 稀疏有监督样本）

**策略**：CTR/CVR 用全部样本，Rating 仅用评分样本。
**依据**：行为是稠密信号（谁都有），评分是稀疏信号（仅打过分的样本有监督）。稠密任务在全部样本学习、稀疏任务在有监督样本学习，损失按任务加权重，是处理任务样本量不均的标准做法。

---

## 8. 重排（阶段 ⑥，规则实现，不训练模型）

| 规则 | 作用 |
|------|------|
| 过滤已收藏/已看 | 去重，避免重复推荐 |
| **MMR（最大边际相关）** | 多样性：候选按 `(精排分 − λ·与已选集合的相似度)` 贪心挑选，避免 Top10 全是一个类目/作者 |
| 新资源加权 | 冷启动：给 `age_days` 小的资源加权重，保证新内容曝光 |
| 截取 top-N | 最终输出 N 个（如 20） |

### 8.1 技术选型：重排实现（规则 vs 训练模型）

**评估维度**：问题本质匹配度、可解释性、训练数据需求

| 方案 | 问题匹配 | 可解释性 | 数据需求 | 结论 |
|------|---------|---------|---------|------|
| 训练重排模型（如 ListRank） | 重排目标（多样性/去重/曝光）本质是**约束满足**而非预测 | 低 | 需大量曝光序列数据，一期不具备 | 不采用 |
| 规则（MMR + 去重 + 新资源加权） | 约束满足可直接求解 | 高 | 无 | ✅ 采用 |

**决策**：规则实现。
**依据**：一期把「规则如何影响体验」讲清楚，等数据积累后再训练重排模型才有意义。

---

## 9. 训练方案

| 阶段 | 数据 | 超参（初始） | 早停指标 |
|------|------|-------------|---------|
| 召回训练 | 行为正样本 + in-batch 负样本 | Adam, lr=1e-3, batch=1024, τ=0.05 | val Recall@50 |
| 排序训练 | 曝光样本（CTR 正负）+ 评分样本 | Adam, lr=1e-3, batch=256 | val AUC / RMSE |

- 训练流程：`scripts/train_all.py` = 数据准备 → 召回训练 → 排序训练 → 评估
- 结果工件落盘：`model/`（checkpoint、特征编码器、vocab、评估报告）
- 设备：CPU 可跑通（数据量小）；有 GPU 则自动使用
- 可复现：固定随机种子，config 落盘

### 9.1 训练策略设计

**1. 两阶段独立训练 vs 联合训练 → 独立训练。**
召回/排序目标不同（召回最大化命中、排序最大化排序质量），联合优化互相干扰；工业界同样独立训练。

**2. 训练顺序：召回先于排序（数据依赖决定）。**
排序模型需要「曝光未点」负样本与正样本，候选集由召回产出——召回先行才有数据喂排序。这是流水线依赖，不是偏好。

**3. 双轨验证：模拟轨道证明正确性，公开轨道证明有效性。**

| 轨道 | 验证目标 | 不足以证明 |
|------|---------|-----------|
| 模拟数据 | 流水线正确（能跑通、能收敛） | 模型有效（可能只是模拟分布自洽） |
| MovieLens | 模型有效（真实数据指标合理） | 接入平台的流水线正确性 |

两者互补，缺一不可。

**4. 超参初始值：起点而非定论。**
Adam + lr=1e-3 + 小 batch 是深度 CTR 模型的稳妥起点；τ=0.05 为双塔常用温度。全部走 config 可调，后续用验证集早停/网格调优。

---

## 10. 评估方案

| 层级 | 指标 | 说明 |
|------|------|------|
| 召回 | Recall@K, HitRate@K, NDCG@K | 测试集上每个用户取交互过的资源算命中 |
| 排序 | AUC（CTR/CVR）, GAUC, RMSE（评分） | 任务头分别评估 |
| 端到端 | 模拟「用户→推荐→用户点没点」的 NDCG@K | 完整流水线效果 |

切分：按用户时间序 80/10/10；另设**冷启动验证子集**（新用户/新资源）单独评估。

### 10.1 评估指标选型

**评估维度**：能否回答该阶段的核心问题、与推荐目标对齐

| 指标 | 回答的问题 | 对应目标 | 用在 |
|------|-----------|---------|------|
| Recall@K / HitRate@K | 用户感兴趣的资源捞出来多少？ | 召回率 | 召回 |
| NDCG@K | 排序质量（位置加权命中）？ | 排序质量 | 召回/端到端 |
| AUC / GAUC | 打分能否区分点/不点？ | 区分度 | 排序 |
| RMSE | 评分预测误差？ | 预测精度 | 评分任务 |

### 10.2 数据切分策略（时间序 vs 随机）

**评估维度**：贴近真实部署、泄漏风险

| 方案 | 泄漏风险 | 部署拟合 | 结论 |
|------|---------|---------|------|
| 随机切分 | 未来行为混入训练，指标虚高 | 低 | 不采用 |
| 用户时间序切分（历史→训练，近期→验证，最新→测试） | 无时间泄漏 | 高，模拟真实「用历史推未来」 | ✅ 采用 |

### 10.3 冷启动子集

**策略**：单设新用户/新资源子集单独评估。
**依据**：冷启动是推荐核心难题，单独评估可暴露「无历史用户」的表现，避免被老用户高指标掩盖。

---

## 11. 批量推理与接入（阶段 ⑦，接入方案二期细化）

### 11.1 批量推理（一期）

```
加载训练好的模型（召回 + 排序）
  → 遍历所有活跃用户：
      ③ 召回候选（全量点积 top-K）
      ④ 规则粗筛
      ⑤ 精排打分
      ⑥ 重排 → top-N
  → 输出 recommendations（user_id, resource_ids）
```

### 11.2 与平台接入（二期，概述）

- engine 作为**独立微服务**，平台后端通过 REST 调用（平台侧已定义 engine client 接口）
- 候选：批量推理结果写入平台 `recommendations` 缓存表（Redis 缓存推荐结果），平台首页直接读
- 预留：`serving/api.py` 提供 REST 骨架（如 `GET /recommend`、`POST /feedback` 供二期扩展）
- 数据回流：平台行为数据 → engine 训练集（二期定义同步机制）

---

## 12. 开发顺序

1. **数据层**：schema → 模拟器 → 预处理（可独立测试，先跑通数据流水线）
2. **特征层**：画像特征 → 资源特征（配单测验证数值正确 + 防泄漏）
3. **召回**：双塔 DSSM → 训练 → 评估 Recall@K
4. **排序**：多任务 DeepFM → 训练 → 评估 AUC/RMSE
5. **全链路**：infer_batch → 重排 → 输出 top-N
6. **公开集验证**：MovieLens 轨道跑通，确认指标合理
7. **二期**：REST 服务化 + 平台接入

## 13. 决策记录

| # | 决策项 | 选择 |
|---|--------|------|
| 1 | 技术栈 | Python + PyTorch |
| 2 | 模型架构 | 两阶段：双塔 DSSM 召回 + 多任务 DeepFM 排序 |
| 3 | 数据策略 | 模拟器打通流水线 + MovieLens-1M 真实验证 |
| 4 | 交付边界 | 离线训练 + 批量推理 |
| 5 | 接入方式 | 独立服务 REST；批量写 recommendations 缓存表 |
| 6 | 多任务结构 | shared-bottom + 独立任务头（ESMM 留作扩展） |
| 7 | 粗排 | 一期用规则粗筛，不训练独立粗排模型 |
| 8 | 召回检索 | 一期全量点积，不建 ANN（资源量大后切 faiss） |
| 9 | 重排 | 规则实现（MMR + 去重 + 新资源加权），不训练重排模型 |
| 10 | 排序分融合 | final_score = α·pCTR + β·pCVR + γ·(rating/5) |
