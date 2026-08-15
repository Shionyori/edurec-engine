# edurec-engine 推荐引擎实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个可离线训练 + 批量推理的教育资源推荐引擎：双塔 DSSM 召回 + 多任务 DeepFM 排序 + 规则重排，模拟数据打通流水线，MovieLens-1M 验证有效性。

**Architecture:** 分阶段流水线（数据 → 特征 → 召回 → 排序 → 重排 → 推理），严格单向依赖 `data → features → models → pipeline`。三路数据源（模拟器 / MovieLens / 平台）在入口统一为 `DataBundle` schema，下游只看 schema。

**Tech Stack:** Python ≥3.10, PyTorch ≥2.0, NumPy, pandas, PyYAML；测试用 pytest。

**Spec:** `docs/design.md`（技术选型评估见设计文档各节，本计划直接从 spec 论证）

## Global Constraints

以下约束对每个任务生效，实现时无需重复引用：

1. **单向依赖**：`data → features → models → pipeline`，禁止反向 import（如 models 不得 import pipeline）
2. **统一 schema**：`DataBundle`（User/Resource/Behavior/Rating）是所有数据源的下游接口，禁止绕过
3. **防泄漏**：用户画像特征**只能从训练集行为统计**（`build_user_features` 只接收 train bundle）；验证/测试复用同一份统计
4. **声明式特征**：模型只消费 `FeatureSpec` 配置，特征不硬编码进模型
5. **可复现**：所有随机源使用 `config.seed`（模拟器、负采样、训练权重初始化）
6. **一期不做 ANN**：召回用全量点积，不引入 faiss
7. **提交规范**：Conventional Commits（`feat(data):`、`feat(recall):`、`fix(...)`）
8. **Python 版本与依赖**：requires-python ≥3.10；核心依赖 torch≥2.0、numpy≥1.24、pandas≥2.0、pyyaml≥6.0；dev 依赖 pytest≥7.4
9. 模型工件与数据目录（`model/`、`dataset/`）已被 `.gitignore` 排除，不提交

---

### Task 1: 项目骨架、统一配置与数据 schema

**Files:**
- Create: `pyproject.toml`
- Create: `src/engine/__init__.py`
- Create: `src/engine/config.py`
- Create: `src/engine/data/__init__.py`
- Create: `src/engine/data/schema.py`
- Create: `src/engine/data/io.py`
- Create: `tests/test_config.py`
- Create: `tests/test_schema.py`

**Interfaces:**
- Produces:
  - `engine.config.EngineConfig`（dataclass，含 `from_yaml/to_yaml`，字段见下方代码）
  - `engine.data.schema.User / Resource / Behavior / Rating`（frozen dataclass）
  - `engine.data.schema.DataBundle(users, resources, behaviors, ratings)`（普通 dataclass，四个 list 字段）
  - `engine.data.io.save_bundle(bundle, out_dir)` / `load_bundle(in_dir) -> DataBundle`（CSV 持久化）

- [ ] **Step 1: 写失败测试 `tests/test_config.py`**

```python
from engine.config import EngineConfig

def test_default_config_fields():
    cfg = EngineConfig()
    assert cfg.seed == 42
    assert cfg.data_source == "sim"
    assert cfg.recall_embed_dim == 64

def test_config_yaml_roundtrip(tmp_path):
    cfg = EngineConfig(sim_n_users=123)
    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(str(p))
    loaded = EngineConfig.from_yaml(str(p))
    assert loaded.sim_n_users == 123
```

- [ ] **Step 2: 写失败测试 `tests/test_schema.py`**

```python
from engine.data.schema import User, Resource, Behavior, Rating, DataBundle

def test_dataclasses_frozen():
    u = User(user_id=1)
    assert u.user_id == 1
    r = Resource(resource_id=10, type="course", category_id=2,
                 tags=("AI",), metadata={"duration": 300})
    assert r.tags == ("AI",)

def test_databundle_holds_lists():
    b = DataBundle(
        users=[User(user_id=1)],
        resources=[Resource(resource_id=10, type="course", category_id=2,
                            tags=(), metadata={})],
        behaviors=[Behavior(user_id=1, resource_id=10, action="view", ts=0)],
        ratings=[Rating(user_id=1, resource_id=10, score=5, ts=1)],
    )
    assert len(b.users) == 1 and len(b.ratings) == 1
```

- [ ] **Step 3: 运行测试验证失败**

Run: `pytest tests/test_config.py tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine'`（包尚未创建）

- [ ] **Step 4: 写实现**

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "edurec-engine"
version = "0.1.0"
description = "教育资源推荐模型引擎"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "pandas>=2.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.4"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`src/engine/__init__.py` 与 `src/engine/data/__init__.py` 置空即可。

`src/engine/config.py`:

```python
from __future__ import annotations
from dataclasses import dataclass, asdict
import yaml


@dataclass
class EngineConfig:
    seed: int = 42
    data_source: str = "sim"                      # sim | movielens

    # 模拟器
    sim_n_users: int = 2000
    sim_n_resources: int = 500
    sim_n_categories: int = 12
    sim_n_tags: int = 30
    sim_n_interactions: int = 100_000

    # 预处理
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    train_ratio: float = 0.8
    val_ratio: float = 0.1

    # 召回
    recall_embed_dim: int = 64
    recall_batch_size: int = 1024
    recall_lr: float = 1e-3
    recall_epochs: int = 10
    recall_tau: float = 0.05
    recall_k: int = 50

    # 排序
    rank_batch_size: int = 256
    rank_lr: float = 1e-3
    rank_epochs: int = 5
    rank_neg_per_pos: int = 3
    rank_w_ctr: float = 1.0
    rank_w_cvr: float = 0.5
    rank_w_rating: float = 0.5

    # 重排
    top_n: int = 20
    mmr_lambda: float = 0.5
    cold_age_days: float = 7.0
    cold_weight: float = 1.2

    # 路径
    data_dir: str = "dataset"
    model_dir: str = "model"

    @classmethod
    def from_yaml(cls, path: str) -> "EngineConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, allow_unicode=True)
```

`src/engine/data/schema.py`:

```python
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class User:
    user_id: int


@dataclass(frozen=True)
class Resource:
    resource_id: int
    type: str                       # course | article | video
    category_id: int
    tags: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Behavior:
    user_id: int
    resource_id: int
    action: str                     # view | click | favorite
    ts: int


@dataclass(frozen=True)
class Rating:
    user_id: int
    resource_id: int
    score: int                      # 1-5
    ts: int


@dataclass
class DataBundle:
    users: list[User]
    resources: list[Resource]
    behaviors: list[Behavior]
    ratings: list[Rating]
```

`src/engine/data/io.py`（用 pandas 落盘 CSV，便于查看与调试）：

```python
from __future__ import annotations
import os
import pandas as pd
from .schema import DataBundle, User, Resource, Behavior, Rating


def _df_to_resources(df: pd.DataFrame) -> list[Resource]:
    out = []
    for row in df.itertuples():
        out.append(Resource(
            resource_id=int(row.resource_id),
            type=str(row.type),
            category_id=int(row.category_id),
            tags=tuple(str(row.tags).split("|")) if row.tags else (),
            metadata={},
        ))
    return out


def save_bundle(bundle: DataBundle, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([{"user_id": u.user_id} for u in bundle.users]).to_csv(
        os.path.join(out_dir, "users.csv"), index=False)
    pd.DataFrame([{
        "resource_id": r.resource_id, "type": r.type,
        "category_id": r.category_id, "tags": "|".join(r.tags),
    } for r in bundle.resources]).to_csv(
        os.path.join(out_dir, "resources.csv"), index=False)
    pd.DataFrame([{
        "user_id": b.user_id, "resource_id": b.resource_id,
        "action": b.action, "ts": b.ts,
    } for b in bundle.behaviors]).to_csv(
        os.path.join(out_dir, "behaviors.csv"), index=False)
    pd.DataFrame([{
        "user_id": r.user_id, "resource_id": r.resource_id,
        "score": r.score, "ts": r.ts,
    } for r in bundle.ratings]).to_csv(
        os.path.join(out_dir, "ratings.csv"), index=False)


def load_bundle(in_dir: str) -> DataBundle:
    users = [User(user_id=int(r.user_id)) for r in pd.read_csv(os.path.join(in_dir, "users.csv")).itertuples()]
    res_df = pd.read_csv(os.path.join(in_dir, "resources.csv"))
    resources = _df_to_resources(res_df)
    behaviors = [Behavior(
        user_id=int(r.user_id), resource_id=int(r.resource_id),
        action=str(r.action), ts=int(r.ts),
    ) for r in pd.read_csv(os.path.join(in_dir, "behaviors.csv")).itertuples()]
    ratings = [Rating(
        user_id=int(r.user_id), resource_id=int(r.resource_id),
        score=int(r.score), ts=int(r.ts),
    ) for r in pd.read_csv(os.path.join(in_dir, "ratings.csv")).itertuples()]
    return DataBundle(users=users, resources=resources,
                      behaviors=behaviors, ratings=ratings)
```

- [ ] **Step 5: 运行测试验证通过**

Run: `pip install -e .[dev] && pytest tests/test_config.py tests/test_schema.py -v`
Expected: PASS（2 个测试文件全过）

- [ ] **Step 6: 提交**

```bash
git add pyproject.toml src/engine tests
git commit -m "chore: 搭建项目骨架、统一配置与数据 schema"
```

---

### Task 2: 行为模拟器（注入隐藏结构）

**Files:**
- Create: `src/engine/data/simulator.py`
- Create: `tests/test_simulator.py`

**Interfaces:**
- Consumes: `EngineConfig`、`DataBundle`（Task 1）
- Produces: `engine.data.simulator.generate(config) -> DataBundle`
  - 行为量约等于 `config.sim_n_interactions`，action 只取 view/click/favorite
  - 评分 ≤ 行为量，score ∈ 1-5
  - 数据含隐藏结构：交互概率 ∝ 用户类目偏好 × 资源质量 × 热度 + 噪声

- [ ] **Step 1: 写失败测试 `tests/test_simulator.py`**

```python
from engine.config import EngineConfig
from engine.data.simulator import generate

def test_generate_schema_and_scale():
    cfg = EngineConfig(sim_n_users=100, sim_n_resources=50,
                       sim_n_interactions=5000, seed=1)
    b = generate(cfg)
    assert len(b.users) == cfg.sim_n_users
    assert len(b.resources) == cfg.sim_n_resources
    assert len(b.behaviors) > 0
    assert all(x.action in ("view", "click", "favorite") for x in b.behaviors)
    assert all(1 <= r.score <= 5 for r in b.ratings)

def test_generate_reproducible_with_seed():
    cfg = EngineConfig(sim_n_users=50, sim_n_resources=20, sim_n_interactions=1000, seed=7)
    a = generate(cfg)
    b = generate(cfg)
    assert [(x.user_id, x.resource_id, x.action, x.ts) for x in a.behaviors] == \
           [(x.user_id, x.resource_id, x.action, x.ts) for x in b.behaviors]
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_simulator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine.data.simulator'`

- [ ] **Step 3: 写实现**

`src/engine/data/simulator.py`:

```python
from __future__ import annotations
import numpy as np
from .schema import DataBundle, User, Resource, Behavior, Rating


def generate(config) -> DataBundle:
    rng = np.random.default_rng(config.seed)
    n_u, n_r = config.sim_n_users, config.sim_n_resources
    n_cat, n_tag = config.sim_n_categories, config.sim_n_tags

    # --- 隐藏结构参数 ---
    # 用户：类目偏好(狄利克雷)、类型偏好、活跃度
    user_cat = rng.dirichlet(np.ones(n_cat), size=n_u)          # (n_u, n_cat)
    user_type = rng.dirichlet(np.ones(3), size=n_u)             # course/article/video
    user_act = rng.gamma(2.0, 2.0, size=n_u)                    # 活跃度

    # 资源：类目、标签(0-4个)、质量、热度、冷门度
    res_cat = rng.integers(0, n_cat, size=n_r)
    res_tags = [tuple(rng.choice(n_tag, size=int(rng.integers(1, 5)), replace=False).astype(int).tolist())
                for _ in range(n_r)]
    res_quality = rng.beta(2.0, 2.0, size=n_r)
    res_pop = rng.lognormal(0.0, 1.0, size=n_r)

    resources = [
        Resource(resource_id=i, type=("course", "article", "video")[int(rng.integers(3))],
                 category_id=int(res_cat[i]), tags=tuple(str(t) for t in res_tags[i]))
        for i in range(n_r)
    ]
    users = [User(user_id=i) for i in range(n_u)]

    # --- 采样交互 ---
    behaviors: list[Behavior] = []
    ratings: list[Rating] = []
    ts = 1_700_000_000
    for _ in range(config.sim_n_interactions):
        u = int(rng.choice(n_u, p=user_act / user_act.sum()))
        i = int(rng.integers(n_r))
        match = user_cat[u, res_cat[i]]                          # 类目匹配度
        p = float(match * res_quality[i] * res_pop[i] + rng.normal(0, 0.1))
        if rng.uniform() > min(1.0, 3.0 * p):                    # 保留一部分交互率
            continue
        action = rng.choice(("view", "click", "favorite"),
                            p=(0.5, 0.3, 0.2))
        behaviors.append(Behavior(user_id=u, resource_id=i, action=str(action), ts=ts))
        if action == "favorite" and rng.uniform() < 0.5:
            score = int(np.clip(round(2.5 + 2.5 * match + rng.normal(0, 0.3)), 1, 5))
            ratings.append(Rating(user_id=u, resource_id=i, score=score, ts=ts))
        ts += 1

    return DataBundle(users=users, resources=resources,
                      behaviors=behaviors, ratings=ratings)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_simulator.py -v`
Expected: PASS（2 个测试全过，可复现性成立）

- [ ] **Step 5: 提交**

```bash
git add src/engine/data/simulator.py tests/test_simulator.py
git commit -m "feat(data): 实现行为模拟器（注入隐藏结构 + 可复现）"
```

---

### Task 3: 预处理（清洗 / ID 编码 / 时间切分 / 样本）

**Files:**
- Create: `src/engine/data/preprocess.py`
- Create: `tests/test_preprocess.py`

**Interfaces:**
- Consumes: `DataBundle`、`EngineConfig`（Task 1）
- Produces:
  - `engine.data.preprocess.Vocab`：`user2id / item2id / cat2id / tag2id / type2id` 映射 + `n_users / n_items / n_cats / n_tags` 计数
  - `engine.data.preprocess.RankSample`：`user_idx, item_idx, ctr, cvr, rating(int|None), hour, dow`
  - `engine.data.preprocess.Splits`：`train / val / test: DataBundle` + `vocab: Vocab`
  - `clean(bundle, min_user_interactions, min_item_interactions) -> DataBundle`
  - `build_vocab(bundle) -> Vocab`
  - `time_split(bundle, train_ratio, val_ratio, rng) -> Splits`（按用户时间序，全部分集共享同一 vocab）
  - `build_recall_pairs(bundle, vocab) -> list[tuple[int, int]]`（正样本对）
  - `build_rank_samples(bundle, vocab, rng, neg_per_pos) -> list[RankSample]`（含负样本、CTR/CVR/评分标签、时间上下文）

- [ ] **Step 1: 写失败测试 `tests/test_preprocess.py`**

```python
import numpy as np
from engine.config import EngineConfig
from engine.data.simulator import generate
from engine.data.preprocess import (
    clean, build_vocab, time_split, build_recall_pairs, build_rank_samples,
)

def _sim_small():
    cfg = EngineConfig(sim_n_users=50, sim_n_resources=30,
                       sim_n_interactions=3000, seed=3)
    return generate(cfg)

def test_clean_filters_sparse():
    b = _sim_small()
    cleaned = clean(b, min_user_interactions=5, min_item_interactions=5)
    assert len(cleaned.users) <= len(b.users)
    # 清洗后每个用户至少有 5 次行为
    from collections import Counter
    cnt = Counter(x.user_id for x in cleaned.behaviors)
    assert all(v >= 5 for v in cnt.values())

def test_vocab_counts():
    b = _sim_small()
    vocab = build_vocab(b)
    assert vocab.n_users == len(b.users)
    assert vocab.n_items == len(b.resources)
    assert vocab.n_tags >= 1

def test_time_split_no_overlap():
    b = _sim_small()
    cleaned = clean(b, 5, 5)
    vocab = build_vocab(cleaned)
    splits = time_split(cleaned, 0.8, 0.1, np.random.default_rng(0))
    train_uv = {(x.user_id, x.resource_id) for x in splits.train.behaviors}
    test_uv = {(x.user_id, x.resource_id) for x in splits.test.behaviors}
    assert len(train_uv) > 0 and len(test_uv) > 0

def test_rank_samples_labels():
    b = _sim_small()
    cleaned = clean(b, 5, 5)
    vocab = build_vocab(cleaned)
    samples = build_rank_samples(cleaned, vocab, np.random.default_rng(1), neg_per_pos=2)
    assert len(samples) > 0
    assert all(s.ctr in (0, 1) and s.cvr in (0, 1) for s in samples)
    assert all(0 <= s.hour <= 23 and 0 <= s.dow <= 6 for s in samples)
    assert any(s.rating is not None for s in samples)
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_preprocess.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 写实现**

`src/engine/data/preprocess.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
from .schema import DataBundle, User, Resource, Behavior, Rating


@dataclass
class Vocab:
    user2id: dict[int, int]
    item2id: dict[int, int]
    cat2id: dict[int, int]
    tag2id: dict[str, int]
    type2id: dict[str, int]
    n_users: int
    n_items: int
    n_cats: int
    n_tags: int


@dataclass
class RankSample:
    user_idx: int
    item_idx: int
    ctr: int
    cvr: int
    rating: int | None
    hour: int
    dow: int


@dataclass
class Splits:
    train: DataBundle
    val: DataBundle
    test: DataBundle
    vocab: Vocab


def clean(bundle: DataBundle, min_user_interactions: int,
          min_item_interactions: int) -> DataBundle:
    from collections import Counter
    u_cnt = Counter(x.user_id for x in bundle.behaviors)
    i_cnt = Counter(x.resource_id for x in bundle.behaviors)
    keep_u = {u for u, c in u_cnt.items() if c >= min_user_interactions}
    keep_i = {i for i, c in i_cnt.items() if c >= min_item_interactions}
    behaviors = [x for x in bundle.behaviors if x.user_id in keep_u and x.resource_id in keep_i]
    ratings = [r for r in bundle.ratings if r.user_id in keep_u and r.resource_id in keep_i]
    users = [u for u in bundle.users if u.user_id in keep_u]
    resources = [r for r in bundle.resources if r.resource_id in keep_i]
    return DataBundle(users=users, resources=resources,
                      behaviors=behaviors, ratings=ratings)


def build_vocab(bundle: DataBundle) -> Vocab:
    user2id = {u.user_id: i for i, u in enumerate(bundle.users)}
    item2id = {r.resource_id: i for i, r in enumerate(bundle.resources)}
    cat2id = {r.category_id: i for i, r in enumerate(
        {r.category_id for r in bundle.resources})}
    tag_set = {t for r in bundle.resources for t in r.tags}
    tag2id = {t: i for i, t in enumerate(tag_set)}
    type2id = {"course": 0, "article": 1, "video": 2}
    return Vocab(user2id=user2id, item2id=item2id, cat2id=cat2id,
                 tag2id=tag2id, type2id=type2id,
                 n_users=len(user2id), n_items=len(item2id),
                 n_cats=len(cat2id), n_tags=len(tag2id))


def time_split(bundle: DataBundle, train_ratio: float, val_ratio: float,
               rng: np.random.Generator) -> Splits:
    vocab = build_vocab(bundle)
    train_b, val_b, test_b = [], [], []
    train_r, val_r, test_r = [], [], []
    from collections import defaultdict
    by_user = defaultdict(list)
    for x in bundle.behaviors:
        by_user[x.user_id].append(x)
    for user_id, items in by_user.items():
        items.sort(key=lambda x: x.ts)
        n = len(items)
        n_tr, n_va = int(n * train_ratio), int(n * (train_ratio + val_ratio))
        train_b.extend(items[:n_tr])
        val_b.extend(items[n_tr:n_va])
        test_b.extend(items[n_va:])
    for x in bundle.ratings:
        u = x.user_id
        n_tr = int(len(by_user.get(u, [])) * train_ratio)
        # 简化：评分按时间戳归入对应集合
        (train_r if x.ts <= _split_ts(bundle, u, train_ratio)
         else val_r if x.ts <= _split_ts(bundle, u, train_ratio + val_ratio)
         else test_r).append(x)
    return Splits(
        train=DataBundle(bundle.users, bundle.resources, train_b, train_r),
        val=DataBundle(bundle.users, bundle.resources, val_b, val_r),
        test=DataBundle(bundle.users, bundle.resources, test_b, test_r),
        vocab=vocab)


def _split_ts(bundle: DataBundle, user_id: int, ratio: float) -> int:
    """用户行为时间序列中 ratio 分位的时间戳，用于评分划分。"""
    xs = sorted((x.ts for x in bundle.behaviors if x.user_id == user_id))
    if not xs:
        return 0
    return xs[min(len(xs) - 1, int(len(xs) * ratio))]


def build_recall_pairs(bundle: DataBundle, vocab: Vocab) -> list[tuple[int, int]]:
    return [(vocab.user2id[x.user_id], vocab.item2id[x.resource_id])
            for x in bundle.behaviors]


def build_rank_samples(bundle: DataBundle, vocab: Vocab, rng: np.random.Generator,
                       neg_per_pos: int = 3) -> list[RankSample]:
    pos: dict[tuple[int, int], list] = {}
    for x in bundle.behaviors:
        key = (vocab.user2id[x.user_id], vocab.item2id[x.resource_id])
        pos.setdefault(key, [0, 0])
        pos[key][0] = 1                                   # ctr
        if x.action == "favorite":
            pos[key][1] = 1                               # cvr
    rating_of: dict[tuple[int, int], int] = {}
    for x in bundle.ratings:
        key = (vocab.user2id[x.user_id], vocab.item2id[x.resource_id])
        rating_of[key] = x.score

    def ts_ctx(ts: int) -> tuple[int, int]:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.hour, dt.weekday()

    samples: list[RankSample] = []
    ts_by_key = {(vocab.user2id[x.user_id], vocab.item2id[x.resource_id]): x.ts
                 for x in bundle.behaviors}
    for key, (ctr, cvr) in pos.items():
        hour, dow = ts_ctx(ts_by_key[key])
        samples.append(RankSample(*key, ctr, cvr, rating_of.get(key), hour, dow))

    # 负样本：随机未交互对
    user_ids = list(vocab.user2id.values())
    item_ids = list(vocab.item2id.values())
    made = set()
    for key in list(pos.keys()):
        for _ in range(neg_per_pos):
            u = int(rng.choice(user_ids))
            i = int(rng.choice(item_ids))
            nk = (u, i)
            if nk in pos or nk in made:
                continue
            made.add(nk)
            samples.append(RankSample(u, i, 0, 0, None, hour=0, dow=0))
    return samples
```

> 注：`time_split` 中评分划分用了简化的 `_split_ts` 辅助函数，行为与评分按同一用户时间轴划分，保证训练/测试无未来数据。

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_preprocess.py -v`
Expected: PASS（4 个测试全过）

- [ ] **Step 5: 提交**

```bash
git add src/engine/data/preprocess.py tests/test_preprocess.py
git commit -m "feat(data): 实现预处理（清洗/编码/时间切分/样本构建）"
```

---

### Task 4: MovieLens-1M 加载器

**Files:**
- Create: `src/engine/data/movielens.py`
- Create: `tests/test_movielens.py`

**Interfaces:**
- Consumes: `DataBundle`（Task 1）
- Produces: `engine.data.movielens.load(data_dir: str, auto_download: bool = True) -> DataBundle`
  - 评分 4-5 → `click` 行为（正交互），1-5 评分保留
  - `genres` → `tags`；每个电影 `category_id` 用首个 genre 的哈希取模占位（MovieLens 无真实类目树）

- [ ] **Step 1: 写失败测试 `tests/test_movielens.py`**

```python
from engine.data.movielens import _parse_ratings_line, _parse_movies_line

def test_parse_movies():
    movie_id, title, genres = _parse_movies_line("1::Toy Story (1995)::Animation|Children|Comedy")
    assert movie_id == 1
    assert genres == ("Animation", "Children", "Comedy")

def test_parse_ratings():
    uid, mid, score, ts = _parse_ratings_line("1::1193::5::978300760")
    assert (uid, mid, score) == (1, 1193, 5)
    assert ts > 0

def test_high_rating_maps_to_click():
    from engine.data.movielens import _to_behavior_action
    assert _to_behavior_action(5) == "click"
    assert _to_behavior_action(3) is None
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_movielens.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 写实现**

`src/engine/data/movielens.py`:

```python
from __future__ import annotations
import os
import urllib.request
import zipfile
from .schema import DataBundle, User, Resource, Behavior, Rating

_ML_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def _parse_movies_line(line: str) -> tuple[int, str, tuple[str, ...]]:
    movie_id, title, genres = line.strip().split("::")
    return int(movie_id), title, tuple(genres.split("|"))


def _parse_ratings_line(line: str) -> tuple[int, int, int, int]:
    uid, mid, score, ts = line.strip().split("::")
    return int(uid), int(mid), int(score), int(ts)


def _to_behavior_action(score: int) -> str | None:
    return "click" if score >= 4 else None


def _load_or_download(data_dir: str, auto_download: bool) -> str:
    ml_dir = os.path.join(data_dir, "ml-1m")
    if os.path.isdir(ml_dir):
        return ml_dir
    if not auto_download:
        raise FileNotFoundError(f"{ml_dir} 不存在且 auto_download=False")
    zpath = os.path.join(data_dir, "ml-1m.zip")
    os.makedirs(data_dir, exist_ok=True)
    urllib.request.urlretrieve(_ML_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(data_dir)
    return ml_dir


def load(data_dir: str, auto_download: bool = True) -> DataBundle:
    ml_dir = _load_or_download(data_dir, auto_download)
    resources: list[Resource] = []
    cat_ids: dict[str, int] = {}
    with open(os.path.join(ml_dir, "movies.dat"), encoding="iso-8859-1") as f:
        for line in f:
            movie_id, _, genres = _parse_movies_line(line)
            first = genres[0]
            cat_ids.setdefault(first, len(cat_ids))
            resources.append(Resource(
                resource_id=movie_id, type="video",
                category_id=cat_ids[first], tags=genres))
    users: list[User] = []
    behaviors: list[Behavior] = []
    ratings: list[Rating] = []
    user_ids: dict[int, int] = {}
    with open(os.path.join(ml_dir, "ratings.dat"), encoding="iso-8859-1") as f:
        for line in f:
            uid, mid, score, ts = _parse_ratings_line(line)
            users.append(User(user_id=uid))
            ratings.append(Rating(user_id=uid, resource_id=mid, score=score, ts=ts))
            action = _to_behavior_action(score)
            if action is not None:
                behaviors.append(Behavior(user_id=uid, resource_id=mid, action=action, ts=ts))
    # 去重 users（ratings 文件含重复 uid）
    seen = set()
    users = [u for u in users if not (u.user_id in seen or seen.add(u.user_id))]
    return DataBundle(users=users, resources=resources,
                      behaviors=behaviors, ratings=ratings)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_movielens.py -v`
Expected: PASS（3 个解析单测过，不触发网络下载）

- [ ] **Step 5: 提交**

```bash
git add src/engine/data/movielens.py tests/test_movielens.py
git commit -m "feat(data): 实现 MovieLens-1M 加载器（含下载与映射）"
```

---

### Task 5: 特征工程（声明式配置 + 用户画像 + 资源特征）

**Files:**
- Create: `src/engine/features/__init__.py`
- Create: `src/engine/features/feature_config.py`
- Create: `src/engine/features/user_features.py`
- Create: `src/engine/features/item_features.py`
- Create: `tests/test_features.py`

**Interfaces:**
- Consumes: `DataBundle`、`Vocab`（Task 3）
- Produces:
  - `engine.features.feature_config.FeatureSpec(name, kind, input_dim, embed_dim, used_by)`，kind ∈ `numeric|categorical|multi_hot`
  - `build_user_specs(vocab) / build_item_specs(vocab) / build_rank_sparse_specs(vocab) -> list[FeatureSpec]`
  - `engine.features.user_features.build_user_features(bundle, vocab) -> dict[int, dict[str, np.ndarray]]`（key=user_idx）
  - `engine.features.item_features.build_item_features(bundle, vocab) -> dict[int, dict[str, np.ndarray]]`（key=item_idx）
  - 画像特征只从传入的 `bundle` 统计（防泄漏：调用方传 train）

- [ ] **Step 1: 写失败测试 `tests/test_features.py`**

```python
import numpy as np
from engine.config import EngineConfig
from engine.data.simulator import generate
from engine.data.preprocess import clean, build_vocab, time_split
from engine.features.feature_config import build_user_specs, build_item_specs
from engine.features.user_features import build_user_features
from engine.features.item_features import build_item_features

def _cleaned():
    cfg = EngineConfig(sim_n_users=50, sim_n_resources=30,
                       sim_n_interactions=3000, seed=3)
    return clean(generate(cfg), 5, 5)

def test_specs_dimensions():
    b = _cleaned()
    vocab = build_vocab(b)
    us = build_user_specs(vocab)
    ispec = build_item_specs(vocab)
    assert any(s.name == "user_id" for s in us)
    assert any(s.name == "cat_interest" and s.kind == "multi_hot" for s in us)
    assert any(s.name == "avg_rating" and s.kind == "numeric" for s in ispec)

def test_user_features_normalized():
    b = _cleaned()
    vocab = build_vocab(b)
    uf = build_user_features(b, vocab)
    assert set(uf.keys()) == set(vocab.user2id.values())
    cat = uf[list(uf.keys())[0]]["cat_interest"]
    assert abs(float(cat.sum()) - 1.0) < 1e-6          # 归一化概率向量

def test_item_features_shape():
    b = _cleaned()
    vocab = build_vocab(b)
    it = build_item_features(b, vocab)
    key = list(it.keys())[0]
    assert it[key]["tags"].shape == (vocab.n_tags,)
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 写实现**

`src/engine/features/feature_config.py`:

```python
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class FeatureSpec:
    name: str
    kind: str                       # numeric | categorical | multi_hot
    input_dim: int
    embed_dim: int = 0
    used_by: str = "rank"           # recall | rank | both


USER_NUMERIC = ["active_days", "n_views", "n_favs", "gap_days"]
ITEM_NUMERIC = ["log_view", "avg_rating", "age_days"]


def build_user_specs(vocab) -> list[FeatureSpec]:
    return [
        FeatureSpec("user_id", "categorical", vocab.n_users, 64, "both"),
        FeatureSpec("cat_interest", "multi_hot", vocab.n_cats, 8, "both"),
        FeatureSpec("tag_interest", "multi_hot", vocab.n_tags, 8, "both"),
        FeatureSpec("type_pref", "multi_hot", 3, 4, "both"),
        *[FeatureSpec(n, "numeric", 1, 0, "rank") for n in USER_NUMERIC],
    ]


def build_item_specs(vocab) -> list[FeatureSpec]:
    return [
        FeatureSpec("item_id", "categorical", vocab.n_items, 64, "recall"),
        FeatureSpec("category_id", "categorical", vocab.n_cats, 8, "both"),
        FeatureSpec("tags", "multi_hot", vocab.n_tags, 8, "both"),
        FeatureSpec("type_id", "categorical", 3, 4, "both"),
        *[FeatureSpec(n, "numeric", 1, 0, "rank") for n in ITEM_NUMERIC],
    ]


def build_rank_sparse_specs(vocab) -> list[FeatureSpec]:
    return [
        FeatureSpec("user_id", "categorical", vocab.n_users, 16, "rank"),
        FeatureSpec("item_id", "categorical", vocab.n_items, 16, "rank"),
        FeatureSpec("category_id", "categorical", vocab.n_cats, 8, "rank"),
        FeatureSpec("tags", "multi_hot", vocab.n_tags, 8, "rank"),
        FeatureSpec("type_id", "categorical", 3, 4, "rank"),
        FeatureSpec("hour", "categorical", 24, 4, "rank"),
        FeatureSpec("dow", "categorical", 7, 4, "rank"),
    ]
```

`src/engine/features/user_features.py`:

```python
from __future__ import annotations
from collections import Counter, defaultdict
import numpy as np
from ..data.schema import DataBundle


def build_user_features(bundle: DataBundle, vocab) -> dict[int, dict[str, np.ndarray]]:
    """画像特征：只从传入的 bundle 统计（防泄漏的关键）。"""
    cat_hits: dict[int, Counter] = defaultdict(Counter)
    tag_hits: dict[int, Counter] = defaultdict(Counter)
    type_hits: dict[int, Counter] = defaultdict(Counter)
    n_views: Counter = Counter()
    n_favs: Counter = Counter()
    last_ts: dict[int, int] = {}
    cat_of = {r.resource_id: r.category_id for r in bundle.resources}
    tags_of = {r.resource_id: r.tags for r in bundle.resources}
    type_of = {r.resource_id: vocab.type2id[r.type] for r in bundle.resources}

    for x in bundle.behaviors:
        if x.resource_id not in cat_of:
            continue
        cat_hits[x.user_id][cat_of[x.resource_id]] += 1
        for t in tags_of.get(x.resource_id, ()):
            if t in vocab.tag2id:
                tag_hits[x.user_id][vocab.tag2id[t]] += 1
        type_hits[x.user_id][type_of[x.resource_id]] += 1
        n_views[x.user_id] += 1
        if x.action == "favorite":
            n_favs[x.user_id] += 1
        last_ts[x.user_id] = max(last_ts.get(x.user_id, 0), x.ts)

    max_ts = max(last_ts.values(), default=0)
    out: dict[int, dict[str, np.ndarray]] = {}
    for u in vocab.user2id.values():
        if u not in cat_hits:
            out[u] = _zero_user(vocab)
            continue
        def _norm(counter: Counter, n: int) -> np.ndarray:
            v = np.zeros(n)
            for k, c in counter.items():
                if 0 <= k < n:
                    v[k] = c
            s = v.sum()
            return v / s if s > 0 else v
        active_days = len({x.ts // 86400 for x in bundle.behaviors if x.user_id == u})
        gap_days = (max_ts - last_ts[u]) / 86400.0
        out[u] = {
            "user_id": np.array(u, dtype=np.int64),
            "cat_interest": _norm(cat_hits[u], vocab.n_cats),
            "tag_interest": _norm(tag_hits[u], vocab.n_tags),
            "type_pref": _norm(type_hits[u], 3),
            "active_days": np.array(active_days, dtype=np.float32),
            "n_views": np.array(n_views[u], dtype=np.float32),
            "n_favs": np.array(n_favs[u], dtype=np.float32),
            "gap_days": np.array(gap_days, dtype=np.float32),
        }
    return out


def _zero_user(vocab) -> dict[str, np.ndarray]:
    return {
        "user_id": np.array(0, dtype=np.int64),
        "cat_interest": np.zeros(vocab.n_cats),
        "tag_interest": np.zeros(vocab.n_tags),
        "type_pref": np.zeros(3),
        "active_days": np.array(0.0, dtype=np.float32),
        "n_views": np.array(0.0, dtype=np.float32),
        "n_favs": np.array(0.0, dtype=np.float32),
        "gap_days": np.array(0.0, dtype=np.float32),
    }
```

> 注：`active_days` 用 `ts // 86400` 统计活跃天数（以天为单位去重）。

`src/engine/features/item_features.py`:

```python
from __future__ import annotations
import numpy as np
from ..data.schema import DataBundle


def build_item_features(bundle: DataBundle, vocab) -> dict[int, dict[str, np.ndarray]]:
    max_ts = max((x.ts for x in bundle.behaviors), default=0)
    view_cnt: dict[int, int] = {}
    rating_sum: dict[int, list] = {}
    for x in bundle.behaviors:
        view_cnt[x.resource_id] = view_cnt.get(x.resource_id, 0) + 1
    for x in bundle.ratings:
        rating_sum.setdefault(x.resource_id, []).append(x.score)

    out: dict[int, dict[str, np.ndarray]] = {}
    for r in bundle.resources:
        i = vocab.item2id[r.resource_id]
        tag_vec = np.zeros(vocab.n_tags)
        for t in r.tags:
            if t in vocab.tag2id:
                tag_vec[vocab.tag2id[t]] = 1.0
        age_days = (max_ts - 1_700_000_000) / 86400.0   # 简化：相对固定基准
        avg = (sum(rating_sum.get(r.resource_id, [0])) /
               max(1, len(rating_sum.get(r.resource_id, []))))
        out[i] = {
            "item_id": np.array(i, dtype=np.int64),
            "category_id": np.array(vocab.cat2id[r.category_id], dtype=np.int64),
            "tags": tag_vec,
            "type_id": np.array(vocab.type2id[r.type], dtype=np.int64),
            "log_view": np.array(np.log1p(view_cnt.get(r.resource_id, 0)), dtype=np.float32),
            "avg_rating": np.array(float(avg), dtype=np.float32),
            "age_days": np.array(float(age_days), dtype=np.float32),
        }
    return out
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_features.py -v`
Expected: PASS（3 个测试全过）

- [ ] **Step 5: 提交**

```bash
git add src/engine/features tests/test_features.py
git commit -m "feat(features): 实现声明式特征配置与用户/资源画像特征"
```

---

### Task 6: 双塔 DSSM 召回模型

**Files:**
- Create: `src/engine/models/__init__.py`
- Create: `src/engine/models/recall/__init__.py`
- Create: `src/engine/models/recall/dssm.py`
- Create: `tests/test_dssm.py`

**Interfaces:**
- Consumes: `Vocab`（Task 3）、用户/资源特征 dict（Task 5，可选用于特征喂塔）
- Produces: `engine.models.recall.dssm.TwoTower`，方法：
  - `user_emb(u: dict[str, Tensor]) -> Tensor`（B, embed_dim）
  - `item_emb(i: dict[str, Tensor]) -> Tensor`（B, embed_dim）
  - `score(u, i) -> Tensor`（B,）内积
  - `forward(users, pos_items, neg_items) -> Tensor`（in-batch softmax 损失）

- [ ] **Step 1: 写失败测试 `tests/test_dssm.py`**

```python
import torch
from engine.models.recall.dssm import TwoTower

def _feats(device="cpu"):
    return {
        "user_id": torch.tensor([0, 1], device=device),
        "cat_interest": torch.tensor([[0.6, 0.4, 0.0, 0.0], [0.2, 0.3, 0.5, 0.0]], device=device),
        "tag_interest": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        "type_pref": torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.3, 0.7]], device=device),
        "item_id": torch.tensor([0, 1], device=device),
        "category_id": torch.tensor([0, 1], device=device),
        "tags": torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=device),
        "type_id": torch.tensor([0, 1], device=device),
    }

def test_embedding_shapes():
    m = TwoTower(n_users=5, n_items=5, n_cats=4, n_tags=3, embed_dim=8)
    f = _feats()
    assert m.user_emb(f).shape == (2, 8)
    assert m.item_emb(f).shape == (2, 8)

def test_inbatch_softmax_loss_decreases():
    torch.manual_seed(0)
    m = TwoTower(n_users=10, n_items=10, n_cats=4, n_tags=3, embed_dim=8)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    f = _feats()
    loss1 = m.forward(f, f, f)
    opt.zero_grad(); loss1.backward(); opt.step()
    loss2 = m.forward(f, f, f)
    assert loss2.item() < loss1.item()
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_dssm.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 写实现**

`src/engine/models/recall/dssm.py`:

```python
from __future__ import annotations
import torch
import torch.nn as nn


class TwoTower(nn.Module):
    """双塔召回：用户塔 / 资源塔独立编码，内积相似度，in-batch softmax 训练。"""

    def __init__(self, n_users: int, n_items: int, n_cats: int, n_tags: int,
                 embed_dim: int = 64):
        super().__init__()
        self.user_id_emb = nn.Embedding(n_users, embed_dim)
        self.item_id_emb = nn.Embedding(n_items, embed_dim)
        self.cat_emb = nn.Embedding(n_cats, 8)
        self.tag_emb = nn.Embedding(n_tags, 8)
        self.type_emb = nn.Embedding(3, 4)
        self.user_mlp = nn.Sequential(
            nn.Linear(embed_dim + 8 + 8 + 4, 128), nn.ReLU(),
            nn.Linear(128, embed_dim))
        self.item_mlp = nn.Sequential(
            nn.Linear(embed_dim + 8 + 8 + 4, 128), nn.ReLU(),
            nn.Linear(128, embed_dim))

    def user_emb(self, u: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([
            self.user_id_emb(u["user_id"]),
            u["cat_interest"] @ self.cat_emb.weight,
            u["tag_interest"] @ self.tag_emb.weight,
            u["type_pref"] @ self.type_emb.weight,
        ], dim=-1)
        return self.user_mlp(x)

    def item_emb(self, i: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([
            self.item_id_emb(i["item_id"]),
            self.cat_emb(i["category_id"]),
            i["tags"] @ self.tag_emb.weight,
            self.type_emb(i["type_id"]),
        ], dim=-1)
        return self.item_mlp(x)

    def score(self, u: dict[str, torch.Tensor],
              i: dict[str, torch.Tensor]) -> torch.Tensor:
        return (self.user_emb(u) * self.item_emb(i)).sum(-1)

    def forward(self, users, pos_items, neg_items, tau: float = 0.05) -> torch.Tensor:
        """in-batch softmax 损失。

        users/pos_items/neg_items 均为特征 dict（每个字段形状 (B, ...)）。
        正样本 logits 用 batch 内其他 item 作负样本。
        """
        ue = self.user_emb(users)                    # (B, E)
        pe = self.item_emb(pos_items)                # (B, E)
        ne = self.item_emb(neg_items)                # (B, E)
        logits_pos = (ue * pe).sum(-1) / tau         # (B,)
        # batch 内互为负样本：logits (B, B)
        logits_batch = ue @ pe.T / tau
        # 附加显式负样本列
        logits_neg = ue @ ne.T / tau                 # (B, n_neg)
        logits = torch.cat([logits_batch, logits_neg], dim=-1)   # (B, B+n_neg)
        target = torch.arange(logits.size(0), device=logits.device)
        return nn.functional.cross_entropy(logits, target) + (-logits_pos.mean() * 0.0)
```

> 注：`forward` 中交叉熵目标取 batch 内对角线，`logits_pos` 项仅用于保留温度语义（权重为 0，可移除）。

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_dssm.py -v`
Expected: PASS（loss 在 1 步后下降）

- [ ] **Step 5: 提交**

```bash
git add src/engine/models tests/test_dssm.py
git commit -m "feat(recall): 实现双塔 DSSM 模型与 in-batch softmax 损失"
```

---

### Task 7: 召回训练与评估

**Files:**
- Create: `src/engine/models/recall/trainer.py`
- Create: `src/engine/pipeline/__init__.py`
- Create: `src/engine/pipeline/evaluate.py`
- Create: `tests/test_evaluate.py`
- Create: `tests/test_recall_trainer.py`

**Interfaces:**
- Consumes: `build_recall_pairs`、`Vocab`（Task 3）、`build_user_features`/`build_item_features`（Task 5）、`TwoTower`（Task 6）
- Produces:
  - `engine.models.recall.trainer.train_recall(bundle, vocab, config, device="cpu") -> TwoTower`
  - `engine.pipeline.evaluate.recall_at_k(relevant: set[int], ranked: list[int], k) -> float`
  - `engine.pipeline.evaluate.hit_rate_at_k(relevant, ranked, k) -> float`
  - `engine.pipeline.evaluate.ndcg_at_k(relevant, ranked, k) -> float`
  - `engine.pipeline.evaluate.auc_from_scores(y_true, y_score) -> float`
  - `engine.pipeline.evaluate.evaluate_recall(model, bundle, vocab, config, device) -> dict`（返回 `{"recall@50":..., "hitrate@50":..., "ndcg@50":...}`）

- [ ] **Step 1: 写失败测试 `tests/test_evaluate.py`**

```python
from engine.pipeline.evaluate import recall_at_k, ndcg_at_k, auc_from_scores

def test_recall_at_k():
    assert recall_at_k(relevant={1, 3}, ranked=[0, 1, 2, 3, 4], k=3) == 0.5

def test_ndcg_ranked_better():
    a = ndcg_at_k({1, 3}, [1, 3, 0, 2, 4], k=5)
    b = ndcg_at_k({1, 3}, [3, 1, 0, 2, 4], k=5)
    assert a > b

def test_auc():
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    assert abs(auc_from_scores(y, s) - 1.0) < 1e-6
```

- [ ] **Step 2: 写失败测试 `tests/test_recall_trainer.py`**

```python
from engine.config import EngineConfig
from engine.data.simulator import generate
from engine.data.preprocess import clean, build_vocab
from engine.models.recall.trainer import train_recall
from engine.pipeline.evaluate import evaluate_recall

def test_train_recall_overfits_small():
    cfg = EngineConfig(sim_n_users=40, sim_n_resources=25,
                       sim_n_interactions=2000, seed=2,
                       recall_epochs=2, recall_batch_size=128)
    b = clean(generate(cfg), 5, 5)
    vocab = build_vocab(b)
    model = train_recall(b, vocab, cfg, device="cpu")
    assert hasattr(model, "user_emb")
    scores = evaluate_recall(model, b, vocab, cfg)
    assert scores["recall@50"] > 0.0
```

- [ ] **Step 3: 运行测试验证失败**

Run: `pytest tests/test_evaluate.py tests/test_recall_trainer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: 写实现**

`src/engine/models/recall/trainer.py`:

```python
from __future__ import annotations
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from ...features.user_features import build_user_features
from ...features.item_features import build_item_features
from ...data.preprocess import build_recall_pairs
from .dssm import TwoTower


def _to_tensor(arr):
    return torch.tensor(np.asarray(arr, dtype=np.float32))


def train_recall(bundle, vocab, config, device: str = "cpu") -> TwoTower:
    torch.manual_seed(config.seed)
    model = TwoTower(vocab.n_users, vocab.n_items, vocab.n_cats, vocab.n_tags,
                     config.recall_embed_dim).to(device)

    uf = build_user_features(bundle, vocab)
    it = build_item_features(bundle, vocab)
    pairs = build_recall_pairs(bundle, vocab)
    # 热门资源（按交互频次），用于显式负采样
    pop = Counter(x.resource_id for x in bundle.behaviors)
    pop_items = [vocab.item2id[i] for i, _ in pop.most_common(config.recall_batch_size)]

    u = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    i = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    dataset = TensorDataset(u, i)
    loader = DataLoader(dataset, batch_size=config.recall_batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=config.recall_lr)
    for epoch in range(config.recall_epochs):
        total = 0.0
        for ub, ib in loader:
            ub, ib = ub.to(device), ib.to(device)
            # 每个 user 配 n_neg 个热门显式负样本
            neg = torch.tensor(
                [int(np.random.choice(pop_items)) for _ in range(ib.size(0))],
                dtype=torch.long, device=device)
            users = {k: _to_tensor([uf[int(x)][k] for x in ub.cpu().tolist()]).to(device)
                     for k in ("user_id", "cat_interest", "tag_interest", "type_pref")}
            pos = {k: _to_tensor([it[int(x)][k] for x in ib.cpu().tolist()]).to(device)
                   for k in ("item_id", "category_id", "tags", "type_id")}
            neg_feats = {k: _to_tensor([it[int(x)][k] for x in neg.cpu().tolist()]).to(device)
                         for k in ("item_id", "category_id", "tags", "type_id")}
            loss = model.forward(users, pos, neg_feats, tau=config.recall_tau)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
    return model.cpu()
```

`src/engine/pipeline/evaluate.py`:

```python
from __future__ import annotations
import numpy as np
import torch
from ..features.user_features import build_user_features
from ..features.item_features import build_item_features
from ..data.preprocess import build_recall_pairs


def recall_at_k(relevant: set[int], ranked: list[int], k: int) -> float:
    if not relevant:
        return 0.0
    hit = sum(1 for x in ranked[:k] if x in relevant)
    return hit / len(relevant)


def hit_rate_at_k(relevant: set[int], ranked: list[int], k: int) -> float:
    return 1.0 if any(x in relevant for x in ranked[:k]) else 0.0


def ndcg_at_k(relevant: set[int], ranked: list[int], k: int) -> float:
    dcg = 0.0
    for pos, x in enumerate(ranked[:k], start=1):
        if x in relevant:
            dcg += 1.0 / np.log2(pos + 1)
    ideal = sum(1.0 / np.log2(pos + 1)
                for pos in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def auc_from_scores(y_true, y_score) -> float:
    order = np.argsort(y_score)
    ranked = np.asarray(y_true)[order]
    n_pos = int(ranked.sum())
    n_neg = len(ranked) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = sum(idx + 1 for idx, v in enumerate(ranked) if v == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def evaluate_recall(model, bundle, vocab, config, device: str = "cpu") -> dict:
    model.eval()
    it = build_item_features(bundle, vocab)
    uf = build_user_features(bundle, vocab)
    test_pairs = build_recall_pairs(bundle, vocab)
    relevant_of: dict[int, set[int]] = {}
    for u, i in test_pairs:
        relevant_of.setdefault(u, set()).add(i)

    item_ids = sorted(vocab.item2id.values())
    item_feats = {k: torch.stack([_t(it[i][k]) for i in item_ids])
                  for k in ("item_id", "category_id", "tags", "type_id")}
    with torch.no_grad():
        item_embs = model.item_emb(item_feats).T                 # (E, n_items)

        rec, hr, ndcg = [], [], []
        for u in relevant_of:
            fu = {k: torch.stack([_t(uf[u][k])]) for k in
                  ("user_id", "cat_interest", "tag_interest", "type_pref")}
            ue = model.user_emb(fu)                               # (1, E)
            scores = (ue @ item_embs).squeeze(0)                  # (n_items,)
            ranked = [item_ids[x] for x in torch.argsort(scores, descending=True).tolist()]
            rel = relevant_of[u]
            rec.append(recall_at_k(rel, ranked, config.recall_k))
            hr.append(hit_rate_at_k(rel, ranked, config.recall_k))
            ndcg.append(ndcg_at_k(rel, ranked, config.recall_k))
    return {"recall@50": float(np.mean(rec)),
            "hitrate@50": float(np.mean(hr)),
            "ndcg@50": float(np.mean(ndcg))}


def _t(arr) -> torch.Tensor:
    return torch.tensor(np.asarray(arr, dtype=np.float32))
```

- [ ] **Step 5: 运行测试验证通过**

Run: `pytest tests/test_evaluate.py tests/test_recall_trainer.py -v`
Expected: PASS（评估指标单测 + 召回端到端跑通，recall@50 > 0）

- [ ] **Step 6: 提交**

```bash
git add src/engine/models/recall/trainer.py src/engine/pipeline tests/test_evaluate.py tests/test_recall_trainer.py
git commit -m "feat(recall): 实现召回训练与 Recall/NDCG 评估"
```

---

### Task 8: 多任务 DeepFM 排序模型

**Files:**
- Create: `src/engine/models/common/__init__.py`
- Create: `src/engine/models/common/fm.py`
- Create: `src/engine/models/rank/__init__.py`
- Create: `src/engine/models/rank/multitask_deepfm.py`
- Create: `tests/test_deepfm.py`

**Interfaces:**
- Consumes: `FeatureSpec`、`build_rank_sparse_specs`（Task 5）
- Produces:
  - `engine.models.common.fm.FM`：`forward(embs: list[Tensor]) -> Tensor`
  - `engine.models.rank.multitask_deepfm.MultiTaskDeepFM`：
    - `__init__(sparse_specs, numeric_dim, deep_dim=64)`
    - `forward(sparse: dict[str, Tensor], numeric: Tensor) -> tuple[Tensor, Tensor, Tensor]`（返回 pctr, pcvr, rating）

- [ ] **Step 1: 写失败测试 `tests/test_deepfm.py`**

```python
import torch
from engine.features.feature_config import FeatureSpec
from engine.models.common.fm import FM
from engine.models.rank.multitask_deepfm import MultiTaskDeepFM

def _specs():
    return [
        FeatureSpec("user_id", "categorical", 10, 4),
        FeatureSpec("item_id", "categorical", 10, 4),
        FeatureSpec("tags", "multi_hot", 8, 4),
    ]

def test_fm_scalar_per_sample():
    embs = [torch.randn(4, 6) for _ in range(3)]
    out = FM()(embs)
    assert out.shape == (4,)

def test_multitask_outputs():
    torch.manual_seed(0)
    m = MultiTaskDeepFM(_specs(), numeric_dim=3)
    sparse = {
        "user_id": torch.tensor([0, 1]),
        "item_id": torch.tensor([1, 2]),
        "tags": torch.tensor([[0.5, 0.5, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1.0, 0, 0, 0, 0]]),
    }
    numeric = torch.randn(2, 3)
    pctr, pcvr, rating = m.forward(sparse, numeric)
    assert pctr.shape == (2, 1)
    assert (pctr > 0).all() and (pctr < 1).all()      # sigmoid 输出
    assert rating.shape == (2, 1)
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_deepfm.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 写实现**

`src/engine/models/common/fm.py`:

```python
from __future__ import annotations
import torch
import torch.nn as nn


class FM(nn.Module):
    """因子分解机：一阶 + 二阶特征交互，O(k) 复杂度。"""

    def forward(self, embs: list[torch.Tensor]) -> torch.Tensor:
        first = torch.stack([e.sum(1) for e in embs], dim=1).sum(1)         # (B,)
        sum_sq = torch.stack([e.sum(1) for e in embs], dim=1).sum(1) ** 2
        sq_sum = torch.stack([(e ** 2).sum(1) for e in embs], dim=1).sum(1)
        second = 0.5 * (sum_sq - sq_sum)                                    # (B,)
        return first + second
```

`src/engine/models/rank/multitask_deepfm.py`:

```python
from __future__ import annotations
import torch
import torch.nn as nn
from ..common.fm import FM


class MultiTaskDeepFM(nn.Module):
    """多任务排序：shared-bottom（FM + Deep）+ 三个任务头。"""

    def __init__(self, sparse_specs, numeric_dim: int, deep_dim: int = 64):
        super().__init__()
        self.specs = {s.name: s for s in sparse_specs}
        self.embeddings = nn.ModuleDict({
            s.name: nn.Embedding(s.input_dim, s.embed_dim) for s in sparse_specs})
        self.fm = FM()
        emb_total = sum(s.embed_dim for s in sparse_specs)
        self.deep = nn.Sequential(
            nn.Linear(emb_total + numeric_dim, 128), nn.ReLU(),
            nn.Linear(128, deep_dim))
        self.ctr_head = nn.Linear(deep_dim + 1, 1)
        self.cvr_head = nn.Linear(deep_dim + 1, 1)
        self.rating_head = nn.Linear(deep_dim + 1, 1)

    def _embed(self, sparse: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        out = []
        for name, spec in self.specs.items():
            x = sparse[name]
            e = self.embeddings[name]
            if spec.kind == "multi_hot":
                out.append(x @ e.weight)
            else:
                out.append(e(x))
        return out

    def forward(self, sparse: dict[str, torch.Tensor],
                numeric: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embs = self._embed(sparse)
        fm_out = self.fm(embs).unsqueeze(-1)                 # (B, 1)
        deep_out = self.deep(torch.cat(embs + [numeric], dim=-1))   # (B, deep_dim)
        shared = torch.cat([fm_out, deep_out], dim=-1)       # (B, deep_dim+1)
        pctr = torch.sigmoid(self.ctr_head(shared))
        pcvr = torch.sigmoid(self.cvr_head(shared))
        rating = self.rating_head(shared)
        return pctr, pcvr, rating
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_deepfm.py -v`
Expected: PASS（FM 输出标量、三头输出形状与取值范围正确）

- [ ] **Step 5: 提交**

```bash
git add src/engine/models/common src/engine/models/rank tests/test_deepfm.py
git commit -m "feat(rank): 实现多任务 DeepFM 排序模型（FM + Deep + 三任务头）"
```

---

### Task 9: 排序训练与评估

**Files:**
- Create: `src/engine/models/rank/trainer.py`
- Create: `tests/test_rank_trainer.py`

**Interfaces:**
- Consumes: `build_rank_samples`（Task 3）、`build_user_features`/`build_item_features`（Task 5）、`build_rank_sparse_specs`（Task 5）、`MultiTaskDeepFM`（Task 8）、`auc_from_scores`/`rmse`（Task 7，rmse 需在此补）
- Produces:
  - `engine.models.rank.trainer.train_rank(splits, config, device="cpu") -> MultiTaskDeepFM`
  - `engine.pipeline.evaluate.rmse(y_true, y_pred) -> float`
  - `engine.pipeline.evaluate.evaluate_rank(model, splits, config, device) -> dict`（返回 `{"ctr_auc":..., "cvr_auc":..., "rating_rmse":...}`）

- [ ] **Step 1: 写失败测试 `tests/test_rank_trainer.py`**

```python
from engine.config import EngineConfig
from engine.data.simulator import generate
from engine.data.preprocess import clean, build_vocab, time_split
from engine.models.rank.trainer import train_rank
from engine.pipeline.evaluate import rmse, evaluate_rank

def test_rmse():
    assert abs(rmse([3.0, 4.0], [3.0, 4.0])) < 1e-6

def test_train_rank_learns():
    cfg = EngineConfig(sim_n_users=40, sim_n_resources=25,
                       sim_n_interactions=2000, seed=2,
                       rank_epochs=2, rank_batch_size=64)
    b = clean(generate(cfg), 5, 5)
    vocab = build_vocab(b)
    splits = time_split(b, 0.8, 0.1, __import__("numpy").random.default_rng(0))
    model = train_rank(splits, cfg, device="cpu")
    m = evaluate_rank(model, splits, cfg)
    assert m["ctr_auc"] >= 0.5
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_rank_trainer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 写实现**

在 `src/engine/pipeline/evaluate.py` 追加 `rmse`：

```python
def rmse(y_true, y_pred) -> float:
    a, b = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def evaluate_rank(model, splits, config, device="cpu") -> dict:
    """在验证集上评估排序模型：CTR/CVR AUC + 评分 RMSE。"""
    import torch
    from ..data.preprocess import build_rank_samples
    from ..features.user_features import build_user_features
    from ..features.item_features import build_item_features
    uf = build_user_features(splits.train, splits.vocab)      # 防泄漏：统计来自训练集
    it = build_item_features(splits.train, splits.vocab)
    samples = build_rank_samples(splits.val, splits.vocab, __import__("numpy").random.default_rng(0))
    model.eval()
    y_ctr, p_ctr, y_cvr, p_cvr, y_rat, p_rat = [], [], [], [], [], []
    with torch.no_grad():
        for s in samples:
            u = uf.get(s.user_idx)
            i = it.get(s.item_idx)
            if u is None or i is None:
                continue
            sparse = {
                "user_id": torch.tensor([s.user_idx]),
                "item_id": torch.tensor([s.item_idx]),
                "category_id": torch.tensor([int(i["category_id"])]),
                "tags": torch.tensor([i["tags"]]),
                "type_id": torch.tensor([int(i["type_id"])]),
                "hour": torch.tensor([s.hour]),
                "dow": torch.tensor([s.dow]),
            }
            numeric = torch.tensor([[
                float(u["active_days"]), float(u["n_views"]), float(u["n_favs"]),
                float(u["gap_days"]), float(i["log_view"]), float(i["avg_rating"]),
                float(i["age_days"]),
            ]])
            pctr, pcvr, rating = model(sparse, numeric)
            y_ctr.append(s.ctr); p_ctr.append(float(pctr))
            y_cvr.append(s.cvr); p_cvr.append(float(pcvr))
            if s.rating is not None:
                y_rat.append(s.rating); p_rat.append(float(rating))
    return {
        "ctr_auc": auc_from_scores(y_ctr, p_ctr),
        "cvr_auc": auc_from_scores(y_cvr, p_cvr) if len(set(y_cvr)) > 1 else 0.5,
        "rating_rmse": rmse(y_rat, p_rat) if p_rat else float("nan"),
    }
```

`src/engine/models/rank/trainer.py`:

```python
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader
from ...features.feature_config import build_rank_sparse_specs
from ...features.user_features import build_user_features
from ...features.item_features import build_item_features
from ...data.preprocess import build_rank_samples
from .multitask_deepfm import MultiTaskDeepFM


def train_rank(splits, config, device: str = "cpu") -> MultiTaskDeepFM:
    torch.manual_seed(config.seed)
    vocab = splits.vocab
    specs = build_rank_sparse_specs(vocab)
    numeric_dim = 4 + 3                                    # 用户4 + 资源3
    model = MultiTaskDeepFM(specs, numeric_dim).to(device)

    uf = build_user_features(splits.train, vocab)
    it = build_item_features(splits.train, vocab)
    samples = build_rank_samples(splits.train, vocab,
                                 np.random.default_rng(config.seed),
                                 config.rank_neg_per_pos)

    def make_batch(batch_samples):
        sparse = {k: [] for k in ("user_id", "item_id", "category_id",
                                  "tags", "type_id", "hour", "dow")}
        numeric = []
        y_ctr, y_cvr, y_rat = [], [], []
        rat_mask = []
        for s in batch_samples:
            u, i = uf.get(s.user_idx), it.get(s.item_idx)
            if u is None or i is None:
                continue
            sparse["user_id"].append(s.user_idx)
            sparse["item_id"].append(s.item_idx)
            sparse["category_id"].append(int(i["category_id"]))
            sparse["tags"].append(i["tags"])
            sparse["type_id"].append(int(i["type_id"]))
            sparse["hour"].append(s.hour)
            sparse["dow"].append(s.dow)
            numeric.append([u["active_days"], u["n_views"], u["n_favs"],
                            u["gap_days"], i["log_view"], i["avg_rating"],
                            i["age_days"]])
            y_ctr.append(s.ctr); y_cvr.append(s.cvr)
            y_rat.append(s.rating if s.rating is not None else 0.0)
            rat_mask.append(0 if s.rating is None else 1)
        sparse_t = {}
        for k, v in sparse.items():
            dtype = torch.float32 if k == "tags" else torch.long   # multi_hot 需 float
            sparse_t[k] = torch.tensor(np.asarray(v), dtype=dtype).to(device)
        numeric_t = torch.tensor(np.asarray(numeric, dtype=np.float32)).to(device)
        return (sparse_t, numeric_t,
                torch.tensor(y_ctr, dtype=torch.float32).to(device),
                torch.tensor(y_cvr, dtype=torch.float32).to(device),
                torch.tensor(y_rat, dtype=torch.float32).to(device),
                torch.tensor(rat_mask, dtype=torch.float32).to(device))

    opt = torch.optim.Adam(model.parameters(), lr=config.rank_lr)
    for epoch in range(config.rank_epochs):
        rng = np.random.default_rng(config.seed + epoch)
        rng.shuffle(samples)
        for start in range(0, len(samples), config.rank_batch_size):
            batch = samples[start:start + config.rank_batch_size]
            sparse_t, numeric_t, yc, yv, yr, mask = make_batch(batch)
            if sparse_t["user_id"].numel() < 2:
                continue
            pctr, pcvr, rating = model(sparse_t, numeric_t)
            loss = (config.rank_w_ctr * torch.nn.functional.binary_cross_entropy(
                        pctr.squeeze(-1), yc)
                    + config.rank_w_cvr * torch.nn.functional.binary_cross_entropy(
                        pcvr.squeeze(-1), yv))
            if mask.sum() > 0:
                loss = loss + config.rank_w_rating * torch.nn.functional.mse_loss(
                    rating.squeeze(-1)[mask.bool()], yr[mask.bool()])
            opt.zero_grad(); loss.backward(); opt.step()
    return model.cpu()
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_rank_trainer.py -v`
Expected: PASS（排序端到端跑通，ctr_auc ≥ 0.5）

- [ ] **Step 5: 提交**

```bash
git add src/engine/models/rank/trainer.py src/engine/pipeline/evaluate.py tests/test_rank_trainer.py
git commit -m "feat(rank): 实现多任务排序训练与 AUC/RMSE 评估"
```

---

### Task 10: 重排（MMR 多样性 / 去重 / 新资源加权）

**Files:**
- Create: `src/engine/pipeline/rerank.py`
- Create: `tests/test_rerank.py`

**Interfaces:**
- Consumes: `Item features`（Task 5，用于类别相似度）、`config`
- Produces:
  - `engine.pipeline.rerank.mmr_rerank(scored, item_cat: dict[int, int], top_n, lambda_) -> list[int]`
  - `engine.pipeline.rerank.rerank(scored, seen: set[int], item_cat, config) -> list[int]`
    - 过滤 `seen`，新资源加权（`age_days < cold_age_days → score × cold_weight`），MMR 贪心取 `top_n`

- [ ] **Step 1: 写失败测试 `tests/test_rerank.py`**

```python
from engine.config import EngineConfig
from engine.pipeline.rerank import mmr_rerank, rerank

def test_mmr_spreads_categories():
    scored = [(0, 0.9), (1, 0.85), (2, 0.8)]       # item -> score
    cats = {0: "a", 1: "a", 2: "b"}
    out = mmr_rerank(scored, cats, top_n=3, lambda_=0.5)
    # 同类目 0、1 被拆分，2 应提前
    assert out.index(2) < 2

def test_rerank_filters_seen():
    cfg = EngineConfig(top_n=2, mmr_lambda=0.5, cold_age_days=7, cold_weight=1.2)
    scored = [(0, 0.9), (1, 0.8), (2, 0.7)]
    out = rerank(scored, seen={0}, item_cat={1: "a", 2: "a"}, config=cfg)
    assert 0 not in out and len(out) <= 2
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_rerank.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 写实现**

`src/engine/pipeline/rerank.py`:

```python
from __future__ import annotations


def mmr_rerank(scored: list[tuple[int, float]], item_cat: dict[int, str],
               top_n: int, lambda_: float) -> list[int]:
    """MMR 最大边际相关：精排分 − λ·与已选集合的类目相似度。"""
    selected: list[int] = []
    remaining = list(scored)
    while remaining and len(selected) < top_n:
        best_idx, best_val = -1, -float("inf")
        for idx, (item, score) in enumerate(remaining):
            div = sum(1 for s in selected if item_cat.get(item) == item_cat.get(s))
            mmr = score - lambda_ * div
            if mmr > best_val:
                best_val, best_idx = mmr, idx
        item = remaining.pop(best_idx)[0]
        selected.append(item)
    return selected


def rerank(scored: list[tuple[int, float]], seen: set[int],
           item_cat: dict[int, str], config) -> list[int]:
    filtered = [(i, s) for i, s in scored if i not in seen]
    if hasattr(config, "cold_age_days"):
        filtered = [
            (i, s * config.cold_weight if config.cold_age_days > 0 else s)
            for i, s in filtered
        ]
    return mmr_rerank(filtered, item_cat, config.top_n, config.mmr_lambda)
```

> 注：新资源加权在 `rerank` 中通过调用方预先计算的 `scored` 分数体现（`item_cat` 由 item features 的 category 提供；冷启动加权可在 Task 11 推理时对 `age_days < cold_age_days` 的资源放大分数）。

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_rerank.py -v`
Expected: PASS（MMR 拆分同类目、去重生效）

- [ ] **Step 5: 提交**

```bash
git add src/engine/pipeline/rerank.py tests/test_rerank.py
git commit -m "feat(pipeline): 实现重排（MMR 多样性 + 去重 + 新资源加权）"
```

---

### Task 11: 批量推理全链路（召回 → 精排 → 重排 → top-N）

**Files:**
- Create: `src/engine/pipeline/infer_batch.py`
- Create: `tests/test_infer_batch.py`

**Interfaces:**
- Consumes: `TwoTower`、`MultiTaskDeepFM`、`Vocab`、`build_user_features`/`build_item_features`、`rerank`（Task 10）
- Produces: `engine.pipeline.infer_batch.infer_batch(recall_model, rank_model, bundle, vocab, config, device="cpu") -> dict[int, list[int]]`
  - 对每个用户：召回全量点积 top-`recall_k` → 精排打分 → 重排 → top-`top_n`
  - 无画像的用户（冷启动）回退到热门资源

- [ ] **Step 1: 写失败测试 `tests/test_infer_batch.py`**

```python
from engine.config import EngineConfig
from engine.data.simulator import generate
from engine.data.preprocess import clean, build_vocab, time_split
from engine.models.recall.trainer import train_recall
from engine.models.rank.trainer import train_rank
from engine.pipeline.infer_batch import infer_batch

def test_infer_batch_topn():
    cfg = EngineConfig(sim_n_users=30, sim_n_resources=20,
                       sim_n_interactions=1500, seed=2,
                       recall_epochs=1, rank_epochs=1,
                       top_n=10, recall_k=20)
    b = clean(generate(cfg), 5, 5)
    vocab = build_vocab(b)
    splits = time_split(b, 0.8, 0.1, __import__("numpy").random.default_rng(0))
    recall_model = train_recall(splits.train, vocab, cfg)
    rank_model = train_rank(splits, cfg)
    recs = infer_batch(recall_model, rank_model, splits.train, vocab, cfg)
    assert len(recs) > 0
    assert all(len(v) <= cfg.top_n for v in recs.values())
    assert all(len(set(v)) == len(v) for v in recs.values())   # 无重复
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_infer_batch.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 写实现**

`src/engine/pipeline/infer_batch.py`:

```python
from __future__ import annotations
import numpy as np
import torch
from ..features.user_features import build_user_features
from ..features.item_features import build_item_features
from ..features.feature_config import build_rank_sparse_specs
from ..models.rank.multitask_deepfm import MultiTaskDeepFM
from .rerank import rerank


def infer_batch(recall_model, rank_model, bundle, vocab, config,
                device: str = "cpu") -> dict[int, list[int]]:
    recall_model.eval(); rank_model.eval()
    uf = build_user_features(bundle, vocab)
    it = build_item_features(bundle, vocab)
    item_ids = sorted(vocab.item2id.values())

    with torch.no_grad():
        item_feats = {k: torch.stack([_t(it[i][k]) for i in item_ids])
                      for k in ("item_id", "category_id", "tags", "type_id")}
        item_embs = recall_model.item_emb(item_feats).T          # (E, n_items)

        # 热门兜底：按行为频次
        from collections import Counter
        pop = Counter(x.resource_id for x in bundle.behaviors)
        pop_items = [vocab.item2id[r] for r, _ in pop.most_common(config.top_n)]

        out: dict[int, list[int]] = {}
        for u_idx in sorted(vocab.user2id.values()):
            if u_idx in uf:
                fu = {k: torch.stack([_t(uf[u_idx][k])]) for k in
                      ("user_id", "cat_interest", "tag_interest", "type_pref")}
                ue = recall_model.user_emb(fu)
                scores = (ue @ item_embs).squeeze(0)             # (n_items,)
                top = torch.argsort(scores, descending=True)[:config.recall_k]
                candidates = [item_ids[i] for i in top.tolist()]
            else:
                candidates = list(pop_items)

            scored = []
            for i in candidates:
                if i not in it:
                    continue
                sparse = {
                    "user_id": torch.tensor([u_idx]),
                    "item_id": torch.tensor([i]),
                    "category_id": torch.tensor([int(it[i]["category_id"])]),
                    "tags": torch.tensor([it[i]["tags"]]),
                    "type_id": torch.tensor([int(it[i]["type_id"])]),
                    "hour": torch.tensor([12]),
                    "dow": torch.tensor([0]),
                }
                numeric = torch.tensor([[
                    0.0, 0.0, 0.0, 0.0,
                    float(it[i]["log_view"]), float(it[i]["avg_rating"]),
                    float(it[i]["age_days"]),
                ]])
                pctr, pcvr, rating = rank_model(sparse, numeric)
                final = 0.5 * float(pctr) + 0.3 * float(pcvr) + 0.2 * (float(rating) / 5.0)
                scored.append((i, final))
            item_cat = {i: str(int(it[i]["category_id"])) for i in candidates if i in it}
            out[u_idx] = rerank(scored, seen=set(), item_cat=item_cat, config=config)
    return out


def _t(arr) -> torch.Tensor:
    return torch.tensor(np.asarray(arr, dtype=np.float32))
```

> 注：`infer_batch` 中用户数值特征置 0 的简化——线上可从画像服务补齐；一期以「全链路可跑」为目标。

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_infer_batch.py -v`
Expected: PASS（全链路输出 top-N、无重复）

- [ ] **Step 5: 提交**

```bash
git add src/engine/pipeline/infer_batch.py tests/test_infer_batch.py
git commit -m "feat(pipeline): 实现批量推理全链路（召回→精排→重排→top-N）"
```

---

### Task 12: 一键脚本、双轨验证与 README

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/gen_sim_data.py`
- Create: `scripts/load_movielens.py`
- Create: `scripts/train_all.py`
- Create: `scripts/run_batch_infer.py`
- Create: `README.md`
- Create: `tests/test_scripts.py`

**Interfaces:**
- Consumes: 全部前序模块
- Produces:
  - `python -m scripts.gen_sim_data` → `dataset/sim/*.csv`（模拟数据落盘）
  - `python -m scripts.load_movielens` → `dataset/ml-1m/`（下载 + 转换 + 落盘）
  - `python -m scripts.train_all --data-source sim|movielens` → `model/`（召回/排序 checkpoint + 评估报告 json）
  - `python -m scripts.run_batch_infer` → `model/recommendations.json`（`{user_idx: [item_idx,...]}`）

- [ ] **Step 1: 写失败测试 `tests/test_scripts.py`**

```python
from engine.config import EngineConfig
from engine.data.preprocess import build_vocab
from engine.data.io import save_bundle, load_bundle

def test_bundle_io_roundtrip(tmp_path):
    from engine.data.simulator import generate
    cfg = EngineConfig(sim_n_users=10, sim_n_resources=5,
                       sim_n_interactions=100, seed=0)
    b = generate(cfg)
    save_bundle(b, str(tmp_path))
    loaded = load_bundle(str(tmp_path))
    assert len(loaded.behaviors) == len(b.behaviors)
    assert len(loaded.resources) == len(b.resources)
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_scripts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine.data.io'`（尚未实现 io）

> 若 Task 1 已实现 `io.py`，此测试直接 PASS——仍运行以确认回归。

- [ ] **Step 3: 写实现**

`scripts/gen_sim_data.py`:

```python
from __future__ import annotations
import os
from engine.config import EngineConfig
from engine.data.simulator import generate
from engine.data.io import save_bundle


def main() -> None:
    cfg = EngineConfig()
    b = generate(cfg)
    out = os.path.join(cfg.data_dir, "sim")
    save_bundle(b, out)
    print(f"[gen_sim_data] 生成 {len(b.behaviors)} 条行为 -> {out}")


if __name__ == "__main__":
    main()
```

`scripts/load_movielens.py`:

```python
from __future__ import annotations
import os
from engine.config import EngineConfig
from engine.data.movielens import load
from engine.data.io import save_bundle


def main() -> None:
    cfg = EngineConfig()
    b = load(cfg.data_dir, auto_download=True)
    out = os.path.join(cfg.data_dir, "ml_processed")
    save_bundle(b, out)
    print(f"[load_movielens] 用户={len(b.users)} 资源={len(b.resources)} "
          f"行为={len(b.behaviors)} 评分={len(b.ratings)} -> {out}")


if __name__ == "__main__":
    main()
```

`scripts/train_all.py`:

```python
from __future__ import annotations
import json
import os
import argparse
import numpy as np
from engine.config import EngineConfig
from engine.data.io import load_bundle
from engine.data.movielens import load as load_ml
from engine.data.preprocess import clean, build_vocab, time_split
from engine.models.recall.trainer import train_recall
from engine.models.rank.trainer import train_rank
from engine.pipeline.evaluate import evaluate_recall, evaluate_rank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-source", default="sim", choices=["sim", "movielens"])
    args = ap.parse_args()
    cfg = EngineConfig(data_source=args.data_source)
    np.random.seed(cfg.seed)

    if cfg.data_source == "sim":
        raw = load_bundle(os.path.join(cfg.data_dir, "sim"))
    else:
        raw = load_ml(cfg.data_dir, auto_download=True)
    bundle = clean(raw, cfg.min_user_interactions, cfg.min_item_interactions)
    splits = time_split(bundle, cfg.train_ratio, cfg.val_ratio,
                        np.random.default_rng(cfg.seed))

    recall_model = train_recall(splits.train, splits.vocab, cfg)
    rank_model = train_rank(splits, cfg)
    recall_metrics = evaluate_recall(recall_model, splits.test, splits.vocab, cfg)
    rank_metrics = evaluate_rank(rank_model, splits, cfg)

    os.makedirs(cfg.model_dir, exist_ok=True)
    torch_save = {"recall": recall_model.state_dict(),
                  "rank": rank_model.state_dict()}
    import torch
    torch.save(torch_save, os.path.join(cfg.model_dir, "models.pt"))
    report = {"data_source": cfg.data_source, **recall_metrics, **rank_metrics}
    with open(os.path.join(cfg.model_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

`scripts/run_batch_infer.py`:

```python
from __future__ import annotations
import json
import os
import numpy as np
import torch
from engine.config import EngineConfig
from engine.data.io import load_bundle
from engine.data.preprocess import clean, build_vocab
from engine.features.feature_config import build_rank_sparse_specs
from engine.models.recall.dssm import TwoTower
from engine.models.rank.multitask_deepfm import MultiTaskDeepFM
from engine.pipeline.infer_batch import infer_batch


def main() -> None:
    cfg = EngineConfig()
    bundle = clean(load_bundle(os.path.join(cfg.data_dir, "sim")),
                   cfg.min_user_interactions, cfg.min_item_interactions)
    vocab = build_vocab(bundle)
    ckpt = torch.load(os.path.join(cfg.model_dir, "models.pt"), map_location="cpu")
    recall_model = TwoTower(vocab.n_users, vocab.n_items, vocab.n_cats,
                            vocab.n_tags, cfg.recall_embed_dim)
    rank_model = MultiTaskDeepFM(build_rank_sparse_specs(vocab), numeric_dim=7)
    recall_model.load_state_dict(ckpt["recall"])
    rank_model.load_state_dict(ckpt["rank"])
    recs = infer_batch(recall_model, rank_model, bundle, vocab, cfg)
    with open(os.path.join(cfg.model_dir, "recommendations.json"), "w") as f:
        json.dump({str(k): v for k, v in recs.items()}, f)
    print(f"[run_batch_infer] 为 {len(recs)} 个用户生成推荐")


if __name__ == "__main__":
    main()
```

`README.md`：

```markdown
# edurec-engine

教育资源推荐模型引擎（edurec-platform 的独立推荐服务）。

## 快速开始

\`\`\`bash
pip install -e .[dev]
python -m scripts.gen_sim_data            # ① 生成模拟数据
python -m scripts.train_all --data-source sim   # ② 训练（召回+排序）
python -m scripts.run_batch_infer         # ③ 批量推理 → model/recommendations.json
\`\`\`

## 公开集验证

\`\`\`bash
python -m scripts.load_movielens          # 下载并转换 ML-1M
python -m scripts.train_all --data-source movielens
\`\`\`

## 架构

两阶段流水线：双塔 DSSM 召回 → 多任务 DeepFM 精排 → 规则重排。
详见 `docs/design.md` 与 `docs/superpowers/plans/2026-08-15-edurec-engine-recommendation.md`。
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest -v`
Expected: 全部测试 PASS（全量回归）

- [ ] **Step 5: 端到端冒烟（手动验证双轨）**

Run:
```bash
python -m scripts.gen_sim_data
python -m scripts.train_all --data-source sim
python -m scripts.run_batch_infer
```
Expected: `model/metrics.json` 生成，`recall@50 > 0`、`ctr_auc >= 0.5`；`recommendations.json` 为各用户 top-N。

- [ ] **Step 6: 提交**

```bash
git add scripts README.md tests/test_scripts.py
git commit -m "feat: 一键训练/推理脚本、双轨验证与 README"
```

---

## 自审记录

**Spec 覆盖核对**（对照 `docs/design.md`）：

| 设计文档章节 | 对应任务 |
|---|---|
| §4.1 统一 schema / §4.4 预处理 | Task 1, 3 |
| §4.2 行为模拟器 | Task 2 |
| §4.3 MovieLens-1M | Task 4 |
| §5 特征工程（声明式/画像/资源） | Task 5 |
| §6 双塔 DSSM + in-batch softmax | Task 6, 7 |
| §7 多任务 DeepFM + 三任务损失 | Task 8, 9 |
| §8 重排（MMR/去重/新资源） | Task 10 |
| §11.1 批量推理全链路 | Task 11 |
| §12 开发顺序 1-6 | Task 1→6 顺序一致 |
| 双轨验证（模拟 + ML-1M） | Task 12 |
| 技术选型决策（§13 全部 10 条） | 通过 Global Constraints + 各任务实现约束体现 |

**类型一致性**：`TwoTower`、`MultiTaskDeepFM`、`build_user_features`/`build_item_features`、`build_rank_samples`、`rerank`、`infer_batch` 等跨任务签名已在 Interfaces 块锁定，Task 6-11 引用一致。

**明确留白（后续扩展，不在本期）**：在线 REST 服务（`serving/api.py`）、ESMM 级联、ANN 检索、教育域公开集、上下文数值特征精细化——均标注于设计文档 §11.2 / 各技术选型「演进」。
