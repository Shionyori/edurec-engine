# edurec-engine

教育资源推荐模型引擎（edurec-platform 的独立推荐服务）。

两阶段流水线：双塔 DSSM 召回 → 多任务 DeepFM 精排 → 规则重排（MMR 多样性 + 冷启动加权）。

## 快速开始（模拟数据演示）

```bash
# 环境：Python 3.10+，依赖 torch / numpy / pandas / pyyaml
pip install -e .[dev]

python -m scripts.gen_sim_data                  # ① 生成模拟数据 → dataset/sim/
python -m scripts.train_all --data-source sim   # ② 训练 → model/models.pt + metrics.json
python -m scripts.run_batch_infer               # ③ 推理 → model/recommendations.json
```

产物 `model/recommendations.json`：`{"<用户ID>": [<资源ID>, ...]}`，为**输入数据集的原始 ID**
（内部编码已在出口回映射），覆盖数据集**全量用户**（交互过少的冷启动用户走热门兜底）。

## 三种数据源

| 轨道 | 用途 | 数据 | 命令 |
|---|---|---|---|
| `sim` | 流水线演示/联调 | `dataset/sim/`（模拟器生成） | 见上 |
| `movielens` | 真实公开集验证模型有效性 | 自动下载 ML-1M | 见下 |
| `platform` | 平台真实数据接入 | platform 导出的 v1 快照 | 见下 |

### 公开集验证（MovieLens-1M）

```bash
python -m scripts.load_movielens            # 下载并转换 → dataset/ml_processed/
python -m scripts.train_all --data-source movielens
```

### 平台真实数据回路（platform 对接）

前置：platform 侧 `export_snapshot` 导出快照后，手动拷到本仓库：

```bash
# platform 交接（见 edurec-platform/docs/data-handoff.md）：
#   cp -r <platform>/backend/data/snapshots/<run_id> dataset/platform_snapshot/<run_id>

ENGINE_SNAPSHOT_DIR=dataset/platform_snapshot/<run_id> \
  python -m scripts.train_all --data-source platform     # 训练
ENGINE_SNAPSHOT_DIR=dataset/platform_snapshot/<run_id> \
  python -m scripts.run_batch_infer                       # 推理
```

- `train_all` 亦支持 `--snapshot-dir <dir>`；`run_batch_infer` 亦可使用环境变量
  `ENGINE_DATA_SOURCE` / `ENGINE_SNAPSHOT_DIR`（优先级高于配置默认值）。
- 快照格式：`meta.json`（`contract_version=1`）+ `users/resources/categories/behaviors/ratings.csv`，
  loader 见 `src/engine/data/platform.py`。
- **训练与推理必须使用同一份快照（同一 `<run_id>`）**，否则词表与模型不匹配。

## 目录与产物约定

| 目录/文件 | 说明 |
|---|---|
| `dataset/` | 各数据源数据（sim、ml_processed、platform_snapshot/<run_id>/） |
| `model/models.pt` | 模型权重（recall + rank state_dict） |
| `model/metrics.json` | 训练/评估指标（按数据源覆盖） |
| `model/recommendations.json` | 批量推理结果（原始 ID） |

`dataset/`、`model/` 不入库（见 `.gitignore`）。**engine 只读写本仓库目录**，
与 platform 的文件交接一律手动拷贝（方向见 edurec-platform `docs/data-handoff.md`）。

## 配置

- 默认配置：`src/engine/config.py` 的 `EngineConfig`（dataclass，含随机种子、阈值、
  召回/排序超参、`top_n=20` 等），支持 yaml 加载/导出。
- 全流程固定随机种子（默认 42），同数据可复现。

## 测试

```bash
python -m pytest tests   # 数据/特征/模型/流水线/loader 单测
```

## 文档

- `docs/design.md` —— 架构与决策记录
- `docs/superpowers/plans/2026-08-15-edurec-engine-recommendation.md` —— 原始方案
- 与 platform 的对接契约与交接手册见 edurec-platform：`docs/engine-integration-v2.md`、`docs/data-handoff.md`
