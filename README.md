# edurec-engine

教育资源推荐模型引擎（独立仓库）：双塔 DSSM 召回 → 多任务 DeepFM 精排 → MMR 重排。

## 环境与测试

```bash
pip install -e .[dev]
python -m pytest tests
```

## 使用方式

### 配合 platform（日常，无需手动）

engine 是 platform 的离线推荐引擎。日常使用时**无需在本仓库手动操作**——在 platform 仓库执行：

```bash
cd ../edurec-platform
bash scripts/handoff.sh        # 自动：导出快照 → 本仓库训练 + 推理 → 结果回传导入
```

`handoff.sh` 会自动用本仓库的 `.venv` 完成训练与全量推理，并把 `model/recommendations.json` 拷回 platform 导入。

### 独立演示（本仓库单独跑）

```bash
python -m scripts.gen_sim_data
python -m scripts.train_all --data-source sim
python -m scripts.run_batch_infer     # → model/recommendations.json
```

### 平台真实数据（手动，等价于 handoff.sh）

```bash
# 快照由 platform export_snapshot 产出后手动拷入（见 platform docs/data-handoff.md）
ENGINE_SNAPSHOT_DIR=dataset/platform_snapshot/<run_id> python -m scripts.train_all --data-source platform
ENGINE_SNAPSHOT_DIR=dataset/platform_snapshot/<run_id> python -m scripts.run_batch_infer
```

## 说明

- 产物 `recommendations.json`：`{原始用户ID: [原始资源ID,…]}`，覆盖全量用户，冷启动走热门。
- 训练与推理须用**同一份快照**（同一 `<run_id>`）。
- 数据源：`sim` / `movielens` / `platform`；MovieLens 先跑 `python -m scripts.load_movielens`。
- 参数见 `src/engine/config.py`（默认 top_n=20，seed 可复现）。
- `dataset/`、`model/` 不入库；engine 只读写本仓库，与 platform 交接一律手动拷贝。
- 架构详见 `docs/design.md`。
