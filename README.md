# edurec-engine

教育资源推荐模型引擎（edurec-platform 的独立推荐服务）。

## 快速开始

```bash
pip install -e .[dev]
python -m scripts.gen_sim_data            # ① 生成模拟数据
python -m scripts.train_all --data-source sim   # ② 训练（召回+排序）
python -m scripts.run_batch_infer         # ③ 批量推理 → model/recommendations.json
```

## 公开集验证

```bash
python -m scripts.load_movielens          # 下载并转换 ML-1M
python -m scripts.train_all --data-source movielens
```

## 架构

两阶段流水线：双塔 DSSM 召回 → 多任务 DeepFM 精排 → 规则重排。
详见 `docs/design.md` 与 `docs/superpowers/plans/2026-08-15-edurec-engine-recommendation.md`。
