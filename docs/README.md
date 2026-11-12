# 文档索引

## 规范（改契约先看这里）

| 文档 | 内容 |
|---|---|
| [`platform-contract.md`](platform-contract.md) | 快照进 / 排名出的主契约（权威） |
| [`kg-service-contract.md`](kg-service-contract.md) | 知识生成 HTTP 服务（可选在线能力） |
| [`data-flow.md`](data-flow.md) | 当前实现的数据流、命令、失效与降级 |
| [`../contracts/v1/README.md`](../contracts/v1/README.md) | 契约的可执行版本：schema + fixtures + 校验器 |

## 模型与实验

| 文档 | 内容 |
|---|---|
| [`design.md`](design.md) | 语义双塔 + 质量融合的设计与决策 |
| [`architecture-review.md`](architecture-review.md) | 分层/数据访问/缓存/配置的评估、重构记录与路线图 |
| [`gaussian-recall-status.md`](gaussian-recall-status.md) | 概率式扩展的负面结论与后续方向 |
| [`references.md`](references.md) | 引用来源（CourseHub / EMBEDD-ER / 数据集） |

## 代码边界

- `src/engine/`：推荐引擎（训练/推理）。
- `src/kg/`：知识图谱/概念生成的生产路径。
- `src/kg/research/`：EMBEDD-ER 研究脚本，不在运行时路径（见 `src/kg/README.md`）。

> 历史计划、平台设计存档、论文原文抽取文本已移除；需要时从 git 历史取。
