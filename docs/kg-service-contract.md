# 知识生成服务契约（可选在线能力，v1）

本文是 **engine 侧**对 `scripts/kg_server.py` 的规范性说明。它独立于
[`platform-contract.md`](platform-contract.md) 的「快照进、排名出」主契约：

- 主契约是**离线批量**，是推荐结果的唯一来源；
- 本契约是**可选在线能力**，只服务「标题/描述 → 概念」的自动生成；
- engine 未启动时，平台只失去「自动生成概念」，推荐、搜索、学习路径不受影响。

机器可读的请求/响应形状见
[`../contracts/v1/kg-service.schema.json`](../contracts/v1/kg-service.schema.json)，
样本见 [`../contracts/v1/fixtures/kg-generate.response.json`](../contracts/v1/fixtures/kg-generate.response.json)。

## 启动

```bash
# engine 仓库根目录；model/ 与依赖需先准备好（model/ 不入库）
pip install -e .[kg]          # rdflib + scikit-learn（服务最小依赖）
.venv/bin/python scripts/kg_server.py --host 127.0.0.1 --port 8090
```

平台侧配置：

```yaml
knowledge:
  generator_url: "http://127.0.0.1:8090"
  generator_timeout_seconds: 180
```

## 接口

### `GET /health`

```json
{
  "status": "ok",
  "engine": "edurec-engine",
  "version": "1.0",
  "contract_version": 1,
  "capabilities": ["knowledge/generate", "knowledge/generate_batch"]
}
```

### `POST /knowledge/generate`

请求：

```json
{"title": "数学分析", "description": "极限、导数与积分", "point_code": "mathematical-analysis"}
```

- `title` 必填、非空；`description` / `point_code` 可选。
- `Content-Type: application/json`，请求体必须是合法 UTF-8 JSON。

成功响应（HTTP 200）：

```json
{
  "contract_version": 1,
  "point_code": "mathematical-analysis",
  "title": "数学分析",
  "concepts": [
    {"name": "积分", "page_rank": 0.045},
    {"name": "导数", "page_rank": 0.042}
  ],
  "elapsed_seconds": 1.2
}
```

`concepts` 已按 `page_rank` 降序、去重、过滤停用词与过短名称；默认最多 30 个。

### `POST /knowledge/generate_batch`

请求：

```json
{"items": [{"title": "…", "description": "…", "point_code": "…"}]}
```

成功响应（HTTP 200）：

```json
{
  "contract_version": 1,
  "results": [
    {"contract_version": 1, "point_code": "…", "title": "…", "concepts": []},
    {"point_code": "…", "title": "…", "concepts": [], "error": "ValueError: …"}
  ]
}
```

单条失败不阻断整批：失败项带 `error`、`concepts` 为空数组，HTTP 仍为 200。

## 错误语义

| HTTP | 场景 | 响应 |
|---|---|---|
| 400 | 请求体为空/非 JSON、`title` 为空、`items` 非非空数组 | `{"error": "…"}` |
| 404 | 未知路径 | `{"error": "not found"}` |
| 500 | 生成内部错误（Wikifier/模型/IO） | `{"error": "…"}` |

平台侧把非 2xx 视为失败，只影响当前生成任务；不会回滚已有概念。

## 版本与兼容

- `contract_version` 当前为 `1`，出现在 `/health` 与每个响应顶层。
- **允许**新增可选字段（平台忽略未知字段）、调整概念数量/排序质量。
- **破坏性变更**（删除字段、改类型、改路径、改错误语义）必须提升
  `contract_version`；平台读到不认识的版本应降级为「不自动生成」，不得猜测。
- 该服务**无状态**、不落库；幂等性由平台按「学科 + 概念名」去重保证。
  同一请求重复调用不保证返回完全相同的排序（Wikifier 在线 API 可能波动）。
- 这是唯一允许的 platform → engine 在线调用。除此之外两仓库不互相调用接口，
  也不读写对方目录（主链路只走文件交接）。
