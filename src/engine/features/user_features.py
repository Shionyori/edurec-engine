from __future__ import annotations
from collections import Counter, defaultdict
import numpy as np
from ..data.schema import DataBundle


def build_user_features(bundle: DataBundle, vocab) -> dict[int, dict[str, np.ndarray]]:
    """画像特征：只从传入的 bundle 统计（防泄漏的关键）。

    返回 dict 以**编码用户 ID**（vocab.user2id 的值）为 key，与召回/排序
    训练及推理侧消费方一致；统计时用 vocab 把原始用户/资源 ID 映射为编码 ID，
    避免清洗后「原始 ID == 编码 ID」的隐式假设失效导致错配。
    """
    cat_hits: dict[int, Counter] = defaultdict(Counter)
    tag_hits: dict[int, Counter] = defaultdict(Counter)
    type_hits: dict[int, Counter] = defaultdict(Counter)
    n_views: Counter = Counter()
    n_favs: Counter = Counter()
    last_ts: dict[int, int] = {}
    active_days: dict[int, set[int]] = defaultdict(set)
    cat_of = {r.resource_id: r.category_id for r in bundle.resources}
    tags_of = {r.resource_id: r.tags for r in bundle.resources}
    type_of = {r.resource_id: vocab.type2id[r.type] for r in bundle.resources}

    for x in bundle.behaviors:
        if x.user_id not in vocab.user2id or x.resource_id not in vocab.item2id:
            continue                      # 不在词表（如被清洗掉的）不参与画像
        u = vocab.user2id[x.user_id]
        cat_hits[u][cat_of[x.resource_id]] += 1
        for t in tags_of.get(x.resource_id, ()):
            if t in vocab.tag2id:
                tag_hits[u][vocab.tag2id[t]] += 1
        type_hits[u][type_of[x.resource_id]] += 1
        n_views[u] += 1
        if x.action == "favorite":
            n_favs[u] += 1
        last_ts[u] = max(last_ts.get(u, 0), x.ts)
        active_days[u].add(x.ts // 86400)

    max_ts = max(last_ts.values(), default=0)

    def _norm(counter: Counter, n: int) -> np.ndarray:
        v = np.zeros(n)
        for k, c in counter.items():
            if 0 <= k < n:
                v[k] = c
        s = v.sum()
        return v / s if s > 0 else v

    out: dict[int, dict[str, np.ndarray]] = {}
    for u in vocab.user2id.values():
        out[u] = {
            "user_id": np.array(u, dtype=np.int64),
            "cat_interest": _norm(cat_hits[u], vocab.n_cats),
            "tag_interest": _norm(tag_hits[u], vocab.n_tags),
            "type_pref": _norm(type_hits[u], 3),
            "active_days": np.array(len(active_days[u]), dtype=np.float32),
            "n_views": np.array(n_views[u], dtype=np.float32),
            "n_favs": np.array(n_favs[u], dtype=np.float32),
            "gap_days": np.array(
                (max_ts - last_ts[u]) / 86400.0 if u in last_ts else 0.0,
                dtype=np.float32),
        }
    return out
