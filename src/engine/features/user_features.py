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

    def _norm(counter: Counter, n: int) -> np.ndarray:
        v = np.zeros(n)
        for k, c in counter.items():
            if 0 <= k < n:
                v[k] = c
        s = v.sum()
        return v / s if s > 0 else v

    out: dict[int, dict[str, np.ndarray]] = {}
    for u in vocab.user2id.values():
        if u not in cat_hits:
            out[u] = _zero_user(vocab)
            continue
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
