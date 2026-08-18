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
