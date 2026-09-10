from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
from .schema import DataBundle


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
    cat2id = {c: i for i, c in enumerate(
        sorted({r.category_id for r in bundle.resources}))}
    tag_set = {t for r in bundle.resources for t in r.tags}
    tag2id = {t: i for i, t in enumerate(sorted(tag_set))}
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
    """构建正负样本。注意：负样本数可能少于 neg_per_pos * 正样本数，
    因为随机采到的未交互对若与正样本或已采样负样本冲突会被跳过。"""
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
            samples.append(RankSample(u, i, 0, 0, None,
                                      hour=int(rng.integers(0, 24)),
                                      dow=int(rng.integers(0, 7))))
    return samples
