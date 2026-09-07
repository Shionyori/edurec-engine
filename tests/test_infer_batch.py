import numpy as np
from engine.config import EngineConfig
from engine.data.schema import User, Resource, Behavior, Rating, DataBundle
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
    splits = time_split(b, 0.8, 0.1, np.random.default_rng(0))
    recall_model = train_recall(splits.train, vocab, cfg)
    rank_model = train_rank(splits, cfg)
    recs = infer_batch(recall_model, rank_model, splits.train, vocab, cfg)
    assert len(recs) > 0
    assert all(len(v) <= cfg.top_n for v in recs.values())
    assert all(len(set(v)) == len(v) for v in recs.values())   # 无重复


def _gapped_bundle() -> DataBundle:
    """构造「原始 ID 有空洞、不等于内部编码」的数据集，用于验证 ID 往返。"""
    user_ids = [0, 1, 2, 4, 9, 15, 40]          # 原始用户 ID 不连续
    res_ids = [100, 200, 300, 400, 500]         # 原始资源 ID 不连续
    types = ("course", "article", "video")
    rng = np.random.default_rng(7)

    resources = [
        Resource(resource_id=i, type=types[rng.integers(3)],
                 category_id=int(rng.integers(1, 4)), tags=("t",))
        for i in res_ids
    ]
    users = [User(user_id=u) for u in user_ids]

    behaviors: list[Behavior] = []
    ratings: list[Rating] = []
    ts = 1_700_000_000
    for ui, uid in enumerate(user_ids):
        for ri, rid in enumerate(res_ids):
            for rep in range(4):                # 每人每资源 4 次交互，保证 train/val 都有
                action = ("view", "click", "favorite")[(ui + ri + rep) % 3]
                behaviors.append(Behavior(user_id=uid, resource_id=rid,
                                          action=action, ts=ts + (ui * 1000 + ri * 10 + rep)))
                if action == "favorite":
                    ratings.append(Rating(user_id=uid, resource_id=rid,
                                          score=4 + rep % 2, ts=ts))
    return DataBundle(users=users, resources=resources,
                      behaviors=behaviors, ratings=ratings)


def test_infer_batch_returns_original_ids_and_covers_all():
    """推理结果必须使用数据集原始 ID，且每个目标用户都有推荐（ID 往返回归）。"""
    cfg = EngineConfig(recall_epochs=1, rank_epochs=1,
                       top_n=10, recall_k=50)
    b = _gapped_bundle()
    vocab = build_vocab(b)
    splits = time_split(b, 0.8, 0.1, np.random.default_rng(0))
    recall_model = train_recall(splits.train, vocab, cfg)
    rank_model = train_rank(splits, cfg)

    orig_uids = {u.user_id for u in b.users}
    orig_rids = {r.resource_id for r in b.resources}
    recs = infer_batch(recall_model, rank_model, splits.train, vocab, cfg,
                       target_users=sorted(orig_uids))

    # key 必须是原始用户 ID（若退回内部编码 0..6 将不等）
    assert set(recs.keys()) == orig_uids
    for v in recs.values():
        assert v, "每个用户都应有推荐"
        # value 必须是原始资源 ID，而不是内部编码 0..4
        assert all(x in orig_rids for x in v)
        assert len(v) <= cfg.top_n
        assert len(set(v)) == len(v)
