from engine.config import EngineConfig
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
    splits = time_split(b, 0.8, 0.1, __import__("numpy").random.default_rng(0))
    recall_model = train_recall(splits.train, vocab, cfg)
    rank_model = train_rank(splits, cfg)
    recs = infer_batch(recall_model, rank_model, splits.train, vocab, cfg)
    assert len(recs) > 0
    assert all(len(v) <= cfg.top_n for v in recs.values())
    assert all(len(set(v)) == len(v) for v in recs.values())   # 无重复
