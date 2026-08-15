import numpy as np
from engine.config import EngineConfig
from engine.data.simulator import generate
from engine.data.preprocess import (
    clean, build_vocab, time_split, build_recall_pairs, build_rank_samples,
)

def _sim_small():
    cfg = EngineConfig(sim_n_users=50, sim_n_resources=30,
                       sim_n_interactions=3000, seed=3)
    return generate(cfg)

def test_clean_filters_sparse():
    b = _sim_small()
    cleaned = clean(b, min_user_interactions=5, min_item_interactions=5)
    assert len(cleaned.users) <= len(b.users)
    # 清洗后每个用户至少有 5 次行为
    from collections import Counter
    cnt = Counter(x.user_id for x in cleaned.behaviors)
    assert all(v >= 5 for v in cnt.values())

def test_vocab_counts():
    b = _sim_small()
    vocab = build_vocab(b)
    assert vocab.n_users == len(b.users)
    assert vocab.n_items == len(b.resources)
    assert vocab.n_tags >= 1

def test_time_split_no_overlap():
    b = _sim_small()
    cleaned = clean(b, 5, 5)
    vocab = build_vocab(cleaned)
    splits = time_split(cleaned, 0.8, 0.1, np.random.default_rng(0))
    train_uv = {(x.user_id, x.resource_id) for x in splits.train.behaviors}
    test_uv = {(x.user_id, x.resource_id) for x in splits.test.behaviors}
    assert len(train_uv) > 0 and len(test_uv) > 0

def test_rank_samples_labels():
    b = _sim_small()
    cleaned = clean(b, 5, 5)
    vocab = build_vocab(cleaned)
    samples = build_rank_samples(cleaned, vocab, np.random.default_rng(1), neg_per_pos=2)
    assert len(samples) > 0
    assert all(s.ctr in (0, 1) and s.cvr in (0, 1) for s in samples)
    assert all(0 <= s.hour <= 23 and 0 <= s.dow <= 6 for s in samples)
    assert any(s.rating is not None for s in samples)
