import numpy as np
from engine.config import EngineConfig
from engine.data.simulator import generate
from engine.data.preprocess import clean, build_vocab
from engine.features.feature_config import build_user_specs, build_item_specs
from engine.features.user_features import build_user_features
from engine.features.item_features import build_item_features


def _cleaned():
    cfg = EngineConfig(sim_n_users=50, sim_n_resources=30,
                       sim_n_interactions=3000, seed=3)
    return clean(generate(cfg), 5, 5)


def test_specs_dimensions():
    b = _cleaned()
    vocab = build_vocab(b)
    us = build_user_specs(vocab)
    ispec = build_item_specs(vocab)
    assert any(s.name == "user_id" for s in us)
    assert any(s.name == "cat_interest" and s.kind == "multi_hot" for s in us)
    assert any(s.name == "avg_rating" and s.kind == "numeric" for s in ispec)


def test_user_features_normalized():
    b = _cleaned()
    vocab = build_vocab(b)
    uf = build_user_features(b, vocab)
    assert set(uf.keys()) == set(vocab.user2id.values())
    cat = uf[list(uf.keys())[0]]["cat_interest"]
    assert abs(float(cat.sum()) - 1.0) < 1e-6          # 归一化概率向量


def test_item_features_shape():
    b = _cleaned()
    vocab = build_vocab(b)
    it = build_item_features(b, vocab)
    key = list(it.keys())[0]
    assert it[key]["tags"].shape == (vocab.n_tags,)
