from engine.config import EngineConfig
from engine.pipeline.rerank import mmr_rerank, rerank


def test_mmr_spreads_categories():
    scored = [(0, 0.9), (1, 0.85), (2, 0.8)]       # item -> score
    cats = {0: "a", 1: "a", 2: "b"}
    out = mmr_rerank(scored, cats, top_n=3, lambda_=0.5)
    # 同类目 0、1 被拆分，2 应提前
    assert out.index(2) < 2


def test_rerank_filters_seen():
    cfg = EngineConfig(top_n=2, mmr_lambda=0.5, cold_age_days=7, cold_weight=1.2)
    scored = [(0, 0.9), (1, 0.8), (2, 0.7)]
    out = rerank(scored, seen={0}, item_cat={1: "a", 2: "a"}, config=cfg)
    assert 0 not in out and len(out) <= 2
