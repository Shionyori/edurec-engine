from collections import Counter

from engine.config import EngineConfig
from engine.data.simulator import generate

def test_generate_schema_and_scale():
    cfg = EngineConfig(sim_n_users=100, sim_n_resources=50,
                       sim_n_interactions=5000, seed=1)
    b = generate(cfg)
    assert len(b.users) == cfg.sim_n_users
    assert len(b.resources) == cfg.sim_n_resources
    assert len(b.behaviors) > 0
    assert all(x.action in ("view", "click", "favorite") for x in b.behaviors)
    assert all(1 <= r.score <= 5 for r in b.ratings)
    assert len(b.ratings) <= len(b.behaviors)

def test_generate_reproducible_with_seed():
    cfg = EngineConfig(sim_n_users=50, sim_n_resources=20, sim_n_interactions=1000, seed=7)
    a = generate(cfg)
    b = generate(cfg)
    assert [(x.user_id, x.resource_id, x.action, x.ts) for x in a.behaviors] == \
           [(x.user_id, x.resource_id, x.action, x.ts) for x in b.behaviors]

def test_generate_behavior_volume_target():
    cfg = EngineConfig(sim_n_users=100, sim_n_resources=50,
                       sim_n_interactions=5000, seed=1)
    b = generate(cfg)
    assert len(b.behaviors) >= int(cfg.sim_n_interactions * 0.8)

def test_generate_hidden_structure_skew():
    cfg = EngineConfig(sim_n_users=100, sim_n_resources=50,
                       sim_n_interactions=5000, seed=1)
    b = generate(cfg)
    counts = Counter(x.resource_id for x in b.behaviors)
    ranked = sorted(counts.values(), reverse=True)
    top20 = max(1, int(len(ranked) * 0.2))
    assert sum(ranked[:top20]) / sum(ranked) > 0.4

def test_generate_with_few_tags_does_not_raise():
    cfg = EngineConfig(sim_n_users=50, sim_n_resources=20,
                       sim_n_categories=5, sim_n_tags=2,
                       sim_n_interactions=1000, seed=1)
    b = generate(cfg)
    assert len(b.behaviors) > 0
