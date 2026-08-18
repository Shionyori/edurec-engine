from engine.config import EngineConfig
from engine.data.simulator import generate
from engine.data.preprocess import clean, build_vocab, time_split
from engine.models.rank.trainer import train_rank
from engine.pipeline.evaluate import rmse, evaluate_rank


def test_rmse():
    assert abs(rmse([3.0, 4.0], [3.0, 4.0])) < 1e-6


def test_train_rank_learns():
    cfg = EngineConfig(sim_n_users=40, sim_n_resources=25,
                       sim_n_interactions=2000, seed=2,
                       rank_epochs=2, rank_batch_size=64)
    b = clean(generate(cfg), 5, 5)
    vocab = build_vocab(b)
    splits = time_split(b, 0.8, 0.1, __import__("numpy").random.default_rng(0))
    model = train_rank(splits, cfg, device="cpu")
    m = evaluate_rank(model, splits, cfg)
    assert m["ctr_auc"] >= 0.5
