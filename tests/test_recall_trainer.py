from engine.config import EngineConfig
from engine.data.simulator import generate
from engine.data.preprocess import clean, build_vocab
from engine.models.recall.trainer import train_recall
from engine.pipeline.evaluate import evaluate_recall


def test_train_recall_overfits_small():
    cfg = EngineConfig(sim_n_users=40, sim_n_resources=25,
                       sim_n_interactions=2000, seed=2,
                       recall_epochs=2, recall_batch_size=128)
    b = clean(generate(cfg), 5, 5)
    vocab = build_vocab(b)
    model = train_recall(b, vocab, cfg, device="cpu")
    assert hasattr(model, "user_emb")
    scores = evaluate_recall(model, b, vocab, cfg)
    assert scores["recall@50"] > 0.0
