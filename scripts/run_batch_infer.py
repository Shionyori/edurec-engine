from __future__ import annotations
import json
import os

import numpy as np
import torch

from engine.config import EngineConfig
from engine.data.io import load_bundle
from engine.data.preprocess import clean, build_vocab
from engine.features.feature_config import build_rank_sparse_specs
from engine.models.recall.dssm import TwoTower
from engine.models.rank.multitask_deepfm import MultiTaskDeepFM
from engine.pipeline.infer_batch import infer_batch


def main() -> None:
    cfg = EngineConfig()
    bundle = clean(load_bundle(os.path.join(cfg.data_dir, "sim")),
                   cfg.min_user_interactions, cfg.min_item_interactions)
    vocab = build_vocab(bundle)
    ckpt = torch.load(os.path.join(cfg.model_dir, "models.pt"), map_location="cpu")
    recall_model = TwoTower(vocab.n_users, vocab.n_items, vocab.n_cats,
                            vocab.n_tags, cfg.recall_embed_dim)
    rank_model = MultiTaskDeepFM(build_rank_sparse_specs(vocab), numeric_dim=7)
    recall_model.load_state_dict(ckpt["recall"])
    rank_model.load_state_dict(ckpt["rank"])
    recs = infer_batch(recall_model, rank_model, bundle, vocab, cfg)
    with open(os.path.join(cfg.model_dir, "recommendations.json"), "w",
              encoding="utf-8") as f:
        json.dump({str(k): v for k, v in recs.items()}, f)
    print(f"[run_batch_infer] 为 {len(recs)} 个用户生成推荐 -> "
          f"{os.path.join(cfg.model_dir, 'recommendations.json')}")


if __name__ == "__main__":
    main()
