from __future__ import annotations
import argparse
import json
import os

import numpy as np
import torch

from engine.config import EngineConfig
from engine.data.io import load_bundle
from engine.data.movielens import load as load_ml
from engine.data.platform import load as load_platform
from engine.data.preprocess import clean, time_split
from engine.models.recall.trainer import train_recall
from engine.models.rank.trainer import train_rank
from engine.pipeline.evaluate import evaluate_recall, evaluate_rank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-source", default="sim",
                    choices=["sim", "movielens", "platform"])
    args = ap.parse_args()
    cfg = EngineConfig(data_source=args.data_source)
    if args.snapshot_dir:
        cfg.snapshot_dir = args.snapshot_dir
    np.random.seed(cfg.seed)

    if cfg.data_source == "sim":
        raw = load_bundle(os.path.join(cfg.data_dir, "sim"))
    else:
        raw = load_ml(cfg.data_dir, auto_download=True)
    bundle = clean(raw, cfg.min_user_interactions, cfg.min_item_interactions)
    splits = time_split(bundle, cfg.train_ratio, cfg.val_ratio,
                        np.random.default_rng(cfg.seed))

    recall_model = train_recall(splits.train, splits.vocab, cfg)
    rank_model = train_rank(splits, cfg)
    recall_metrics = evaluate_recall(recall_model, splits.test, splits.vocab, cfg)
    rank_metrics = evaluate_rank(rank_model, splits, cfg)

    os.makedirs(cfg.model_dir, exist_ok=True)
    torch.save({"recall": recall_model.state_dict(),
                "rank": rank_model.state_dict()},
               os.path.join(cfg.model_dir, "models.pt"))
    report = {"data_source": cfg.data_source, **recall_metrics, **rank_metrics}
    with open(os.path.join(cfg.model_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
