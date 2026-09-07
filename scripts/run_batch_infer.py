from __future__ import annotations
import argparse
import json
import os

import numpy as np
import torch

from engine.config import EngineConfig
from engine.data.io import load_bundle
from engine.data.platform import load as load_platform
from engine.data.preprocess import clean, build_vocab
from engine.features.feature_config import build_rank_sparse_specs
from engine.models.recall.dssm import TwoTower
from engine.models.rank.multitask_deepfm import MultiTaskDeepFM
from engine.pipeline.infer_batch import infer_batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-source", default=None,
                    choices=["sim", "movielens", "platform"])
    ap.add_argument("--snapshot-dir", default=None,
                    help="platform 快照目录（data_source=platform 时）")
    args = ap.parse_args()
    cfg = EngineConfig()
    if args.data_source is not None:
        cfg.data_source = args.data_source
    elif os.environ.get("ENGINE_DATA_SOURCE"):
        cfg.data_source = os.environ["ENGINE_DATA_SOURCE"]
    if args.snapshot_dir is not None:
        cfg.snapshot_dir = args.snapshot_dir
    elif os.environ.get("ENGINE_SNAPSHOT_DIR"):
        cfg.snapshot_dir = os.environ["ENGINE_SNAPSHOT_DIR"]
    if cfg.data_source == "platform":
        if not cfg.snapshot_dir:
            raise SystemExit("data_source=platform 需要配置 snapshot_dir")
        raw = load_platform(cfg.snapshot_dir)
    else:
        raw = load_bundle(os.path.join(cfg.data_dir, "sim"))
    all_users = sorted(u.user_id for u in raw.users)   # 覆盖全量用户（含冷启动）
    bundle = clean(raw, cfg.min_user_interactions, cfg.min_item_interactions)
    vocab = build_vocab(bundle)
    ckpt = torch.load(os.path.join(cfg.model_dir, "models.pt"), map_location="cpu")
    recall_model = TwoTower(vocab.n_users, vocab.n_items, vocab.n_cats,
                            vocab.n_tags, cfg.recall_embed_dim)
    rank_model = MultiTaskDeepFM(build_rank_sparse_specs(vocab), numeric_dim=7)
    recall_model.load_state_dict(ckpt["recall"])
    rank_model.load_state_dict(ckpt["rank"])
    recs = infer_batch(recall_model, rank_model, bundle, vocab, cfg,
                       target_users=all_users)
    with open(os.path.join(cfg.model_dir, "recommendations.json"), "w",
              encoding="utf-8") as f:
        json.dump({str(k): v for k, v in recs.items()}, f)
    print(f"[run_batch_infer] 为 {len(recs)}/{len(all_users)} 个用户生成推荐"
          f"(key/value 均为数据集原始 ID) -> "
          f"{os.path.join(cfg.model_dir, 'recommendations.json')}")


if __name__ == "__main__":
    main()
