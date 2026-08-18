from __future__ import annotations
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from ...features.user_features import build_user_features
from ...features.item_features import build_item_features
from ...data.preprocess import build_recall_pairs
from .dssm import TwoTower

# 特征按 dtype 分组：categorical 用作 Embedding 索引（long），multi_hot 是稠密向量（float）
_USER_INDEX = {"user_id"}
_USER_VEC = {"cat_interest", "tag_interest", "type_pref"}
_ITEM_INDEX = {"item_id", "category_id", "type_id"}
_ITEM_VEC = {"tags"}


def _batch(feats, ids, index_keys, vec_keys, device: str) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k in index_keys:
        out[k] = torch.tensor(
            np.asarray([feats[i][k] for i in ids], dtype=np.int64),
            dtype=torch.long, device=device)
    for k in vec_keys:
        out[k] = torch.tensor(
            np.asarray([feats[i][k] for i in ids], dtype=np.float32),
            dtype=torch.float32, device=device)
    return out


def train_recall(bundle, vocab, config, device: str = "cpu") -> TwoTower:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = TwoTower(vocab.n_users, vocab.n_items, vocab.n_cats, vocab.n_tags,
                     config.recall_embed_dim).to(device)

    uf = build_user_features(bundle, vocab)
    it = build_item_features(bundle, vocab)
    pairs = build_recall_pairs(bundle, vocab)
    # 热门资源（按交互频次），用于显式负采样
    pop = Counter(x.resource_id for x in bundle.behaviors)
    pop_items = [vocab.item2id[i] for i, _ in pop.most_common(config.recall_batch_size)]
    if not pop_items:
        pop_items = sorted(vocab.item2id.values())

    u = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    i = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    dataset = TensorDataset(u, i)
    loader = DataLoader(dataset, batch_size=config.recall_batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=config.recall_lr)
    for epoch in range(config.recall_epochs):
        for ub, ib in loader:
            ub, ib = ub.to(device), ib.to(device)
            ub_l, ib_l = ub.tolist(), ib.tolist()
            # 每个 user 配一个热门显式负样本
            neg = [int(rng.choice(pop_items)) for _ in range(len(ub_l))]
            users = _batch(uf, ub_l, _USER_INDEX, _USER_VEC, device)
            pos = _batch(it, ib_l, _ITEM_INDEX, _ITEM_VEC, device)
            neg_feats = _batch(it, neg, _ITEM_INDEX, _ITEM_VEC, device)
            loss = model.forward(users, pos, neg_feats, tau=config.recall_tau)
            opt.zero_grad(); loss.backward(); opt.step()
    return model.cpu()
