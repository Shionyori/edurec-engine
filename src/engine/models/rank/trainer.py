from __future__ import annotations
import numpy as np
import torch
from ...features.feature_config import build_rank_sparse_specs
from ...features.user_features import build_user_features
from ...features.item_features import build_item_features
from ...data.preprocess import build_rank_samples
from .multitask_deepfm import MultiTaskDeepFM


def train_rank(splits, config, device: str = "cpu") -> MultiTaskDeepFM:
    torch.manual_seed(config.seed)
    vocab = splits.vocab
    specs = build_rank_sparse_specs(vocab)
    numeric_dim = 4 + 3                                    # 用户4 + 资源3
    model = MultiTaskDeepFM(specs, numeric_dim).to(device)

    uf = build_user_features(splits.train, vocab)          # 防泄漏：仅训练集统计
    it = build_item_features(splits.train, vocab)
    samples = build_rank_samples(splits.train, vocab,
                                 np.random.default_rng(config.seed),
                                 config.rank_neg_per_pos)

    def make_batch(batch_samples):
        sparse = {k: [] for k in ("user_id", "item_id", "category_id",
                                  "tags", "type_id", "hour", "dow")}
        numeric = []
        y_ctr, y_cvr, y_rat = [], [], []
        rat_mask = []
        for s in batch_samples:
            u, i = uf.get(s.user_idx), it.get(s.item_idx)
            if u is None or i is None:
                continue
            sparse["user_id"].append(s.user_idx)
            sparse["item_id"].append(s.item_idx)
            sparse["category_id"].append(int(i["category_id"]))
            sparse["tags"].append(i["tags"])
            sparse["type_id"].append(int(i["type_id"]))
            sparse["hour"].append(s.hour)
            sparse["dow"].append(s.dow)
            numeric.append([u["active_days"], u["n_views"], u["n_favs"],
                            u["gap_days"], i["log_view"], i["avg_rating"],
                            i["age_days"]])
            y_ctr.append(s.ctr); y_cvr.append(s.cvr)
            y_rat.append(s.rating if s.rating is not None else 0.0)
            rat_mask.append(0 if s.rating is None else 1)
        sparse_t = {}
        for k, v in sparse.items():
            dtype = torch.float32 if k == "tags" else torch.long   # multi_hot 需 float
            sparse_t[k] = torch.tensor(np.asarray(v), dtype=dtype).to(device)
        numeric_t = torch.tensor(np.asarray(numeric, dtype=np.float32)).to(device)
        return (sparse_t, numeric_t,
                torch.tensor(y_ctr, dtype=torch.float32).to(device),
                torch.tensor(y_cvr, dtype=torch.float32).to(device),
                torch.tensor(y_rat, dtype=torch.float32).to(device),
                torch.tensor(rat_mask, dtype=torch.float32).to(device))

    opt = torch.optim.Adam(model.parameters(), lr=config.rank_lr)
    for epoch in range(config.rank_epochs):
        rng = np.random.default_rng(config.seed + epoch)
        rng.shuffle(samples)
        for start in range(0, len(samples), config.rank_batch_size):
            batch = samples[start:start + config.rank_batch_size]
            sparse_t, numeric_t, yc, yv, yr, mask = make_batch(batch)
            if sparse_t["user_id"].numel() < 2:
                continue
            pctr, pcvr, rating = model(sparse_t, numeric_t)
            loss = (config.rank_w_ctr * torch.nn.functional.binary_cross_entropy(
                        pctr.squeeze(-1), yc)
                    + config.rank_w_cvr * torch.nn.functional.binary_cross_entropy(
                        pcvr.squeeze(-1), yv))
            if mask.sum() > 0:
                loss = loss + config.rank_w_rating * torch.nn.functional.mse_loss(
                    rating.squeeze(-1)[mask.bool()], yr[mask.bool()])
            opt.zero_grad(); loss.backward(); opt.step()
    return model.cpu()
