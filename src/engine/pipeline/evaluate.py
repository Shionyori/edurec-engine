from __future__ import annotations
import numpy as np
import torch
from ..features.user_features import build_user_features
from ..features.item_features import build_item_features
from ..data.preprocess import build_recall_pairs


def recall_at_k(relevant: set[int], ranked: list[int], k: int) -> float:
    if not relevant:
        return 0.0
    hit = sum(1 for x in ranked[:k] if x in relevant)
    return hit / len(relevant)


def hit_rate_at_k(relevant: set[int], ranked: list[int], k: int) -> float:
    return 1.0 if any(x in relevant for x in ranked[:k]) else 0.0


def ndcg_at_k(relevant: set[int], ranked: list[int], k: int) -> float:
    dcg = 0.0
    for pos, x in enumerate(ranked[:k], start=1):
        if x in relevant:
            dcg += 1.0 / np.log2(pos + 1)
    ideal = sum(1.0 / np.log2(pos + 1)
                for pos in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def auc_from_scores(y_true, y_score) -> float:
    order = np.argsort(y_score)
    ranked = np.asarray(y_true)[order]
    n_pos = int(ranked.sum())
    n_neg = len(ranked) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = sum(idx + 1 for idx, v in enumerate(ranked) if v == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _t(arr, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(np.asarray(arr), dtype=dtype)


def evaluate_recall(model, bundle, vocab, config, device: str = "cpu") -> dict:
    model.eval()
    it = build_item_features(bundle, vocab)
    uf = build_user_features(bundle, vocab)
    test_pairs = build_recall_pairs(bundle, vocab)
    relevant_of: dict[int, set[int]] = {}
    for u, i in test_pairs:
        relevant_of.setdefault(u, set()).add(i)

    item_ids = sorted(vocab.item2id.values())
    item_feats = {
        "item_id": torch.stack([_t(it[i]["item_id"], torch.long) for i in item_ids]),
        "category_id": torch.stack([_t(it[i]["category_id"], torch.long) for i in item_ids]),
        "tags": torch.stack([_t(it[i]["tags"]) for i in item_ids]),
        "type_id": torch.stack([_t(it[i]["type_id"], torch.long) for i in item_ids]),
    }
    k = config.recall_k
    with torch.no_grad():
        item_embs = model.item_emb(item_feats).T                 # (E, n_items)
        rec, hr, ndcg = [], [], []
        for u in relevant_of:
            fu = {
                "user_id": torch.stack([_t(uf[u]["user_id"], torch.long)]),
                "cat_interest": torch.stack([_t(uf[u]["cat_interest"])]),
                "tag_interest": torch.stack([_t(uf[u]["tag_interest"])]),
                "type_pref": torch.stack([_t(uf[u]["type_pref"])]),
            }
            ue = model.user_emb(fu)                               # (1, E)
            scores = (ue @ item_embs).squeeze(0)                  # (n_items,)
            ranked = [item_ids[x] for x in torch.argsort(scores, descending=True).tolist()]
            rel = relevant_of[u]
            rec.append(recall_at_k(rel, ranked, k))
            hr.append(hit_rate_at_k(rel, ranked, k))
            ndcg.append(ndcg_at_k(rel, ranked, k))
    return {f"recall@{k}": float(np.mean(rec)),
            f"hitrate@{k}": float(np.mean(hr)),
            f"ndcg@{k}": float(np.mean(ndcg))}


def rmse(y_true, y_pred) -> float:
    a, b = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def evaluate_rank(model, splits, config, device: str = "cpu") -> dict:
    """在验证集上评估排序模型：CTR/CVR AUC + 评分 RMSE。"""
    import torch
    from ..data.preprocess import build_rank_samples
    from ..features.user_features import build_user_features
    from ..features.item_features import build_item_features
    uf = build_user_features(splits.train, splits.vocab)      # 防泄漏：统计来自训练集
    it = build_item_features(splits.train, splits.vocab)
    samples = build_rank_samples(splits.val, splits.vocab,
                                 __import__("numpy").random.default_rng(0))
    model.eval()
    y_ctr, p_ctr, y_cvr, p_cvr, y_rat, p_rat = [], [], [], [], [], []
    with torch.no_grad():
        for s in samples:
            u = uf.get(s.user_idx)
            i = it.get(s.item_idx)
            if u is None or i is None:
                continue
            sparse = {
                "user_id": torch.tensor([s.user_idx], device=device),
                "item_id": torch.tensor([s.item_idx], device=device),
                "category_id": torch.tensor([int(i["category_id"])], device=device),
                "tags": torch.tensor(np.asarray([i["tags"]], dtype=np.float32),
                                     dtype=torch.float32, device=device),
                "type_id": torch.tensor([int(i["type_id"])], device=device),
                "hour": torch.tensor([s.hour], device=device),
                "dow": torch.tensor([s.dow], device=device),
            }
            numeric = torch.tensor([[
                float(u["active_days"]), float(u["n_views"]), float(u["n_favs"]),
                float(u["gap_days"]), float(i["log_view"]), float(i["avg_rating"]),
                float(i["age_days"]),
            ]], device=device)
            pctr, pcvr, rating = model(sparse, numeric)
            y_ctr.append(s.ctr); p_ctr.append(float(pctr))
            y_cvr.append(s.cvr); p_cvr.append(float(pcvr))
            if s.rating is not None:
                y_rat.append(s.rating); p_rat.append(float(rating))
    return {
        "ctr_auc": auc_from_scores(y_ctr, p_ctr),
        "cvr_auc": auc_from_scores(y_cvr, p_cvr) if len(set(y_cvr)) > 1 else 0.5,
        "rating_rmse": rmse(y_rat, p_rat) if p_rat else float("nan"),
    }
