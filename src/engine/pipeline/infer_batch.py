from __future__ import annotations
from collections import Counter
import numpy as np
import torch
from ..features.user_features import build_user_features
from ..features.item_features import build_item_features
from .rerank import rerank


def _t(arr, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(np.asarray(arr), dtype=dtype)


def infer_batch(recall_model, rank_model, bundle, vocab, config,
                device: str = "cpu") -> dict[int, list[int]]:
    recall_model.eval(); rank_model.eval()
    uf = build_user_features(bundle, vocab)
    it = build_item_features(bundle, vocab)
    item_ids = sorted(vocab.item2id.values())

    with torch.no_grad():
        item_feats = {
            "item_id": torch.stack([_t(it[i]["item_id"], torch.long) for i in item_ids]),
            "category_id": torch.stack([_t(it[i]["category_id"], torch.long) for i in item_ids]),
            "tags": torch.stack([_t(it[i]["tags"]) for i in item_ids]),
            "type_id": torch.stack([_t(it[i]["type_id"], torch.long) for i in item_ids]),
        }
        item_embs = recall_model.item_emb(item_feats).T          # (E, n_items)

        # 热门兜底：按行为频次
        pop = Counter(x.resource_id for x in bundle.behaviors)
        pop_items = [vocab.item2id[r] for r, _ in pop.most_common(config.top_n)]

        out: dict[int, list[int]] = {}
        for u_idx in sorted(vocab.user2id.values()):
            if u_idx in uf:
                fu = {
                    "user_id": torch.stack([_t(uf[u_idx]["user_id"], torch.long)]),
                    "cat_interest": torch.stack([_t(uf[u_idx]["cat_interest"])]),
                    "tag_interest": torch.stack([_t(uf[u_idx]["tag_interest"])]),
                    "type_pref": torch.stack([_t(uf[u_idx]["type_pref"])]),
                }
                ue = recall_model.user_emb(fu)
                scores = (ue @ item_embs).squeeze(0)             # (n_items,)
                top = torch.argsort(scores, descending=True)[:config.recall_k]
                candidates = [item_ids[i] for i in top.tolist()]
            else:
                candidates = list(pop_items)

            scored = []
            for i in candidates:
                if i not in it:
                    continue
                sparse = {
                    "user_id": torch.tensor([u_idx], device=device),
                    "item_id": torch.tensor([i], device=device),
                    "category_id": torch.tensor([int(it[i]["category_id"])], device=device),
                    "tags": torch.tensor(np.asarray([it[i]["tags"]], dtype=np.float32),
                                         dtype=torch.float32, device=device),
                    "type_id": torch.tensor([int(it[i]["type_id"])], device=device),
                    "hour": torch.tensor([12], device=device),
                    "dow": torch.tensor([0], device=device),
                }
                numeric = torch.tensor([[
                    0.0, 0.0, 0.0, 0.0,
                    float(it[i]["log_view"]), float(it[i]["avg_rating"]),
                    float(it[i]["age_days"]),
                ]], device=device)
                pctr, pcvr, rating = rank_model(sparse, numeric)
                final = 0.5 * float(pctr) + 0.3 * float(pcvr) + 0.2 * (float(rating) / 5.0)
                scored.append((i, final))
            item_cat = {i: str(int(it[i]["category_id"])) for i in candidates if i in it}
            out[u_idx] = rerank(scored, seen=set(), item_cat=item_cat, config=config)
    return out
