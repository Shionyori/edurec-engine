from __future__ import annotations
from collections import Counter
from typing import Sequence
import numpy as np
import torch
from ..features.user_features import build_user_features
from ..features.item_features import build_item_features
from .rerank import rerank


def _t(arr, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(np.asarray(arr), dtype=dtype)


def infer_batch(recall_model, rank_model, bundle, vocab, config,
                device: str = "cpu",
                target_users: Sequence[int] | None = None) -> dict[int, list[int]]:
    """批量推理：为每个 target 用户产出 top-N。

    返回 {原始用户ID: [原始资源ID, ...]}——key/value 均与输入数据集的原始 ID
    一致（内部编码在出口经 vocab 回映射，不再输出内部编码）。
    无画像/词表外的冷启动用户直接走热门兜底，保证 target 全量覆盖。

    target_users：需要产出推荐的用户（原始 ID）列表；默认取 bundle 词表内全部用户。
    """
    recall_model.eval(); rank_model.eval()
    uf = build_user_features(bundle, vocab)
    it = build_item_features(bundle, vocab)
    enc_items = sorted(vocab.item2id.values())
    id2item = {e: r for r, e in vocab.item2id.items()}
    target = sorted(set(target_users)) if target_users is not None else sorted(vocab.user2id)

    with torch.no_grad():
        item_feats = {
            "item_id": torch.stack([_t(it[i]["item_id"], torch.long) for i in enc_items]),
            "category_id": torch.stack([_t(it[i]["category_id"], torch.long) for i in enc_items]),
            "tags": torch.stack([_t(it[i]["tags"]) for i in enc_items]),
            "type_id": torch.stack([_t(it[i]["type_id"], torch.long) for i in enc_items]),
        }
        item_embs = recall_model.item_emb(item_feats).T          # (E, n_items)

        # 热门兜底：按行为频次取词表内资源（原始 ID → 编码 ID）
        pop = Counter(x.resource_id for x in bundle.behaviors)
        pop_enc = [vocab.item2id[r] for r, _ in pop.most_common()
                   if r in vocab.item2id][:config.top_n]

        out: dict[int, list[int]] = {}
        for u_orig in target:
            e_u = vocab.user2id.get(u_orig)
            cand_enc: list[int]
            if e_u is None:
                # 冷启动：不在词表（如交互过少的用户），直接热门兜底，跳过精排
                cand_enc = list(pop_enc)
                scored = [(e_i, float(len(cand_enc) - idx)) for idx, e_i in enumerate(cand_enc)]
            else:
                fu = {
                    "user_id": torch.stack([_t(uf[e_u]["user_id"], torch.long)]),
                    "cat_interest": torch.stack([_t(uf[e_u]["cat_interest"])]),
                    "tag_interest": torch.stack([_t(uf[e_u]["tag_interest"])]),
                    "type_pref": torch.stack([_t(uf[e_u]["type_pref"])]),
                }
                ue = recall_model.user_emb(fu)
                scores = (ue @ item_embs).squeeze(0)             # (n_items,)
                top = torch.argsort(scores, descending=True)[:config.recall_k]
                cand_enc = [enc_items[i] for i in top.tolist()]

                scored = []
                for i in cand_enc:
                    if i not in it:
                        continue
                    sparse = {
                        "user_id": torch.tensor([e_u], device=device),
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
                if not scored:                                    # 兜底：候选全为空
                    scored = [(e_i, float(len(pop_enc) - idx)) for idx, e_i in enumerate(cand_enc)]

            item_cat = {e_i: str(int(it[e_i]["category_id"]))
                        for e_i in cand_enc if e_i in it}
            chosen_enc = rerank(scored, seen=set(), item_cat=item_cat, config=config)
            out[u_orig] = [id2item[e] for e in chosen_enc]
    return out
