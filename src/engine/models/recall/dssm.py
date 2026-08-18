from __future__ import annotations
import torch
import torch.nn as nn


class TwoTower(nn.Module):
    """双塔召回：用户塔 / 资源塔独立编码，内积相似度，in-batch softmax 训练。"""

    def __init__(self, n_users: int, n_items: int, n_cats: int, n_tags: int,
                 embed_dim: int = 64):
        super().__init__()
        self.user_id_emb = nn.Embedding(n_users, embed_dim)
        self.item_id_emb = nn.Embedding(n_items, embed_dim)
        self.cat_emb = nn.Embedding(n_cats, 8)
        self.tag_emb = nn.Embedding(n_tags, 8)
        self.type_emb = nn.Embedding(3, 4)
        self.user_mlp = nn.Sequential(
            nn.Linear(embed_dim + 8 + 8 + 4, 128), nn.ReLU(),
            nn.Linear(128, embed_dim))
        self.item_mlp = nn.Sequential(
            nn.Linear(embed_dim + 8 + 8 + 4, 128), nn.ReLU(),
            nn.Linear(128, embed_dim))

    def user_emb(self, u: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([
            self.user_id_emb(u["user_id"]),
            u["cat_interest"] @ self.cat_emb.weight,
            u["tag_interest"] @ self.tag_emb.weight,
            u["type_pref"] @ self.type_emb.weight,
        ], dim=-1)
        return self.user_mlp(x)

    def item_emb(self, i: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([
            self.item_id_emb(i["item_id"]),
            self.cat_emb(i["category_id"]),
            i["tags"] @ self.tag_emb.weight,
            self.type_emb(i["type_id"]),
        ], dim=-1)
        return self.item_mlp(x)

    def score(self, u: dict[str, torch.Tensor],
              i: dict[str, torch.Tensor]) -> torch.Tensor:
        return (self.user_emb(u) * self.item_emb(i)).sum(-1)

    def forward(self, users, pos_items, neg_items, tau: float = 0.05) -> torch.Tensor:
        """in-batch softmax 损失。

        users/pos_items/neg_items 均为特征 dict（每个字段形状 (B, ...)）。
        正样本 logits 用 batch 内其他 item 作负样本。
        """
        ue = self.user_emb(users)                    # (B, E)
        pe = self.item_emb(pos_items)                # (B, E)
        ne = self.item_emb(neg_items)                # (B, E)
        logits_pos = (ue * pe).sum(-1) / tau         # (B,)
        # batch 内互为负样本：logits (B, B)
        logits_batch = ue @ pe.T / tau
        # 附加显式负样本列
        logits_neg = ue @ ne.T / tau                 # (B, n_neg)
        logits = torch.cat([logits_batch, logits_neg], dim=-1)   # (B, B+n_neg)
        target = torch.arange(logits.size(0), device=logits.device)
        return nn.functional.cross_entropy(logits, target) + (-logits_pos.mean() * 0.0)
