from __future__ import annotations
import torch
import torch.nn as nn
from ..common.fm import FM


class MultiTaskDeepFM(nn.Module):
    """多任务排序：shared-bottom（FM + Deep）+ 三个任务头（CTR/CVR/评分）。"""

    def __init__(self, sparse_specs, numeric_dim: int, deep_dim: int = 64):
        super().__init__()
        self.specs = {s.name: s for s in sparse_specs}
        self.embeddings = nn.ModuleDict({
            s.name: nn.Embedding(s.input_dim, s.embed_dim) for s in sparse_specs})
        self.fm = FM()
        emb_total = sum(s.embed_dim for s in sparse_specs)
        self.deep = nn.Sequential(
            nn.Linear(emb_total + numeric_dim, 128), nn.ReLU(),
            nn.Linear(128, deep_dim))
        self.ctr_head = nn.Linear(deep_dim + 1, 1)
        self.cvr_head = nn.Linear(deep_dim + 1, 1)
        self.rating_head = nn.Linear(deep_dim + 1, 1)

    def _embed(self, sparse: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        out = []
        for name, spec in self.specs.items():
            x = sparse[name]
            e = self.embeddings[name]
            if spec.kind == "multi_hot":
                out.append(x @ e.weight)
            else:
                out.append(e(x))
        return out

    def forward(self, sparse: dict[str, torch.Tensor],
                numeric: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embs = self._embed(sparse)
        fm_out = self.fm(embs).unsqueeze(-1)                 # (B, 1)
        deep_out = self.deep(torch.cat(embs + [numeric], dim=-1))   # (B, deep_dim)
        shared = torch.cat([fm_out, deep_out], dim=-1)       # (B, deep_dim+1)
        pctr = torch.sigmoid(self.ctr_head(shared))
        pcvr = torch.sigmoid(self.cvr_head(shared))
        rating = self.rating_head(shared)
        return pctr, pcvr, rating
