from __future__ import annotations
import torch
import torch.nn as nn


class FM(nn.Module):
    """因子分解机：一阶 + 二阶特征交互，O(k) 复杂度（权重即各域 embedding）。"""

    def forward(self, embs: list[torch.Tensor]) -> torch.Tensor:
        first = torch.stack([e.sum(1) for e in embs], dim=1).sum(1)         # (B,)
        sum_sq = torch.stack([e.sum(1) for e in embs], dim=1).sum(1) ** 2
        sq_sum = torch.stack([(e ** 2).sum(1) for e in embs], dim=1).sum(1)
        second = 0.5 * (sum_sq - sq_sum)                                    # (B,)
        return first + second
