from __future__ import annotations


def mmr_rerank(scored: list[tuple[int, float]], item_cat: dict[int, str],
               top_n: int, lambda_: float) -> list[int]:
    """MMR 最大边际相关：精排分 − λ·与已选集合的类目相似度。"""
    selected: list[int] = []
    remaining = list(scored)
    while remaining and len(selected) < top_n:
        best_idx, best_val = -1, -float("inf")
        for idx, (item, score) in enumerate(remaining):
            div = sum(1 for s in selected if item_cat.get(item) == item_cat.get(s))
            mmr = score - lambda_ * div
            if mmr > best_val:
                best_val, best_idx = mmr, idx
        item = remaining.pop(best_idx)[0]
        selected.append(item)
    return selected


def rerank(scored: list[tuple[int, float]], seen: set[int],
           item_cat: dict[int, str], config) -> list[int]:
    filtered = [(i, s) for i, s in scored if i not in seen]
    if hasattr(config, "cold_age_days"):
        filtered = [
            (i, s * config.cold_weight if config.cold_age_days > 0 else s)
            for i, s in filtered
        ]
    return mmr_rerank(filtered, item_cat, config.top_n, config.mmr_lambda)
