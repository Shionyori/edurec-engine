from __future__ import annotations
from dataclasses import dataclass


@dataclass
class FeatureSpec:
    name: str
    kind: str                       # numeric | categorical | multi_hot
    input_dim: int
    embed_dim: int = 0
    used_by: str = "rank"           # recall | rank | both


USER_NUMERIC = ["active_days", "n_views", "n_favs", "gap_days"]
ITEM_NUMERIC = ["log_view", "avg_rating", "age_days"]


def build_user_specs(vocab) -> list[FeatureSpec]:
    return [
        FeatureSpec("user_id", "categorical", vocab.n_users, 64, "both"),
        FeatureSpec("cat_interest", "multi_hot", vocab.n_cats, 8, "both"),
        FeatureSpec("tag_interest", "multi_hot", vocab.n_tags, 8, "both"),
        FeatureSpec("type_pref", "multi_hot", 3, 4, "both"),
        *[FeatureSpec(n, "numeric", 1, 0, "rank") for n in USER_NUMERIC],
    ]


def build_item_specs(vocab) -> list[FeatureSpec]:
    return [
        FeatureSpec("item_id", "categorical", vocab.n_items, 64, "recall"),
        FeatureSpec("category_id", "categorical", vocab.n_cats, 8, "both"),
        FeatureSpec("tags", "multi_hot", vocab.n_tags, 8, "both"),
        FeatureSpec("type_id", "categorical", 3, 4, "both"),
        *[FeatureSpec(n, "numeric", 1, 0, "rank") for n in ITEM_NUMERIC],
    ]


def build_rank_sparse_specs(vocab) -> list[FeatureSpec]:
    return [
        FeatureSpec("user_id", "categorical", vocab.n_users, 16, "rank"),
        FeatureSpec("item_id", "categorical", vocab.n_items, 16, "rank"),
        FeatureSpec("category_id", "categorical", vocab.n_cats, 8, "rank"),
        FeatureSpec("tags", "multi_hot", vocab.n_tags, 8, "rank"),
        FeatureSpec("type_id", "categorical", 3, 4, "rank"),
        FeatureSpec("hour", "categorical", 24, 4, "rank"),
        FeatureSpec("dow", "categorical", 7, 4, "rank"),
    ]
