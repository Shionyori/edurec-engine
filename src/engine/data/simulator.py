from __future__ import annotations
import numpy as np
from .schema import DataBundle, User, Resource, Behavior, Rating


def generate(config) -> DataBundle:
    rng = np.random.default_rng(config.seed)
    n_u, n_r = config.sim_n_users, config.sim_n_resources
    n_cat, n_tag = config.sim_n_categories, config.sim_n_tags

    # --- 隐藏结构参数 ---
    # 用户：类目偏好(狄利克雷)、类型偏好、活跃度
    user_cat = rng.dirichlet(np.ones(n_cat), size=n_u)          # (n_u, n_cat)
    user_type = rng.dirichlet(np.ones(3), size=n_u)             # course/article/video
    user_act = rng.gamma(2.0, 2.0, size=n_u)                    # 活跃度

    # 资源：类目、标签(0-4个)、质量、热度、冷门度
    res_cat = rng.integers(0, n_cat, size=n_r)
    res_tags = [tuple(rng.choice(n_tag, size=int(rng.integers(1, 5)), replace=False).astype(int).tolist())
                for _ in range(n_r)]
    res_quality = rng.beta(2.0, 2.0, size=n_r)
    res_pop = rng.lognormal(0.0, 1.0, size=n_r)

    resources = [
        Resource(resource_id=i, type=("course", "article", "video")[int(rng.integers(3))],
                 category_id=int(res_cat[i]), tags=tuple(str(t) for t in res_tags[i]))
        for i in range(n_r)
    ]
    users = [User(user_id=i) for i in range(n_u)]

    # --- 采样交互 ---
    behaviors: list[Behavior] = []
    ratings: list[Rating] = []
    ts = 1_700_000_000
    for _ in range(config.sim_n_interactions):
        u = int(rng.choice(n_u, p=user_act / user_act.sum()))
        i = int(rng.integers(n_r))
        match = user_cat[u, res_cat[i]]                          # 类目匹配度
        p = float(match * res_quality[i] * res_pop[i] + rng.normal(0, 0.1))
        if rng.uniform() > min(1.0, 3.0 * p):                    # 保留一部分交互率
            continue
        action = rng.choice(("view", "click", "favorite"),
                            p=(0.5, 0.3, 0.2))
        behaviors.append(Behavior(user_id=u, resource_id=i, action=str(action), ts=ts))
        if action == "favorite" and rng.uniform() < 0.5:
            score = int(np.clip(round(2.5 + 2.5 * match + rng.normal(0, 0.3)), 1, 5))
            ratings.append(Rating(user_id=u, resource_id=i, score=score, ts=ts))
        ts += 1

    return DataBundle(users=users, resources=resources,
                      behaviors=behaviors, ratings=ratings)
