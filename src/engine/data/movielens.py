from __future__ import annotations
import os
import urllib.request
import zipfile
from .schema import DataBundle, User, Resource, Behavior, Rating

_ML_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def _parse_movies_line(line: str) -> tuple[int, str, tuple[str, ...]]:
    movie_id, title, genres = line.strip().split("::")
    return int(movie_id), title, tuple(genres.split("|"))


def _parse_ratings_line(line: str) -> tuple[int, int, int, int]:
    uid, mid, score, ts = line.strip().split("::")
    return int(uid), int(mid), int(score), int(ts)


def _to_behavior_action(score: int) -> str | None:
    return "click" if score >= 4 else None


def _load_or_download(data_dir: str, auto_download: bool) -> str:
    ml_dir = os.path.join(data_dir, "ml-1m")
    if os.path.isdir(ml_dir):
        return ml_dir
    if not auto_download:
        raise FileNotFoundError(f"{ml_dir} 不存在且 auto_download=False")
    zpath = os.path.join(data_dir, "ml-1m.zip")
    os.makedirs(data_dir, exist_ok=True)
    urllib.request.urlretrieve(_ML_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(data_dir)
    return ml_dir


def load(data_dir: str, auto_download: bool = True) -> DataBundle:
    ml_dir = _load_or_download(data_dir, auto_download)
    resources: list[Resource] = []
    cat_ids: dict[str, int] = {}
    with open(os.path.join(ml_dir, "movies.dat"), encoding="iso-8859-1") as f:
        for line in f:
            movie_id, _, genres = _parse_movies_line(line)
            first = genres[0]
            cat_ids.setdefault(first, len(cat_ids))
            resources.append(Resource(
                resource_id=movie_id, type="video",
                category_id=cat_ids[first], tags=genres))
    users: list[User] = []
    behaviors: list[Behavior] = []
    ratings: list[Rating] = []
    user_ids: dict[int, int] = {}
    with open(os.path.join(ml_dir, "ratings.dat"), encoding="iso-8859-1") as f:
        for line in f:
            uid, mid, score, ts = _parse_ratings_line(line)
            users.append(User(user_id=uid))
            ratings.append(Rating(user_id=uid, resource_id=mid, score=score, ts=ts))
            action = _to_behavior_action(score)
            if action is not None:
                behaviors.append(Behavior(user_id=uid, resource_id=mid, action=action, ts=ts))
    # 去重 users（ratings 文件含重复 uid）
    seen = set()
    users = [u for u in users if not (u.user_id in seen or seen.add(u.user_id))]
    return DataBundle(users=users, resources=resources,
                      behaviors=behaviors, ratings=ratings)
