from __future__ import annotations
import os
from engine.config import EngineConfig
from engine.data.movielens import load
from engine.data.io import save_bundle


def main() -> None:
    cfg = EngineConfig()
    b = load(cfg.data_dir, auto_download=True)
    out = os.path.join(cfg.data_dir, "ml_processed")
    save_bundle(b, out)
    print(f"[load_movielens] 用户={len(b.users)} 资源={len(b.resources)} "
          f"行为={len(b.behaviors)} 评分={len(b.ratings)} -> {out}")


if __name__ == "__main__":
    main()
