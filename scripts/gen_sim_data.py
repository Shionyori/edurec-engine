from __future__ import annotations
import os
from engine.config import EngineConfig
from engine.data.simulator import generate
from engine.data.io import save_bundle


def main() -> None:
    cfg = EngineConfig()
    b = generate(cfg)
    out = os.path.join(cfg.data_dir, "sim")
    save_bundle(b, out)
    print(f"[gen_sim_data] 生成 {len(b.behaviors)} 条行为 -> {out}")


if __name__ == "__main__":
    main()
