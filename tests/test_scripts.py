from engine.config import EngineConfig
from engine.data.io import save_bundle, load_bundle


def test_bundle_io_roundtrip(tmp_path):
    from engine.data.simulator import generate
    cfg = EngineConfig(sim_n_users=10, sim_n_resources=5,
                       sim_n_interactions=100, seed=0)
    b = generate(cfg)
    save_bundle(b, str(tmp_path))
    loaded = load_bundle(str(tmp_path))
    assert len(loaded.behaviors) == len(b.behaviors)
    assert len(loaded.resources) == len(b.resources)
