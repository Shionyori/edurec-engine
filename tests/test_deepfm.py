import torch
from engine.features.feature_config import FeatureSpec
from engine.models.common.fm import FM
from engine.models.rank.multitask_deepfm import MultiTaskDeepFM


def _specs():
    return [
        FeatureSpec("user_id", "categorical", 10, 4),
        FeatureSpec("item_id", "categorical", 10, 4),
        FeatureSpec("tags", "multi_hot", 8, 4),
    ]


def test_fm_scalar_per_sample():
    embs = [torch.randn(4, 6) for _ in range(3)]
    out = FM()(embs)
    assert out.shape == (4,)


def test_multitask_outputs():
    torch.manual_seed(0)
    m = MultiTaskDeepFM(_specs(), numeric_dim=3)
    sparse = {
        "user_id": torch.tensor([0, 1]),
        "item_id": torch.tensor([1, 2]),
        "tags": torch.tensor([[0.5, 0.5, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1.0, 0, 0, 0, 0]]),
    }
    numeric = torch.randn(2, 3)
    pctr, pcvr, rating = m.forward(sparse, numeric)
    assert pctr.shape == (2, 1)
    assert (pctr > 0).all() and (pctr < 1).all()      # sigmoid 输出
    assert rating.shape == (2, 1)
