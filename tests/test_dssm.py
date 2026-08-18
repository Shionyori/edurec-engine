import torch
from engine.models.recall.dssm import TwoTower


def _feats(device="cpu"):
    return {
        "user_id": torch.tensor([0, 1], device=device),
        "cat_interest": torch.tensor([[0.6, 0.4, 0.0, 0.0], [0.2, 0.3, 0.5, 0.0]], device=device),
        "tag_interest": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        "type_pref": torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.3, 0.7]], device=device),
        "item_id": torch.tensor([0, 1], device=device),
        "category_id": torch.tensor([0, 1], device=device),
        "tags": torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=device),
        "type_id": torch.tensor([0, 1], device=device),
    }


def test_embedding_shapes():
    m = TwoTower(n_users=5, n_items=5, n_cats=4, n_tags=3, embed_dim=8)
    f = _feats()
    assert m.user_emb(f).shape == (2, 8)
    assert m.item_emb(f).shape == (2, 8)


def test_inbatch_softmax_loss_decreases():
    torch.manual_seed(0)
    m = TwoTower(n_users=10, n_items=10, n_cats=4, n_tags=3, embed_dim=8)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    f = _feats()
    loss1 = m.forward(f, f, f)
    opt.zero_grad(); loss1.backward(); opt.step()
    loss2 = m.forward(f, f, f)
    assert loss2.item() < loss1.item()
