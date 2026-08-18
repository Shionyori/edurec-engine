from engine.pipeline.evaluate import recall_at_k, ndcg_at_k, auc_from_scores


def test_recall_at_k():
    assert recall_at_k(relevant={1, 3}, ranked=[0, 1, 2, 3, 4], k=3) == 0.5


def test_ndcg_ranked_better():
    # 单个相关项：排在越前面 NDCG 越高
    a = ndcg_at_k({1}, [1, 3, 0, 2, 4], k=5)
    b = ndcg_at_k({1}, [3, 1, 0, 2, 4], k=5)
    assert a > b


def test_auc():
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    assert abs(auc_from_scores(y, s) - 1.0) < 1e-6
