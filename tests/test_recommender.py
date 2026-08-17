import pandas as pd
import pytest

from playcatch.data import original_dataset, synthetic_dataset
from playcatch.evaluate import leave_one_out_evaluate
from playcatch.recommender import ItemBasedRecommender


@pytest.fixture
def data():
    frame = pd.concat([original_dataset(), synthetic_dataset(42)], ignore_index=True)
    frame["last_played"] = pd.to_datetime(frame["last_played"])
    return frame


def test_recommendations_exclude_heard_items_and_are_sorted(data):
    model = ItemBasedRecommender(n_neighbors=5).fit(data)
    result = model.recommend(4, 5)
    heard = set(data.loc[data.user_id == 4, "song_id"])
    assert len(result) <= 5
    assert set(result.song_id).isdisjoint(heard)
    assert result.score.tolist() == sorted(result.score.tolist(), reverse=True)


def test_unknown_user_uses_fallback(data):
    result = ItemBasedRecommender().fit(data).recommend(999, 3)
    assert len(result) == 3
    assert result.origin.str.startswith("popularidade").all()


def test_invalid_k(data):
    with pytest.raises(ValueError, match="maior que zero"):
        ItemBasedRecommender().fit(data).recommend(1, 0)


def test_leave_one_out_metrics_are_bounded(data):
    metrics, details = leave_one_out_evaluate(data, n_neighbors=5, k=5)
    assert len(details) == 13
    for key in ("hit_rate_at_k", "precision_at_k", "mrr_at_k", "catalog_coverage"):
        assert 0 <= metrics[key] <= 1

