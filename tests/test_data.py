import pandas as pd
import pytest

from playcatch.data import aggregate_interactions, interaction_matrix, original_dataset, synthetic_dataset


def test_original_aggregation_and_matrix():
    frame = original_dataset()
    frame["last_played"] = pd.to_datetime(frame["last_played"])
    aggregated = aggregate_interactions(frame)
    matrix = interaction_matrix(aggregated)
    assert matrix.shape == (3, 8)
    assert matrix.loc[2, 101] == 11
    assert matrix.loc[2, 104] == 8
    assert int(aggregated.play_count.sum()) == 60


def test_synthetic_scope_and_reproducibility():
    first = synthetic_dataset(42)
    second = synthetic_dataset(42)
    pd.testing.assert_frame_equal(first, second)
    assert first.user_id.nunique() == 10
    assert set(range(109, 119)).issubset(set(first.song_id))
    assert set(first.data_origin) == {"synthetic"}

