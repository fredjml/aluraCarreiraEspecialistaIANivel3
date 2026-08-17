import app


def test_gradio_callback_returns_expected_schema():
    result = app.recommend_for_ui(1, 3)
    assert list(result.columns) == ["rank", "song_id", "score", "origin"]
    assert len(result) <= 3

