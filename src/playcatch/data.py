from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"user_id", "song_id", "play_count", "last_played"}

ORIGINAL_ROWS = [
    (1, 101, 5, "2023-10-01"), (1, 102, 3, "2023-10-02"),
    (1, 103, 2, "2023-10-03"), (2, 101, 4, "2023-10-01"),
    (2, 104, 6, "2023-10-02"), (2, 105, 1, "2023-10-03"),
    (3, 106, 7, "2023-10-01"), (3, 107, 3, "2023-10-02"),
    (3, 108, 2, "2023-10-03"), (1, 104, 6, "2023-10-03"),
    (2, 101, 7, "2023-10-03"), (3, 103, 3, "2023-10-01"),
    (3, 104, 4, "2023-10-02"), (1, 106, 4, "2023-10-02"),
    (2, 104, 2, "2023-10-03"), (3, 105, 1, "2023-10-03"),
]


def original_dataset() -> pd.DataFrame:
    frame = pd.DataFrame(
        ORIGINAL_ROWS, columns=["user_id", "song_id", "play_count", "last_played"]
    )
    frame["data_origin"] = "original"
    return frame


def synthetic_dataset(seed: int = 42) -> pd.DataFrame:
    """Create 10 users and 10 new songs with reproducible preference clusters."""
    import numpy as np

    rng = np.random.default_rng(seed)
    clusters = {
        0: [101, 104, 109, 110, 111, 112],
        1: [103, 106, 113, 114, 115, 116],
        2: [102, 105, 107, 108, 117, 118],
    }
    rows: list[tuple[int, int, int, str, str]] = []
    for user_id in range(4, 14):
        cluster = clusters[(user_id - 4) % 3]
        for rank, song_id in enumerate(cluster):
            base = 9 - rank
            count = int(max(1, base + rng.integers(-1, 2)))
            day = 4 + ((user_id + rank) % 10)
            rows.append((user_id, song_id, count, f"2023-10-{day:02d}", "synthetic"))
        bridge_song = [104, 106, 101][(user_id - 4) % 3]
        if bridge_song not in cluster:
            rows.append((user_id, bridge_song, 2, "2023-10-14", "synthetic"))
    return pd.DataFrame(
        rows, columns=["user_id", "song_id", "play_count", "last_played", "data_origin"]
    )


def build_datasets(output_dir: str | Path, seed: int = 42) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    original = original_dataset()
    expanded = pd.concat([original, synthetic_dataset(seed)], ignore_index=True)
    original_path = output / "user_data_original.csv"
    expanded_path = output / "user_data_expanded.csv"
    original.to_csv(original_path, index=False)
    expanded.to_csv(expanded_path, index=False)
    return original_path, expanded_path


def load_and_validate(path: str | Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing_columns)}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("O dataset contém valores ausentes em colunas obrigatórias")
    frame["last_played"] = pd.to_datetime(frame["last_played"], errors="raise")
    for name in ("user_id", "song_id", "play_count"):
        frame[name] = pd.to_numeric(frame[name], errors="raise").astype(int)
    if (frame["play_count"] <= 0).any():
        raise ValueError("play_count deve ser inteiro positivo")
    exact_duplicates = int(frame.duplicated().sum())
    clean = frame.drop_duplicates().copy()
    pair_duplicates = int(clean.duplicated(["user_id", "song_id"], keep=False).sum())
    audit = {
        "raw_rows": len(frame), "clean_rows": len(clean),
        "exact_duplicates_removed": exact_duplicates,
        "rows_in_repeated_pairs": pair_duplicates,
        "missing_values": int(frame.isna().sum().sum()),
    }
    return clean, audit


def aggregate_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["user_id", "song_id"], as_index=False)
        .agg(play_count=("play_count", "sum"), last_played=("last_played", "max"))
        .sort_values(["user_id", "song_id"], kind="stable")
        .reset_index(drop=True)
    )


def interaction_matrix(aggregated: pd.DataFrame) -> pd.DataFrame:
    return aggregated.pivot_table(
        index="user_id", columns="song_id", values="play_count", aggfunc="sum", fill_value=0
    ).sort_index().sort_index(axis=1)

