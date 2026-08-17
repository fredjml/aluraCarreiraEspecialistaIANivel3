from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .data import aggregate_interactions, interaction_matrix, load_and_validate
from .evaluate import leave_one_out_evaluate
from .recommender import ItemBasedRecommender


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_pipeline(data_path: str | Path, output_dir: str | Path, n_neighbors=5, k=5, norm="l2", score_policy="simple"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame, audit = load_and_validate(data_path)
    aggregated = aggregate_interactions(frame)
    matrix = interaction_matrix(aggregated)
    model = ItemBasedRecommender(n_neighbors, norm, score_policy).fit(frame)
    metrics, evaluation = leave_one_out_evaluate(frame, n_neighbors, k, norm, score_policy)
    popular = aggregated.groupby("song_id", as_index=False).play_count.sum().sort_values(["play_count", "song_id"], ascending=[False, True])
    users = aggregated.groupby("user_id", as_index=False).play_count.sum().sort_values(["play_count", "user_id"], ascending=[False, True])
    samples = pd.concat([model.recommend(int(user), k).assign(user_id=int(user)) for user in matrix.index[:5]], ignore_index=True)
    summary = {
        **audit, "dataset_sha256": file_sha256(data_path), "users": int(matrix.shape[0]),
        "songs": int(matrix.shape[1]), "unique_pairs": int((matrix > 0).sum().sum()),
        "matrix_shape": list(matrix.shape), "matrix_density": float((matrix > 0).sum().sum() / matrix.size),
        "total_plays": int(aggregated.play_count.sum()), **metrics,
    }
    aggregated.to_csv(output / "aggregated_interactions.csv", index=False)
    matrix.to_csv(output / "user_item_matrix.csv")
    popular.to_csv(output / "song_popularity.csv", index=False)
    users.to_csv(output / "user_volume.csv", index=False)
    evaluation.to_csv(output / "leave_one_out_details.csv", index=False)
    samples.to_csv(output / "sample_recommendations.csv", index=False)
    (output / "song_to_idx.json").write_text(json.dumps(model.song_to_idx_, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return model, summary

