from __future__ import annotations

import pandas as pd

from .data import aggregate_interactions
from .recommender import ItemBasedRecommender


def leave_one_out_evaluate(
    interactions: pd.DataFrame, n_neighbors: int = 5, k: int = 5,
    norm: str = "l2", score_policy: str = "simple",
) -> tuple[dict, pd.DataFrame]:
    aggregated = aggregate_interactions(interactions)
    details = []
    all_recommended: set[int] = set()
    for user_id, group in aggregated.groupby("user_id", sort=True):
        holdout = group.sort_values(["play_count", "song_id"], ascending=[False, True]).iloc[0]
        mask = ~((aggregated.user_id == user_id) & (aggregated.song_id == holdout.song_id))
        train = aggregated.loc[mask].copy()
        recommender = ItemBasedRecommender(n_neighbors, norm, score_policy).fit(train)
        recommendations = recommender.recommend(int(user_id), k)
        ids = recommendations.song_id.astype(int).tolist()
        all_recommended.update(ids)
        hit = int(int(holdout.song_id) in ids)
        rank = ids.index(int(holdout.song_id)) + 1 if hit else None
        details.append({
            "user_id": int(user_id), "held_out_song_id": int(holdout.song_id),
            "hit": hit, "rank": rank, "recommended_song_ids": ",".join(map(str, ids)),
        })
    result = pd.DataFrame(details)
    users = len(result)
    hits = int(result.hit.sum())
    metrics = {
        "users_evaluated": users,
        "hits": hits,
        "hit_rate_at_k": hits / users if users else 0.0,
        "precision_at_k": hits / (users * k) if users and k else 0.0,
        "mrr_at_k": float(sum(1 / r for r in result.loc[result.hit == 1, "rank"]) / users) if users else 0.0,
        "catalog_coverage": len(all_recommended) / aggregated.song_id.nunique() if len(aggregated) else 0.0,
    }
    return metrics, result

