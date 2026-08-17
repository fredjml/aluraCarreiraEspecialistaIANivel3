from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from .data import aggregate_interactions, interaction_matrix


@dataclass
class ItemBasedRecommender:
    n_neighbors: int = 5
    norm: str = "l2"
    score_policy: str = "simple"

    def fit(self, interactions: pd.DataFrame) -> "ItemBasedRecommender":
        self.aggregated_ = aggregate_interactions(interactions)
        self.user_item_ = interaction_matrix(self.aggregated_)
        self.song_ids_ = self.user_item_.columns.to_numpy(dtype=int)
        item_user = self.user_item_.T.to_numpy(dtype=float)
        if self.norm not in {"l1", "l2"}:
            raise ValueError("norm deve ser 'l1' ou 'l2'")
        self.item_user_normalized_ = normalize(item_user, norm=self.norm, axis=1)
        self.effective_neighbors_ = min(max(1, self.n_neighbors), len(self.song_ids_))
        self.model_ = NearestNeighbors(metric="cosine", algorithm="brute")
        self.model_.fit(self.item_user_normalized_)
        self.song_to_idx_ = {int(song): idx for idx, song in enumerate(self.song_ids_)}
        self.popularity_ = (
            self.aggregated_.groupby("song_id")["play_count"].sum()
            .sort_values(ascending=False, kind="stable")
        )
        return self

    def recommend(self, user_id: int, k: int = 5) -> pd.DataFrame:
        if not hasattr(self, "model_"):
            raise RuntimeError("O modelo precisa ser ajustado antes da recomendação")
        if k <= 0:
            raise ValueError("k deve ser maior que zero")
        if user_id not in self.user_item_.index:
            return self._popularity_fallback(k, "popularidade_usuario_desconhecido")
        history = self.user_item_.loc[user_id]
        heard = set(int(song) for song in history.index[history > 0])
        scores: dict[int, float] = {}
        for song_id in sorted(heard):
            idx = self.song_to_idx_[song_id]
            distances, indices = self.model_.kneighbors(
                self.item_user_normalized_[idx].reshape(1, -1),
                n_neighbors=self.effective_neighbors_,
            )
            strength = float(history[song_id])
            weight = np.log1p(strength) if self.score_policy == "weighted_log1p" else 1.0
            for distance, neighbor_idx in zip(distances[0], indices[0]):
                candidate = int(self.song_ids_[neighbor_idx])
                if candidate == song_id or candidate in heard:
                    continue
                similarity = max(0.0, 1.0 - float(distance))
                scores[candidate] = scores.get(candidate, 0.0) + similarity * weight
        ranked = sorted(scores.items(), key=lambda value: (-value[1], value[0]))[:k]
        if not ranked:
            return self._popularity_fallback(k, "popularidade_sem_candidatos", heard)
        return pd.DataFrame(
            [(rank, song, score, "personalizado") for rank, (song, score) in enumerate(ranked, 1)],
            columns=["rank", "song_id", "score", "origin"],
        )

    def _popularity_fallback(self, k: int, origin: str, heard: set[int] | None = None) -> pd.DataFrame:
        heard = heard or set()
        candidates = [(int(song), float(score)) for song, score in self.popularity_.items() if int(song) not in heard][:k]
        return pd.DataFrame(
            [(rank, song, score, origin) for rank, (song, score) in enumerate(candidates, 1)],
            columns=["rank", "song_id", "score", "origin"],
        )

