from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.sklearn

from .pipeline import run_pipeline


def run_tracked_experiment(data_path, artifacts_root, tracking_db, **params):
    artifacts_root = Path(artifacts_root).resolve()
    artifacts_root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{Path(tracking_db).resolve().as_posix()}")
    mlflow.set_experiment("playcatch-recsys")
    run_dir = artifacts_root / f"run-n{params['n_neighbors']}-k{params['k']}-{params['norm']}-{params['score_policy']}"
    with mlflow.start_run() as run:
        model, summary = run_pipeline(data_path, run_dir, **params)
        mlflow.log_params(params)
        mlflow.log_metrics({key: float(summary[key]) for key in (
            "hit_rate_at_k", "precision_at_k", "mrr_at_k", "catalog_coverage"
        )})
        mlflow.set_tags({"dataset_sha256": summary["dataset_sha256"], "project": "playcatch-recsys", "data_type": "expanded_synthetic"})
        mlflow.log_artifacts(str(run_dir), artifact_path="evidence")
        try:
            mlflow.sklearn.log_model(model.model_, name="knn_model")
        except TypeError:
            mlflow.sklearn.log_model(model.model_, artifact_path="knn_model")
        (run_dir / "mlflow_run.json").write_text(json.dumps({"run_id": run.info.run_id}, indent=2), encoding="utf-8")
        return run.info.run_id, summary

