from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from playcatch.data import build_datasets
from playcatch.pipeline import run_pipeline
from playcatch.tracking import run_tracked_experiment


def main():
    original, expanded = build_datasets(ROOT / "data", seed=42)
    _, original_summary = run_pipeline(original, ROOT / "artifacts" / "original", n_neighbors=5, k=5, norm="l2")
    configurations = [
        {"n_neighbors": 3, "k": 3, "norm": "l2", "score_policy": "simple"},
        {"n_neighbors": 5, "k": 5, "norm": "l2", "score_policy": "simple"},
        {"n_neighbors": 7, "k": 5, "norm": "l1", "score_policy": "weighted_log1p"},
    ]
    runs = []
    for config in configurations:
        run_id, summary = run_tracked_experiment(expanded, ROOT / "artifacts" / "mlflow", ROOT / "mlruns.db", **config)
        runs.append({"run_id": run_id, **config, **summary})
    output = {"original": original_summary, "expanded_runs": runs}
    (ROOT / "artifacts" / "execution_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

