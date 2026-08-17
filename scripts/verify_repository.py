from pathlib import Path
import json
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
summary = json.loads((ROOT / "artifacts" / "execution_summary.json").read_text(encoding="utf-8"))
connection = sqlite3.connect(ROOT / "mlruns.db")
finished = connection.execute("select count(*) from runs where status='FINISHED'").fetchone()[0]
experiments = connection.execute("select count(*) from experiments where lifecycle_stage='active'").fetchone()[0]
assert finished >= 3
assert experiments >= 1
assert len(summary["expanded_runs"]) == 3
assert (ROOT / "data" / "user_data_expanded.csv").exists()
print(f"MLFLOW_FINISHED_RUNS={finished}")
print(f"MLFLOW_ACTIVE_EXPERIMENTS={experiments}")
print("ARTIFACT_MANIFEST=OK")
