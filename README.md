# Playcatch Recommender

POC de recomendação musical item-based com Pandas, Scikit-learn, MLflow/SQLite e Gradio. A base expandida preserva os 16 eventos originais e acrescenta 10 usuários e 10 músicas sintéticos, identificados pela coluna `data_origin`.

## Execução

```powershell
uv sync --extra dev
uv run python scripts/run_pipeline.py
uv run pytest -q
uv run python app.py
```

MLflow UI:

```powershell
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db --host 127.0.0.1 --port 5000
```

Abra `http://127.0.0.1:5000`. A interface Gradio informa seu endereço local no terminal.

## Artefatos

- `data/`: datasets original e expandido.
- `artifacts/original/`: evidências do dataset original.
- `artifacts/mlflow/`: evidências por configuração.
- `mlruns.db`: backend SQLite do MLflow.
- `Analise/`: relatórios executivo-técnicos.

Dados sintéticos servem para teste funcional e não comprovam desempenho em produção.
