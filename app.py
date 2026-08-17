from pathlib import Path
import sys

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent / "src"))
from playcatch.data import build_datasets, load_and_validate
from playcatch.recommender import ItemBasedRecommender

DATA_PATH = Path(__file__).parent / "data" / "user_data_expanded.csv"
if not DATA_PATH.exists():
    build_datasets(DATA_PATH.parent, seed=42)
data, _ = load_and_validate(DATA_PATH)
model = ItemBasedRecommender(n_neighbors=5, norm="l2").fit(data)


def recommend_for_ui(user_id: int, k: int):
    result = model.recommend(int(user_id), int(k)).copy()
    result["score"] = result["score"].round(6)
    return result


demo = gr.Interface(
    fn=recommend_for_ui,
    inputs=[gr.Dropdown(sorted(data.user_id.unique().tolist()), label="Usuário", value=1), gr.Slider(1, 10, value=5, step=1, label="Quantidade")],
    outputs=gr.Dataframe(label="Recomendações", interactive=False),
    title="Playcatch — Recomendação de músicas",
    description="POC educacional item-based. IDs 109–118 e usuários 4–13 incluem dados sintéticos.",
)

if __name__ == "__main__":
    demo.launch(inbrowser=False, share=False)
