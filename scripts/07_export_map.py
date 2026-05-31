# scripts/07_export_map.py
"""
Bonus (UI, етап 1). Проєкція 10k ембедингів у 2D через UMAP для інтерактивної карти.

Метрика UMAP — cosine (наші вектори нормалізовані, близькість = кут). Результат
зберігається у ui/map_data.js як глобальна змінна window.MAP_DATA (щоб фронтенд міг
відкриватися напряму з file:// без CORS) і дублюється у ui/public/map_data.json для бекенду.
"""

import os
import json

import numpy as np
import pandas as pd
import umap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(ROOT, "data", "arxiv_subset.parquet")
EMB = os.path.join(ROOT, "embeddings", "embeddings.npy")
UI_DIR = os.path.join(ROOT, "ui")
PUBLIC_DIR = os.path.join(UI_DIR, "public")


def main() -> None:
    df = pd.read_parquet(PARQUET).reset_index(drop=True)
    emb = np.load(EMB)
    print(f"Завантажено {len(df)} статей, ембединги {emb.shape}")

    print("Рахуємо UMAP (metric=cosine, 2D)... це може зайняти ~хвилину")
    reducer = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1,
        metric="cosine", random_state=42,
    )
    coords = reducer.fit_transform(emb).astype(float)

    # Нормуємо координати у зручний діапазон для відображення.
    coords -= coords.mean(axis=0)

    records = []
    for i, row in df.iterrows():
        records.append({
            "id": f"paper_{i}",
            "x": round(float(coords[i, 0]), 3),
            "y": round(float(coords[i, 1]), 3),
            "title": str(row["title"])[:160],
            "year": int(row["year"]),
            "category": str(row["category"]),
        })

    os.makedirs(PUBLIC_DIR, exist_ok=True)

    # JSON — для бекенду (етап 2).
    with open(os.path.join(PUBLIC_DIR, "map_data.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    # JS-обгортка — щоб фронтенд відкривався напряму з file:// без fetch/CORS.
    with open(os.path.join(UI_DIR, "map_data.js"), "w", encoding="utf-8") as f:
        f.write("window.MAP_DATA = ")
        json.dump(records, f, ensure_ascii=False)
        f.write(";\n")

    top_cats = df["category"].value_counts().head(12)
    print(f"Збережено {len(records)} точок у ui/map_data.js та ui/public/map_data.json")
    print(f"Категорій усього: {df['category'].nunique()}; топ-12 використаємо для легенди:")
    print(top_cats.to_string())


if __name__ == "__main__":
    main()
