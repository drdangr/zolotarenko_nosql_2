# scripts/04_search.py
"""
Частина 3. Пошукові запити.

1) Чистий семантичний пошук у Pinecone.
2) Пошук із фільтрацією за метаданими (рік + категорія).
3) Порівняння метрик схожості (cosine / dot product / L2) на локальних ембедингах.
"""

import os

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)

# Повний датасет — щоб підтягувати повний abstract за id після пошуку.
df = pd.read_parquet(os.path.join(ROOT, "data", "arxiv_subset.parquet")).reset_index(drop=True)
embeddings = np.load(os.path.join(ROOT, "embeddings", "embeddings.npy"))


def encode_query(text: str) -> np.ndarray:
    """Кодує запит у нормалізований 768-вимірний вектор (той самий формат, що й під час індексації)."""
    return model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]


def print_hit(rank: int, meta: dict, score: float | None = None) -> None:
    score_str = f"  score={score:.4f}" if score is not None else ""
    print(f"  {rank}. [{meta.get('category')}, {meta.get('year')}]{score_str}")
    print(f"     {meta.get('title')}")
    print(f"     {str(meta.get('abstract'))[:200]}...")


def semantic_search(query: str, top_k: int = TOP_K, flt: dict | None = None):
    """Семантичний пошук у Pinecone (опційно з фільтром за метаданими)."""
    qv = encode_query(query).tolist()
    return index.query(vector=qv, top_k=top_k, include_metadata=True, filter=flt)


# --------------------------------------------------------------------------
# 3. Чистий семантичний пошук
# --------------------------------------------------------------------------
def demo_pure_semantic() -> None:
    query = "teaching machines to recognize objects in pictures"
    print("=" * 80)
    print(f"[3] ЧИСТИЙ СЕМАНТИЧНИЙ ПОШУК\nЗапит: {query!r}")
    print("=" * 80)
    res = semantic_search(query)
    for rank, match in enumerate(res["matches"], 1):
        print_hit(rank, match["metadata"], match["score"])


# --------------------------------------------------------------------------
# 4. Пошук із фільтрацією
# --------------------------------------------------------------------------
def demo_filtered() -> None:
    max_year = int(df["year"].max())
    recent_cutoff = max_year - 4  # "останні 5 років" вибірки (включно)

    print("\n" + "=" * 80)
    print("[4] ПОШУК ІЗ ФІЛЬТРАЦІЄЮ")
    print("=" * 80)

    query = "reinforcement learning"

    print(f"\n--- Приклад A: '{query}', рік >= {recent_cutoff} ТА категорія cs.LG ---")
    res_a = semantic_search(
        query, flt={"year": {"$gte": recent_cutoff}, "category": {"$eq": "cs.LG"}}
    )
    if res_a["matches"]:
        for rank, m in enumerate(res_a["matches"], 1):
            print_hit(rank, m["metadata"], m["score"])
    else:
        print("  (нічого не знайдено за цим фільтром)")

    print(f"\n--- Приклад B: '{query}', старіші статті (рік < 2015), будь-яка категорія ---")
    res_b = semantic_search(query, flt={"year": {"$lt": 2015}})
    if res_b["matches"]:
        for rank, m in enumerate(res_b["matches"], 1):
            print_hit(rank, m["metadata"], m["score"])
    else:
        print("  (нічого не знайдено за цим фільтром)")


# --------------------------------------------------------------------------
# 5. Порівняння метрик схожості на локальних ембедингах
# --------------------------------------------------------------------------
def demo_metrics() -> None:
    query = "teaching machines to recognize objects in pictures"
    print("\n" + "=" * 80)
    print(f"[5] ПОРІВНЯННЯ МЕТРИК НА ЛОКАЛЬНИХ ЕМБЕДИНГАХ\nЗапит: {query!r}")
    print("=" * 80)

    q = encode_query(query)

    cosine = embeddings @ q                       # вектори нормалізовані → cosine == dot
    dot = embeddings @ q
    l2 = np.linalg.norm(embeddings - q, axis=1)   # менше = ближче

    def top5(scores: np.ndarray, ascending: bool) -> list[int]:
        order = np.argsort(scores)
        return list(order[:5]) if ascending else list(order[::-1][:5])

    metrics = {
        "cosine similarity (↓ rank, більше=краще)": top5(cosine, ascending=False),
        "dot product       (↓ rank, більше=краще)": top5(dot, ascending=False),
        "L2 distance        (↓ rank, менше=краще)": top5(l2, ascending=True),
    }

    for name, idxs in metrics.items():
        print(f"\n--- {name} ---")
        for rank, i in enumerate(idxs, 1):
            row = df.iloc[i]
            print(f"  {rank}. paper_{i}  [{row['category']}, {row['year']}]  {row['title'][:70]}")


def main() -> None:
    demo_pure_semantic()
    demo_filtered()
    demo_metrics()


if __name__ == "__main__":
    main()
