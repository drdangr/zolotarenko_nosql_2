# scripts/03_load_to_pinecone.py
"""
Частина 2. Створення індексу в Pinecone та завантаження ембедингів статей.

Створює (за потреби) serverless-індекс arxiv-papers з метрикою cosine і
розмірністю 768, після чого батчами завантажує вектори разом із метаданими.

Чому abstract обрізається до 500 символів:
    Pinecone обмежує сумарний розмір метаданих одного вектора до 40 KB. Повний
    текст анотації зберігаємо окремо (у Parquet) і підтягуємо за id уже після пошуку.
"""

import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PARQUET = os.path.join(ROOT, "data", "arxiv_subset.parquet")
INPUT_EMBEDDINGS = os.path.join(ROOT, "embeddings", "embeddings.npy")

INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
METRIC = "cosine"
BATCH_SIZE = 200

# Ініціалізація клієнта.
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def ensure_index() -> None:
    """Створюємо індекс, якщо він ще не існує, і чекаємо на готовність."""
    if not pc.has_index(INDEX_NAME):
        print(f"Створюємо індекс '{INDEX_NAME}' (dim={VECTOR_DIM}, metric={METRIC})...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Чекаємо, поки індекс перейде у стан ready.
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
        print("Індекс готовий.")
    else:
        print(f"Індекс '{INDEX_NAME}' вже існує — підключаємось.")


def truncate(value, limit: int) -> str:
    """Безпечно перетворює значення на рядок і обрізає до limit символів."""
    return str(value)[:limit]


def main() -> None:
    ensure_index()
    index = pc.Index(INDEX_NAME)

    df = pd.read_parquet(INPUT_PARQUET).reset_index(drop=True)
    embeddings = np.load(INPUT_EMBEDDINGS)
    assert len(df) == len(embeddings), "Кількість статей і ембедингів не збігається!"
    print(f"Готуємо до завантаження: {len(df)} векторів")

    # Завантажуємо батчами по BATCH_SIZE.
    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Upsert у Pinecone"):
        end = min(start + BATCH_SIZE, len(df))
        vectors = []
        for i in range(start, end):
            row = df.iloc[i]
            vectors.append({
                "id": f"paper_{i}",
                "values": embeddings[i].tolist(),
                "metadata": {
                    "arxiv_id": str(row["id"]),
                    "title": truncate(row["title"], 500),
                    "abstract": truncate(row["abstract"], 500),  # ліміт метаданих Pinecone
                    "authors": truncate(row["authors"], 200),
                    "year": int(row["year"]),
                    "category": str(row["category"]),
                },
            })
        index.upsert(vectors=vectors)

    # Pinecone оновлює статистику з невеликою затримкою — даємо їй "осісти".
    time.sleep(5)
    stats = index.describe_index_stats()
    print(f"\nЗавантаження завершено. Усього векторів в індексі: {stats['total_vector_count']}")


if __name__ == "__main__":
    main()
