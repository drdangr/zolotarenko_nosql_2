# scripts/02_embed.py
"""
Частина 1.3. Отримання ембедингів.

Зчитує підмножину статей, формує вхідний текст у форматі "title [SEP] abstract"
(саме на такому форматі навчена SPECTER2), кодує його моделлю allenai/specter2_base
у нормалізовані 768-вимірні вектори і зберігає їх у embeddings/embeddings.npy.

Порядок векторів у файлі точно відповідає порядку рядків у Parquet — це гарантує,
що embeddings[i] належить статті з рядка i (далі їй присвоюється id "paper_<i>").
"""

import os

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PARQUET = os.path.join(ROOT, "data", "arxiv_subset.parquet")
OUTPUT_DIR = os.path.join(ROOT, "embeddings")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")

MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64


def main() -> None:
    # 1. Завантажуємо датасет.
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Завантажено записів: {len(df)}")

    # 2. Готуємо тексти: title + " [SEP] " + abstract.
    #    Токен [SEP] обов'язковий — модель навчена розділяти заголовок і анотацію саме ним.
    texts = (df["title"].fillna("") + " [SEP] " + df["abstract"].fillna("")).tolist()

    # 3. Завантажуємо модель (на GPU, якщо доступний).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Пристрій: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # 4. Кодуємо: батчами, з прогресом, з L2-нормалізацією векторів.
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # 5. Діагностика.
    print(f"\nОброблено текстів: {embeddings.shape[0]}")
    print(f"Розмірність ембедингів: {embeddings.shape[1]} (очікується 768)")
    print(f"Норма першого ембединга: {np.linalg.norm(embeddings[0]):.6f} (очікується ~1.0)")

    # 6-7. Зберігаємо у NumPy-форматі, попередньо переконавшись, що тека існує.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(OUTPUT_FILE, embeddings)
    print(f"\nЗбережено у {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
