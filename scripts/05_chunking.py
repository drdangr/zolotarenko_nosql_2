# scripts/05_chunking.py
"""
Частина 4. Chunking (розбиття довгих анотацій на частини).

Беремо 30 статей із найдовшими анотаціями і ріжемо їх двома стратегіями:
  * Fixed-size  — фіксована кількість слів із невеликим перекриттям (overlap);
  * Semantic    — об'єднання цілих речень до ліміту слів (речення не розриваємо).

Кожна стратегія завантажується в окремий індекс Pinecone
(arxiv-chunks-fixed / arxiv-chunks-semantic), після чого виконуємо тестовий пошук.
"""

import os
import re
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
TOP_N_PAPERS = 30          # статей із найдовшими анотаціями
FIXED_SIZE = 60            # слів у fixed-size чанку
FIXED_OVERLAP = 15         # перекриття між сусідніми fixed-size чанками
SEMANTIC_MAX_WORDS = 60    # верхня межа слів у semantic-чанку
BATCH_SIZE = 100

INDEX_FIXED = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)
df = pd.read_parquet(os.path.join(ROOT, "data", "arxiv_subset.parquet")).reset_index(drop=True)


# --------------------------------------------------------------------------
# Стратегії розбиття
# --------------------------------------------------------------------------
def chunk_fixed(text: str, size: int = FIXED_SIZE, overlap: int = FIXED_OVERLAP) -> list[str]:
    """Фіксована кількість слів зі зсувом (size - overlap). Речення можуть розриватися."""
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start:start + size]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + size >= len(words):
            break
    return chunks


def split_sentences(text: str) -> list[str]:
    """Грубе розбиття на речення за крапкою/знаком оклику/питання."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_semantic(text: str, max_words: int = SEMANTIC_MAX_WORDS) -> list[str]:
    """Накопичуємо цілі речення, поки не наблизимось до max_words. Речення не розриваємо."""
    chunks, current, count = [], [], 0
    for sent in split_sentences(text):
        n = len(sent.split())
        if current and count + n > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sent)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


# --------------------------------------------------------------------------
# Завантаження чанків у Pinecone
# --------------------------------------------------------------------------
def ensure_index(name: str) -> None:
    if not pc.has_index(name):
        print(f"Створюємо індекс '{name}'...")
        pc.create_index(
            name=name, dimension=VECTOR_DIM, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)


def build_and_upload(index_name: str, chunker, papers: pd.DataFrame) -> int:
    """Ріже анотації обраним chunker'ом, кодує чанки й завантажує їх в індекс."""
    ensure_index(index_name)
    index = pc.Index(index_name)

    records = []  # (vector_id, chunk_text, metadata)
    for _, row in papers.iterrows():
        chunks = chunker(row["abstract"])
        for ci, chunk_text in enumerate(chunks):
            records.append((
                f"{row['id']}_chunk_{ci}",
                chunk_text,
                {
                    "arxiv_id": str(row["id"]),
                    "title": str(row["title"])[:300],
                    "chunk_text": chunk_text[:500],
                    "chunk_id": ci,
                    "year": int(row["year"]),
                    "category": str(row["category"]),
                },
            ))

    texts = [r[1] for r in records]
    vectors = model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True
    )

    for start in tqdm(range(0, len(records), BATCH_SIZE), desc=f"Upsert → {index_name}"):
        batch = []
        for j in range(start, min(start + BATCH_SIZE, len(records))):
            vid, _, meta = records[j]
            batch.append({"id": vid, "values": vectors[j].tolist(), "metadata": meta})
        index.upsert(vectors=batch)

    print(f"  {index_name}: завантажено {len(records)} чанків із {len(papers)} статей "
          f"(в середньому {len(records)/len(papers):.1f} чанків/статтю)")
    return len(records)


# --------------------------------------------------------------------------
# Пошук по чанках
# --------------------------------------------------------------------------
def search_chunks(index_name: str, query: str, top_k: int = 5) -> None:
    index = pc.Index(index_name)
    qv = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()
    res = index.query(vector=qv, top_k=top_k, include_metadata=True)
    for rank, m in enumerate(res["matches"], 1):
        meta = m["metadata"]
        print(f"  {rank}. score={m['score']:.4f}  [{meta['title'][:60]}] (chunk #{meta['chunk_id']})")
        print(f"     …{meta['chunk_text'][:160]}…")


def main() -> None:
    # 1. Обираємо 30 статей із найдовшими анотаціями (за кількістю слів).
    df["abstract_words"] = df["abstract"].str.split().str.len()
    longest = df.sort_values("abstract_words", ascending=False).head(TOP_N_PAPERS)
    print(f"Обрано {len(longest)} статей. Довжина анотацій (слів): "
          f"від {longest['abstract_words'].min()} до {longest['abstract_words'].max()}")

    # 2-5. Дві стратегії → два індекси.
    print("\n>>> Fixed-size chunking")
    build_and_upload(INDEX_FIXED, chunk_fixed, longest)
    print("\n>>> Semantic chunking")
    build_and_upload(INDEX_SEMANTIC, chunk_semantic, longest)
    time.sleep(5)

    # 6. Пошук по кількох тестових запитах в обох індексах.
    queries = [
        "deep learning for facial emotion recognition in videos",
        "convolutional neural networks for image analysis",
    ]
    for query in queries:
        print("\n" + "=" * 80)
        print(f"ЗАПИТ: {query!r}")
        print("=" * 80)
        print("\n[fixed]")
        search_chunks(INDEX_FIXED, query)
        print("\n[semantic]")
        search_chunks(INDEX_SEMANTIC, query)


if __name__ == "__main__":
    main()
