# scripts/06_hybrid_search.py
"""
Частина 5. Гібридний пошук: BM25 + векторний (Pinecone), об'єднані через
Reciprocal Rank Fusion (RRF).

BM25 сильний на точних термінах, абревіатурах та іменах авторів; векторний пошук —
на перефразуваннях і синонімах. RRF поєднує обидва ранжовані списки за формулою:
    RRF(d) = Σ 1 / (k + rank_i(d))
де rank_i(d) — позиція документа d у списку методу i (1-based), k — згладжувальний
параметр (класично 60).
"""

import os
import re

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10        # беремо ширше, щоб RRF мав що переранжовувати
RRF_K = 60        # згладжувальний параметр RRF

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)
df = pd.read_parquet(os.path.join(ROOT, "data", "arxiv_subset.parquet")).reset_index(drop=True)


# --------------------------------------------------------------------------
# Локальний BM25-індекс по заголовках + анотаціях усіх статей
# --------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


corpus_texts = (df["title"] + " " + df["abstract"]).tolist()
tokenized_corpus = [tokenize(t) for t in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)


# --------------------------------------------------------------------------
# Окремі методи пошуку → повертають впорядкований список doc_id ("paper_<i>")
# --------------------------------------------------------------------------
def search_bm25(query: str, top_k: int = TOP_K) -> list[str]:
    scores = bm25.get_scores(tokenize(query))
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [f"paper_{i}" for i in top_idx]


def search_vector(query: str, top_k: int = TOP_K) -> list[str]:
    qv = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()
    res = index.query(vector=qv, top_k=top_k, include_metadata=False)
    return [m["id"] for m in res["matches"]]


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Об'єднує кілька ранжованих списків doc_id у спільний рейтинг за формулою RRF."""
    scores: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def search_hybrid(query: str, top_k: int = TOP_K, k: int = RRF_K):
    bm25_ids = search_bm25(query, top_k)
    vec_ids = search_vector(query, top_k)
    fused = reciprocal_rank_fusion([bm25_ids, vec_ids], k=k)
    return fused, bm25_ids, vec_ids


# --------------------------------------------------------------------------
# Виведення
# --------------------------------------------------------------------------
def doc_idx(doc_id: str) -> int:
    return int(doc_id.split("_")[1])


def show_ids(ids: list[str], limit: int = 5) -> None:
    for rank, doc_id in enumerate(ids[:limit], 1):
        row = df.iloc[doc_idx(doc_id)]
        print(f"  {rank}. {doc_id}  [{row['category']}, {row['year']}]  {row['title'][:70]}")


def show_rrf(fused: list[tuple[str, float]], limit: int = 5) -> None:
    for rank, (doc_id, score) in enumerate(fused[:limit], 1):
        row = df.iloc[doc_idx(doc_id)]
        print(f"  {rank}. {doc_id}  RRF={score:.5f}  [{row['category']}, {row['year']}]  {row['title'][:60]}")


def main() -> None:
    queries = [
        "BERT fine-tuning",                                  # точний термін
        "Yann LeCun convolutional networks",                 # ім'я автора
        "making computers understand human emotions from text",  # перефразування
    ]
    for query in queries:
        fused, bm25_ids, vec_ids = search_hybrid(query)
        print("=" * 80)
        print(f"ЗАПИТ: {query!r}")
        print("=" * 80)
        print("\n[BM25] топ-5:")
        show_ids(bm25_ids)
        print("\n[Векторний] топ-5:")
        show_ids(vec_ids)
        print("\n[Гібридний RRF] топ-5:")
        show_rrf(fused)
        print()


if __name__ == "__main__":
    main()
