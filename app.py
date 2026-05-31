# app.py
"""
Bonus (UI, етап 2-3). Flask-бекенд для інтерактивної карти ембедингів.

Віддає статичний фронтенд (ui/) і реалізує три методи пошуку через /api/search:
  * lexical   — BM25 по заголовку+анотації (локально);
  * semantic  — ембединг запиту SPECTER2 → пошук у Pinecone;
  * hybrid    — BM25 + векторний, об'єднані через Reciprocal Rank Fusion (RRF).

Запуск:  python app.py   (на цій машині через песочницю: python -P app.py)
Потім відкрити http://localhost:5000
"""

import os
import re

import numpy as np
import pandas as pd
import torch
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

ROOT = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(ROOT, "ui")
INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
RRF_K = 60

print("Завантаження ресурсів (датасет, BM25, модель, Pinecone)...")
df = pd.read_parquet(os.path.join(ROOT, "data", "arxiv_subset.parquet")).reset_index(drop=True)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


tokenized_corpus = [tokenize(t) for t in (df["title"] + " " + df["abstract"]).tolist()]
bm25 = BM25Okapi(tokenized_corpus)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
print(f"Готово. Пристрій: {device}. Статей: {len(df)}")


def hit(i: int, score: float, rank: int) -> dict:
    """Формує запис результату з метаданими статті за позицією i.

    Повний abstract та авторів беремо з parquet (df) — це ілюструє підхід Частини 2:
    важкі поля зберігаємо окремо й підтягуємо за id уже після пошуку. arxiv_id дає
    пряме посилання на першоджерело (https://arxiv.org/abs/<arxiv_id>).
    """
    row = df.iloc[i]
    return {
        "id": f"paper_{i}", "rank": rank, "score": round(float(score), 4),
        "arxiv_id": str(row["id"]),
        "title": str(row["title"]), "abstract": str(row["abstract"]),
        "authors": str(row["authors"]),
        "category": str(row["category"]), "year": int(row["year"]),
    }


def search_bm25(query: str, k: int) -> list[dict]:
    scores = bm25.get_scores(tokenize(query))
    top = np.argsort(scores)[::-1][:k]
    return [hit(int(i), scores[i], r) for r, i in enumerate(top, 1)]


def encode(query: str) -> list[float]:
    return model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()


def search_semantic(query: str, k: int) -> list[dict]:
    res = index.query(vector=encode(query), top_k=k, include_metadata=False)
    out = []
    for r, m in enumerate(res["matches"], 1):
        i = int(m["id"].split("_")[1])
        out.append(hit(i, m["score"], r))
    return out


def search_hybrid(query: str, k: int) -> list[dict]:
    pool = max(k, 50)
    bm = [f"paper_{int(i)}" for i in np.argsort(bm25.get_scores(tokenize(query)))[::-1][:pool]]
    res = index.query(vector=encode(query), top_k=pool, include_metadata=False)
    vec = [m["id"] for m in res["matches"]]
    scores = {}
    for ranking in (bm, vec):
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [hit(int(doc_id.split("_")[1]), s, r) for r, (doc_id, s) in enumerate(fused, 1)]


METHODS = {"lexical": search_bm25, "semantic": search_semantic, "hybrid": search_hybrid}

app = Flask(__name__)


@app.route("/")
def home():
    return send_from_directory(UI_DIR, "index.html")


@app.route("/<path:fname>")
def static_files(fname):
    return send_from_directory(UI_DIR, fname)


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    method = data.get("method", "semantic")
    k = int(data.get("k", 20))
    if not query:
        return jsonify({"results": [], "query": query, "method": method})
    if method not in METHODS:
        return jsonify({"error": f"unknown method: {method}"}), 400
    results = METHODS[method](query, k)
    return jsonify({"results": results, "query": query, "method": method})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
