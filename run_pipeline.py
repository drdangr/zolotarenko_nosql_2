#!/usr/bin/env python
"""Послідовний запуск усього пайплайну ДЗ-2 (01→06) однією командою.

Приклади:
  python run_pipeline.py             # кроки 01–06 по черзі
  python run_pipeline.py --with-map  # + крок 07 (UMAP-карта для бонусного UI)
  python run_pipeline.py --from 4    # почати з 04 (напр., не переробляти ембединги)
  python run_pipeline.py --dry-run   # лише показати план, нічого не запускати

Кожен крок запускається в окремому процесі; за першої ж помилки виконання спиняється.
"""

import os
import sys
import time
import argparse
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(ROOT, "scripts")

STEPS = [
    ("01", "01_prepare_data.py",     "Підготовка датасету → data/arxiv_subset.parquet"),
    ("02", "02_embed.py",            "Ембединги SPECTER2 → embeddings/embeddings.npy"),
    ("03", "03_load_to_pinecone.py", "Створення індексу arxiv-papers і завантаження векторів"),
    ("04", "04_search.py",           "Пошукові запити (семантичний / фільтри / метрики)"),
    ("05", "05_chunking.py",         "Chunking: fixed vs semantic → окремі індекси"),
    ("06", "06_hybrid_search.py",    "Гібридний пошук BM25 + вектор через RRF"),
]
MAP_STEP = ("07", "07_export_map.py", "UMAP-проєкція ембедингів → ui/map_data.js (бонус)")


def preflight(step_nums: list[str]) -> list[str]:
    """Перевіряє наявність датасету та .env перед запуском відповідних кроків."""
    problems = []
    if "01" in step_nums and not os.path.exists(
        os.path.join(ROOT, "data", "arxiv-metadata-oai-snapshot.json")
    ):
        problems.append(
            "Немає data/arxiv-metadata-oai-snapshot.json — завантажте датасет "
            "(README, крок 3: kaggle datasets download -d Cornell-University/arxiv)."
        )
    # Кроки 03+ потребують ключа Pinecone.
    if any(n >= "03" for n in step_nums) and not os.path.exists(os.path.join(ROOT, ".env")):
        problems.append("Немає .env — скопіюйте .env.example у .env і впишіть PINECONE_API_KEY.")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description="Послідовний запуск пайплайну ДЗ-2 (01→06).")
    ap.add_argument("--from", dest="start", type=int, default=1, metavar="N",
                    help="почати з кроку N (1..6)")
    ap.add_argument("--with-map", action="store_true", help="додатково запустити крок 07 (карта)")
    ap.add_argument("--dry-run", action="store_true", help="лише показати план, нічого не запускати")
    args = ap.parse_args()

    steps = [s for s in STEPS if int(s[0]) >= args.start]
    if args.with_map:
        steps.append(MAP_STEP)
    if not steps:
        print("Немає кроків для запуску (перевірте --from).")
        return

    print("План запуску:")
    for num, script, desc in steps:
        print(f"  {num}  {script:24s} — {desc}")
    if args.dry_run:
        return

    problems = preflight([s[0] for s in steps])
    if problems:
        print("\n⚠️  Перед запуском усуньте:")
        for p in problems:
            print("   -", p)
        sys.exit(1)

    started = time.time()
    for num, script, desc in steps:
        print(f"\n{'=' * 72}\n[{num}] {desc}\n{'=' * 72}")
        t = time.time()
        result = subprocess.run([sys.executable, os.path.join(SCRIPTS, script)], cwd=ROOT)
        if result.returncode != 0:
            print(f"\n❌ Крок {num} ({script}) завершився з помилкою (код {result.returncode}). Спиняюсь.")
            sys.exit(result.returncode)
        print(f"✓ Крок {num} завершено за {time.time() - t:.0f} c")

    print(f"\n✅ Усе готово за {time.time() - started:.0f} c.")
    print("   Бонусний UI: python app.py  →  http://localhost:5000")


if __name__ == "__main__":
    main()
