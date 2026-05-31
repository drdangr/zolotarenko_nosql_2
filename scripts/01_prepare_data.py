# scripts/01_prepare_data.py
"""
Частина 1.1. Завантаження та підготовка датасету arXiv.

Зчитує сирий JSONL-дамп arXiv (arxiv-metadata-oai-snapshot.json), відбирає
MAX_RECORDS валідних статей (із заголовком та анотацією) і зберігає компактну
підмножину у форматі Parquet із полями: id, title, abstract, authors, year, category.

ВАЖЛИВО про відбір записів:
    Дамп arXiv відсортований за id, тобто старі статті (2007 р.) лежать на початку
    файлу. Якщо просто взяти перші 10 000 рядків, уся підмножина виявиться з
    одного-двох років, і фільтри за роком у Частині 3 ("за останні 5 років",
    "до 2015 р.") втратять сенс. Тому ми робимо РІВНОМІРНУ ВИБІРКУ по всьому файлу
    (reservoir sampling, алгоритм R) з фіксованим сидом — це дає відтворюваний зріз
    статей за всі роки існування arXiv.
"""

import os
import json
import random

import pandas as pd
from tqdm import tqdm

# Корінь проєкту = батьківська тека для scripts/ — щоб шляхи працювали
# незалежно від поточної робочої директорії.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(ROOT, "data", "arxiv-metadata-oai-snapshot.json")
OUTPUT_FILE = os.path.join(ROOT, "data", "arxiv_subset.parquet")
MAX_RECORDS = 10_000
RANDOM_SEED = 42  # відтворюваність вибірки

os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)


def extract_year(paper: dict) -> int:
    """
    Беремо рік з першої версії статті — це дата першої публікації на arXiv.
    (update_date — дата останнього оновлення, вона може бути на роки пізніше.)
    Формат поля created: "Mon, 2 Apr 2007 19:18:42 GMT" — рік стоїть 4-м після split.
    """
    try:
        versions = paper.get("versions", [])
        if versions:
            created = versions[0]["created"]
            return int(created.split()[3])
    except (IndexError, ValueError, KeyError):
        pass
    # Запасний варіант: update_date у форматі "YYYY-MM-DD"
    return int(paper.get("update_date", "2000-01-01")[:4])


def format_authors(paper: dict) -> str:
    """
    authors_parsed — структурований список [["Прізвище", "Ініціали", ""]].
    Збираємо у читабельний рядок "Прізвище І., Прізвище І." (не більше 10 авторів).
    Якщо authors_parsed відсутній — беремо сирий рядок authors.
    """
    parsed = paper.get("authors_parsed", [])
    if parsed:
        parts = []
        for entry in parsed[:10]:  # не більше 10 авторів
            last = entry[0].strip() if len(entry) > 0 else ""
            first = entry[1].strip() if len(entry) > 1 else ""
            if last:
                parts.append(f"{last} {first}".strip())
        return ", ".join(parts)
    # Запасний варіант: сирий рядок авторів
    return paper.get("authors", "").replace("\n", " ")


def build_record(paper: dict) -> dict | None:
    """Перетворює сирий JSON-об'єкт статті на компактний запис або None, якщо
    бракує заголовка/анотації."""
    abstract = paper.get("abstract", "").strip()
    title = paper.get("title", "").strip()
    if not abstract or not title:
        return None

    # categories може містити кілька категорій через пробіл ("cs.LG cs.AI") —
    # беремо першу як основну.
    categories_raw = paper.get("categories", "unknown")
    primary_category = categories_raw.split()[0] if categories_raw else "unknown"

    return {
        "id": paper["id"],
        "title": title.replace("\n", " ").strip(),
        "abstract": abstract.replace("\n", " ").strip(),
        "authors": format_authors(paper),
        "year": extract_year(paper),
        "category": primary_category,
    }


def main() -> None:
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Не знайдено {INPUT_FILE}.\n"
            "Завантажте датасет: `kaggle datasets download -d Cornell-University/arxiv` "
            "та розпакуйте arxiv-metadata-oai-snapshot.json у теку data/."
        )

    rng = random.Random(RANDOM_SEED)
    reservoir: list[dict] = []
    seen = 0  # скільки валідних записів переглянуто

    # Reservoir sampling (алгоритм R): один прохід по файлу, рівномірна вибірка
    # MAX_RECORDS записів з усього датасету незалежно від його розміру.
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Читаємо датасет"):
            line = line.strip()
            if not line:
                continue
            record = build_record(json.loads(line))
            if record is None:
                continue

            if len(reservoir) < MAX_RECORDS:
                reservoir.append(record)
            else:
                j = rng.randint(0, seen)  # 0..seen включно
                if j < MAX_RECORDS:
                    reservoir[j] = record
            seen += 1

    df = pd.DataFrame(reservoir)
    # Стабільне сортування за роком, потім за id — щоб порядок рядків був
    # детермінованим (важливо: позиція рядка стає основою для id "paper_<i>" далі).
    df = df.sort_values(["year", "id"]).reset_index(drop=True)

    print(f"\nЗавантажено статей: {len(df)} (переглянуто валідних: {seen})")
    print("\nРозподіл за категоріями (топ-10):")
    print(df["category"].value_counts().head(10))
    print("\nРозподіл за роками (останні 10 років вибірки):")
    print(df["year"].value_counts().sort_index().tail(10))
    print("\nПриклад запису:")
    print(df.iloc[0].to_dict())

    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nЗбережено у {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
