# Збережений вивід скриптів (чернетка для README)

> Реальний прогон 31.05.2026 на RTX 3090 Ti, Python 3.12, sentence-transformers 4.1.0, pinecone 9.0.1.
> Прогрес-бари (tqdm) обрізані для читабельності.

## 01_prepare_data.py
```
Читаємо датасет: 3058383it [00:30, ~100000it/s]

Завантажено статей: 10000 (переглянуто валідних: 3058383)

Розподіл за категоріями (топ-10):
category
hep-ph               521
cs.CV                481
cs.LG                461
quant-ph             421
hep-th               381
astro-ph             327
cs.CL                289
cond-mat.mtrl-sci    259
gr-qc                226
cond-mat.mes-hall    206

Розподіл за роками (останні 10 років вибірки):
year
2017    447
2018    478
2019    459
2020    582
2021    614
2022    591
2023    651
2024    809
2025    958
2026    423

Приклад запису:
{'id': 'math/9201218', 'title': 'The plank problem for symmetric bodies', ...,
 'year': 1990, 'category': 'math.MG'}

Збережено у data/arxiv_subset.parquet
```

## 02_embed.py
```
No sentence-transformers model found with name allenai/specter2_base. Creating a new one with mean pooling.
Завантажено записів: 10000
Пристрій: cuda

Оброблено текстів: 10000
Розмірність ембедингів: 768 (очікується 768)
Норма першого ембединга: 1.000000 (очікується ~1.0)

Збережено у embeddings/embeddings.npy
```

## 03_load_to_pinecone.py
```
Створюємо індекс 'arxiv-papers' (dim=768, metric=cosine)...
Індекс готовий.
Готуємо до завантаження: 10000 векторів
Upsert у Pinecone: 100% (50/50 батчів)

Завантаження завершено. Усього векторів в індексі: 10000
```

## 04_search.py
```
================================================================================
[3] ЧИСТИЙ СЕМАНТИЧНИЙ ПОШУК
Запит: 'teaching machines to recognize objects in pictures'
================================================================================
  1. [cs.CV, 2013]  score=0.8599  Neural perceptual model to global-local vision...
  2. [cs.CV, 2014]  score=0.8515  CITlab ARGUS for historical handwritten documents
  3. [cs.CV, 2016]  score=0.8469  Utilization of Deep Reinforcement Learning for saccadic-based object visual search
  4. [q-bio.NC, 2022]  score=0.8466  Simulating reaction time for Eureka effect in visual object recognition...
  5. [cs.CV, 2014]  score=0.8425  A Concept Learning Approach to Multisensory Object Perception

[4] ПОШУК ІЗ ФІЛЬТРАЦІЄЮ
--- Приклад A: 'reinforcement learning', рік >= 2022 ТА категорія cs.LG ---
  1. [cs.LG, 2023]  score=0.8634  Human-Inspired Framework to Accelerate Reinforcement Learning
  2. [cs.LG, 2022]  score=0.8489  Deep Reinforcement Learning for Distributed and Uncoordinated Cognitive Radios...
  3. [cs.LG, 2022]  score=0.8422  Reinforcement Learning Agent Design and Optimization with Bandwidth Allocation Model
  4. [cs.LG, 2024]  score=0.8362  Exploiting Structure in Offline Multi-Agent RL...
  5. [cs.LG, 2022]  score=0.8360  Multi-Agent Learning of Numerical Methods for Hyperbolic PDEs...

--- Приклад B: 'reinforcement learning', рік < 2015, будь-яка категорія ---
  1. [cs.LG, 2014]  score=0.8270  Safe Exploration of State and Action Spaces in Reinforcement Learning
  2. [cs.LG, 2013]  score=0.8217  Sample Complexity of Multi-task Reinforcement Learning
  3. [math.OC, 2008]  score=0.8176  Acceleration Operators in the Value Iteration Algorithms...
  4. [math.OC, 2011]  score=0.8050  KL-learning: Online solution of Kullback-Leibler control problems
  5. [cs.AI, 2009]  score=0.8012  What Does Artificial Life Tell Us About Death?

[5] ПОРІВНЯННЯ МЕТРИК НА ЛОКАЛЬНИХ ЕМБЕДИНГАХ
Запит: 'teaching machines to recognize objects in pictures'
cosine / dot / L2 — ІДЕНТИЧНИЙ топ-5 в ІДЕНТИЧНОМУ порядку:
  1. paper_2907  [cs.CV, 2013]  Neural perceptual model to global-local vision...
  2. paper_3268  [cs.CV, 2014]  CITlab ARGUS for historical handwritten documents
  3. paper_6855  [q-bio.NC, 2022]  Simulating reaction time for Eureka effect...
  4. paper_3909  [cs.CV, 2016]  Utilization of Deep Reinforcement Learning for saccadic...
  5. paper_3199  [cs.CV, 2014]  A Concept Learning Approach to Multisensory Object Perception
```

## 05_chunking.py
```
Обрано 30 статей. Довжина анотацій (слів): від 303 до 365

>>> Fixed-size chunking  → arxiv-chunks-fixed: 211 чанків (в середньому 7.0 чанків/статтю)
>>> Semantic chunking    → arxiv-chunks-semantic: 207 чанків (в середньому 6.9 чанків/статтю)

ЗАПИТ: 'deep learning for facial emotion recognition in videos'
[fixed]
  1. 0.8718  [AffWild Net and Aff-Wild Database] (chunk #0)  …Emotions recognition is the task...
  2. 0.8711  [AffWild...] (chunk #1)  …an emotion is and arousal shows how much it is activated. Recent deep learning...
  ...
[semantic]
  1. 0.8809  [AffWild...] (chunk #1)  …Recent deep learning models, that have to do with emotions recognition...
  2. 0.8465  [AffWild...] (chunk #0)  …Emotions recognition is the task of recognizing people's emotions...
  ...

ЗАПИТ: 'convolutional neural networks for image analysis'
[fixed]
  1. 0.8637  [AffWild...] (chunk #4)  …and deep learning models are presented. Then, inspired by them...
  2. 0.8531  [Analysing high resolution digital Mars images using machine learning] (chunk #3)  …a CNN is applied...
  ...
[semantic]
  1. 0.8551  [AffWild...] (chunk #4)
  3. 0.8352  [Analysing ... Mars images ...] (chunk #3)  …a convolutional neural network (CNN) is applied to find water ice patches...
  ...
```
Спостереження: semantic-чанки починаються з межі речення; fixed-чанки рвуть речення посередині.

## 06_hybrid_search.py
```
ЗАПИТ: 'BERT fine-tuning'  (точний термін)
[BM25]      8323 QA bridge design (LLM); 6113 Fine-Tuning Transformers; 6918 Z-BERT-A; 5951 BinaryBERT; 8107 Crowdsourcing
[Векторний] 6405 Q-learning; 3909 saccadic RL; 7264 RL framework; 8625 Test-Time Compute; 9551 LLM evolve  (ПРОМАХ по терміну)
[RRF]       1. 8323 (0.01639); 2. 6405 (0.01639); 3. 6113 (0.01613); 4. 3909 (0.01613); 5. 6918 (0.01587)

ЗАПИТ: 'Yann LeCun convolutional networks'  (ім'я автора)
[BM25]      5441; 7233; 6787; 2495; 5302  (усі про CNN)
[Векторний] 2495; 5441; 5866; 3268; 6291
[RRF]       1. 5441 (0.03252); 2. 2495 (0.03202); 3. 5866 (0.03102); 4. 7233 (0.01613); 5. 6787 (0.01587)

ЗАПИТ: 'making computers understand human emotions from text'  (перефразування)
[BM25]      7825 DepressionEmo; 4533 text-based emotion; 6842 Emoji attention; 5808 Embedded Emotions; 7664 MolCA
[Векторний] 6139 ECA Arthur; 4533 affective computing; 7998 UniMEEC; 5416 Continuous Emotion CV; 7873 Emojis ChatGPT
[RRF]       1. 4533 (0.03226); 2. 7825 (0.03154); 3. 5416 (0.03033); 4. 6139 (0.01639); 5. 6842 (0.01587)
```

## Фінальний стан індексів Pinecone
```
arxiv-papers            векторів: 10000  dim: 768  metric: cosine
arxiv-chunks-fixed      векторів:   211  dim: 768  metric: cosine
arxiv-chunks-semantic   векторів:   207  dim: 768  metric: cosine
```

## Витяг із картки моделі allenai/specter2_base (HuggingFace)
- "SPECTER2 is the successor to SPECTER and is capable of generating task specific embeddings for scientific tasks when paired with adapters."
- "Given the combination of title and abstract of a scientific paper or a short texual query, the model can be used to generate effective embeddings."
- "SPECTER2 has been trained on over 6M triplets of scientific paper citations ... trained with additionally attached task format specific adapter modules on all the SciRepEval training tasks." Формати задач: Classification, Regression, Proximity (Retrieval), Adhoc Search.
- Формат входу: `d['title'] + tokenizer.sep_token + d['abstract']`.
- Явно рекомендованої метрики схожості в картці БАЗОВОЇ моделі НЕМАЄ; для retrieval картка радить proximity-адаптер.
