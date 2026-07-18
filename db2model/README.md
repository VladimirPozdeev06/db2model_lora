# DB2Model — знание о БД в весах, а не в контексте

Ветка SFT/LoRA летней практики. Идея: teacher-модель по профилю конкретной PostgreSQL-БД
генерирует пары «вопрос → SQL», их фильтруют исполнением на живой базе, на очищенных парах
дообучается небольшая target-модель через QLoRA. Знание о базе оказывается в весах, поэтому
на инференсе схема в промпт не подаётся — на вход идёт только вопрос.

Сравнивается с baseline: та же target-модель со схемой в промпте (знание в контексте).

## Результаты

Всё на 91 вопросе из `bird_large` по трём базам (`financial`, `toxicology`, `codebase_community`),
target — Qwen2.5-Coder-3B-Instruct в 4-bit. Подробности и оговорки — в [../EXPERIMENTS.md](../EXPERIMENTS.md).

**Знание в весах против знания в контексте** (сид-скорректировано — см. прогон с сидами в
[../EXPERIMENTS.md](../EXPERIMENTS.md)):

| Арм | Схема в промпте | EX | prompt-токенов |
|---|---|---|---|
| baseline | да | 19.78% | 288 |
| lora | нет | 16.85% ± 1.37 | 91 |

lora обучен трижды (сиды 0/1/2): 15.38 / 18.68 / 16.48 → среднее **16.85 ± 1.37** (разброс 3.3 п.п.).
baseline детерминирован (greedy). Разрыв 19.78 − 16.85 = **2.93 п.п., больше 2σ — реален**: на этой
конфигурации схема в контексте пока устойчиво лучше схемы в весах на ~3 п.п. Тезис ветки — не «догнали
baseline», а «3B вообще удерживает схему трёх баз в весах, отставая контролируемо». Экономия токенов
(288 → 91) реальна, но у локальной 3B почти ничего не стоит — см. размеры схем, продавать её нельзя.

**Ablation** (одиночные прогоны, если не сказано иное; читать с поправкой на шум ±2 п.п.):

| Вариант | EX | вывод |
|---|---|---|
| synth_50 / synth_143 / synth_347 | 9.9 / 11.0 / 17.6% | эффект размера ~8 п.п. — **устойчив** (>2σ) |
| synth_347 vs unfiltered_347 | 16.9\* vs 15.4% | фильтр +1.5 п.п. — **внутри σ, не доказан** |
| real_143 vs synth_143 | 13.2% vs 11.0% | живые чище, но разница в пределах шума |

\* synth_347 взят по среднему трёх сидов (16.85), а не по единственному прогону (17.58), с которого
раньше выводился «фильтр +2.2 п.п.». На сид-скорректированном среднем эффект фильтра падает до +1.5 п.п.
и не выходит за σ = 1.37; unfiltered гонялся один раз, повторов нет — честный вывод: **не доказан**.
Устойчив только эффект размера данных. Различия < 2–3 п.п. на 91 вопросе — шум одного прогона.

## Пайплайн

Всё, кроме обучения, гоняется локально. Обучение — на Kaggle (T4/P100, 4-bit).
База доступна только через ssh-туннель, поэтому Kaggle её не видит: туда уезжает готовый датасет,
оттуда — предсказания, EX считается дома.

```
профиль БД ──► генерация пар ──► фильтр исполнением ──► датасет ──► QLoRA (Kaggle) ──► предсказания ──► EX (дома)
collect_profile   generate_pairs      filter_pairs      build_dataset  kaggle_*.ipynb                  bird_evaluate_only
```

### 0. Подготовка

Туннель к базе (пароль у куратора; `ServerAliveInterval` чтобы не отваливался):

```bash
ssh -N -o ServerAliveInterval=60 -L 5444:10.11.1.6:5444 <user>@lnsigo.mipt.ru -p2278
```

`.env` в корне репозитория (в `.gitignore`, секреты не коммитить): `DB_USER`, `DB_PASS`,
`BENCHMARK_DB_URL=localhost:5444`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL_NAME`.

На Windows все запуски — с `PYTHONIOENCODING=utf-8`, иначе `print` падает на cp1251-консоли.

### 1. Профиль БД (локально, нужен туннель)

```bash
uv run --env-file .env python db2model/collect_profile.py
```

Складывает в `db2model/profiles/<db>.json` схему, FK, `pg_stats` и реальные значения.
Проверить достаточность профиля (teacher отвечает по нему на реальные вопросы, без доступа к БД):

```bash
uv run --env-file .env python db2model/validate_profile.py toxicology 6
```

### 2. Генерация и фильтр (локально, нужен туннель)

```bash
uv run --env-file .env python db2model/generate_pairs.py financial 200
uv run --env-file .env python db2model/filter_pairs.py financial
```

`generate_pairs` кормит профиль teacher-модели и просит N пар; `filter_pairs` выполняет каждый SQL
на живой базе, брак выкидывает в `db2model/clean/<db>.rejected.json` с причинами (доля брака ~30%).

### 3. Сборка датасета (локально)

```bash
uv run python db2model/build_dataset.py       # train/val + manifest
uv run python db2model/build_ablation.py       # варианты для ablation
```

Готовые файлы для заливки собираются в `db2model/kaggle_input/`.

### 4. Обучение (Kaggle)

Залить `db2model/kaggle_input/` как Kaggle Dataset, включить GPU + Internet, поправить `DATA_DIR`
в первой ячейке под путь своего датасета. Затем запустить один из ноутбуков:

- `kaggle_train_lora.ipynb` — основной замер: baseline (со схемой) и lora (без) в одном прогоне.
- `kaggle_ablation.ipynb` — пять вариантов данных, по адаптеру на каждый.

Забрать домой `query_results_*.json` (и адаптер, если нужен артефакт).

### 5. Оценка (локально, нужен туннель)

Каждый файл предсказаний сначала через извлечение SQL (снимает прозу вокруг блоков кода),
потом EX:

```bash
uv run python db2model/extract_sql.py query_results_lora.json
uv run --env-file .env python bird_evaluate_only.py query_results_lora_extracted.json data/bird_large.json
```

## Структура

```
db2model/
  collect_profile.py    профиль БД из PostgreSQL
  validate_profile.py   проверка достаточности профиля
  generate_pairs.py     teacher генерирует пары по профилю
  filter_pairs.py       фильтр по исполнению (noise correction)
  build_dataset.py      сборка train/val + manifest
  build_ablation.py     варианты датасета для ablation
  extract_sql.py        извлечение SQL из ответа модели перед оценкой
  kaggle_train_lora.ipynb   основной замер (baseline vs lora)
  kaggle_ablation.ipynb     ablation (5 вариантов данных)
  profiles/  raw/  clean/  dataset/  results/   артефакты этапов
  kaggle_input/         то, что заливается на Kaggle
```

## Ограничения

- **Teacher = Qwen2.5-Coder-7B**, не GPT-4-class: на эндпоинте одна модель. Дистилляция «сильный →
  слабый» вырождается; честная формулировка вклада — перенос схемы в веса + noise correction.
- **Per-DB против multi-DB.** Обучен один адаптер сразу на три базы. Обобщение на новую БД без
  дообучения — вне скоупа.
- **Шум прогонов.** Главный замер усреднён по 3 сидам (`kaggle_seeds.ipynb`, σ = 1.37 п.п.);
  ablation-варианты (фильтр, real vs synth) — пока по одному сиду, их различия внутри шума.
- **Диалект.** Gold BIRD в SQLite приводится к PostgreSQL через sqlglot; редкие конструкции могут
  не транспилироваться.
