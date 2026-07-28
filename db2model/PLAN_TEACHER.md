# Сильный учитель + мультизадачный датасет (ROUTE)

Закрывает два расхождения ветки SFT/LoRA с 6-недельным планом:

1. **Учитель.** План требует «сильную API-LLM (DeepSeek-V3 / Claude / GPT-класс)».
   Текущий учитель — Qwen2.5-Coder-7B (слабый, зафиксировано как отклонение №1).
2. **Формат датасета.** Неделя 4 требует мультизадачный набор в духе ROUTE:
   «вопрос→SQL» **плюс** schema linking, noise correction, continuation writing.
   Сейчас в датасете только `question → sql`.

## Какой учитель выбран и почему

**Учитель: `Qwen/Qwen3.6-27B`** из локального пула МФТИ-эндпоинта.

Ограничения, которые сузили выбор (из переписки с куратором):
- **Внешние GPT (gpt-4.1/4o) для разметки недели 4 — не дают.** Платный внешний
  API разрешён только под недели 2–3. Для «затратной разметки» — только локально
  поднятые модели из пула или свои в Colab. → gpt-4.1 отпадает не по желанию.
- **`deepseek-ai/DeepSeek-R1-Distill-Llama-70B` на шлюзе отдаёт битый текст:**
  пробелы приходят как `U+0120` (`Ġ`), переносы как `U+010A` (`Ċ`) — проверено на
  тривиальном «hello world». Плюс медленно (15–27 с/вызов) и игнорирует JSON.
  → как учитель непригоден.

Почему Qwen3.6-27B (проверено эмпирически на этом же ключе):
- **чистый вывод**, без токенайзер-артефактов;
- **быстрый** — ~2 с на пару (18 с на батч из 8), против 15–27 с у DeepSeek;
- **уважает формат** — отдаёт валидный JSON-массив, парсер берёт 2/2;
- **27B ≫ нынешней 7B** — закрывает «слабый учитель» (сильный→слабый дистилл в
  рамках open-моделей, как в ROUTE);
- бесплатно (локальный пул), совпадает с моделью Алёны для недель 2–3.

⚠️ Qwen3 — reasoning по умолчанию: без флага весь ответ уходит в `<think>`, а
`content` приходит пустой. `generate_pairs.py` шлёт `enable_thinking=false`
(константа `EXTRA_BODY`) — на шлюзе это гасит reasoning, прочие модели флаг
игнорируют, а не падают.

## Что уже готово

- `generate_pairs.py` — учитель через OpenAI-совместимый клиент (`LLM_*` в `.env`)
  + `enable_thinking=false`. Проверено end-to-end на профиле toxicology: 8 чистых
  пар с валидным SQL по реальным таблицам.
- `build_multitask.py` — детерминированно выводит из отфильтрованных пар три
  вспомогательные задачи ROUTE (учитель и БД не нужны):
  - **schema linking** — `question → tables/columns` (из gold-SQL);
  - **noise correction** — `(question + почти-верный SQL) → исправленный`
    (реальный столбец подменяется на другой реальный; таргет — gold-пара,
    прошедшая execution-фильтр);
  - **continuation writing** — `(question + начало SQL) → хвост` (разрез на
    границе клаузы).
- `build_dataset.py --multitask` — подмешивает aux-задачи, выводя их **только из
  train-части** после сплита (нет утечки val→train). В `manifest.json` — `multitask`,
  `n_multitask`, `by_task`.
- `kaggle_train_lora.ipynb`, `kaggle_train_7b.ipynb` — `to_text` ветвится по `task`,
  обратно-совместимо.

## Runbook

```bash
# .env уже настроен: LLM_MODEL_NAME = "Qwen/Qwen3.6-27B"

# 1. Перегенерировать «вопрос→SQL» новым учителем (перезапишет raw/)
for db in financial toxicology codebase_community; do
  uv run --env-file .env python db2model/generate_pairs.py $db 500
done

# 2. Execution-фильтр на живой БД (нужен ssh-туннель на localhost:5444)
for db in financial toxicology codebase_community; do
  uv run --env-file .env python db2model/filter_pairs.py $db
done

# 3. Собрать датасет С мультизадачами
uv run python db2model/build_dataset.py --multitask

# 4. (опц.) посмотреть сами aux-задачи
uv run python db2model/build_multitask.py   # -> dataset/multitask.json

# 5. Обновить kaggle_input/ и обучить на Kaggle (lora, затем 7b)
#    Строки в лидерборд: qwen27b_sql_only и qwen27b_multitask.
```

## Что это даёт для отчёта

- **Отклонение №1 частично закрыто:** учитель — Qwen3.6-27B (27B) вместо 7B. Это не
  GPT-класс (внешний API под разметку не дали), но сильный open-teacher; вклад
  корректно называть дистилляцией «сильный→слабый» в рамках open-моделей.
- **Методология ROUTE выполнена по букве:** мультизадачный SFT (SQL-generation +
  schema linking + noise correction + continuation writing).
- **Честный ablation на защиту:** `qwen7b_sql_only` (текущие 27.84 ± 0.64) →
  `qwen27b_sql_only` → `qwen27b_multitask`. Разделяет вклад учителя и мультизадач.
