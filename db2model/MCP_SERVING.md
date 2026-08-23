# MCP-сервер для победившей конфигурации

DoD недель 5–6 требует обернуть выбранную связку «модель + способ подачи знания о БД»
в MCP-сервер. Скелет уже есть: `src/adv_text2sql/mcp_servers/text2sql_tool/main.py`
(FastMCP, инструмент `text2sql`, ходит в модель через OpenAI-совместимый клиент).
Что подключать — зависит от того, какой арм победит по лидерборду.

## Случай A — победил baseline или light-RAG (схема/подграф в контексте)

Существующий сервер работает **как есть**: он подаёт схему в промпт через
`Text2SQLGenerator.build()`. Достаточно указать в `.env` рабочий `LLM_BASE_URL`
(эндпоинт МФТИ или локальный) и запустить:

```bash
PYTHONIOENCODING=utf-8 uv run --env-file .env python -m adv_text2sql.mcp_servers.text2sql_tool.main
```

Для light-RAG подменяется источник контекста (подграф вместо полной схемы) — это
на стороне ветки напарницы.

## Случай B — победил арм LoRA (знание в весах, без схемы)

Знание в весах ⇒ на инференсе схема НЕ подаётся, а модель — базовая 3B + адаптер.
Стенд ходит по OpenAI-совместимому API, поэтому адаптер поднимаем как такой сервер
через vLLM (не тащим `transformers`/`peft` внутрь MCP):

```bash
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
    --enable-lora \
    --lora-modules db2model=db2model/adapters/qwen27b_7b_sqlonly1096 \
    --max-lora-rank 16 \
    --tensor-parallel-size 2 \
    --port 8001
```

Адаптер здесь — победитель лидерборда (7B, EX 41.03 ± 1.37). База обязана совпадать с
той, на которой он обучен: 7B-адаптер на 3B-базе vLLM проглотит молча, а качество
упадёт до мусора. На Kaggle это делает `kaggle_vllm_serve.ipynb` (там же cloudflared
наружу). 3B-вариант — адаптер `qwen27b_sqlonly1096`, база `Qwen2.5-Coder-3B-Instruct`,
`--tensor-parallel-size 1`.

`.env` (плюс `DB_USER`, `DB_PASS`, `BENCHMARK_DB_URL` для исполнения SQL через туннель):

```
LLM_BASE_URL=http://localhost:8001/v1
LLM_MODEL_NAME=db2model
TARGET_DB=financial            # какую из 3 баз обслуживает сервер
TEXT2SQL_WITH_SCHEMA=false     # арм весов: схема НЕ подаётся, берётся из адаптера
```

Режим без схемы реализован: `Text2SQLGenerator(..., with_schema=False)` (или флаг
в `build(with_schema=False)`) собирает системный промпт из `SYSTEM_PROMPT_NOSCHEMA_TEMPLATE`
— без блока Schema, знание берётся из весов. `main.py` читает `TEXT2SQL_WITH_SCHEMA`
и `TARGET_DB` из `.env`, строит `db_uri`, вызывает `build()` и исполняет SQL (`execute_safe`,
только SELECT). ambiguity-check и ретраи отключены (`check_ambiguity=False`).

Запуск сервера:

```bash
PYTHONIOENCODING=utf-8 uv run --env-file .env python -m adv_text2sql.mcp_servers.text2sql_tool.main
```

Путь исполнения проверен через туннель (schema-less промпт + `execute_safe`); не проверён
только сам вызов LLM — нужен поднятый vLLM с адаптером.

Требует GPU (локально или Kaggle-сессия с ssh-туннелем к БД для исполнения SQL) —
поэтому это демо-выход, а не то, что двигает числа в отчёте.

## Smoke-тест

```bash
# сервер поднят — дёрнуть инструмент через MCP-клиент или curl к /mcp
```

Проверить на 3–5 живых вопросах, что `text2sql` возвращает исполнимый SQL и таблицу
результата. Воспроизводимость: зафиксировать версию модели, адаптер (`adapter_card.json`)
и `.env` в описании прогона.
