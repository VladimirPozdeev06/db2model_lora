"""Reproducible smoke test of the MCP Text2SQL server's live path.

Exercises the REAL `Text2SQLGenerator.query()` used by the MCP tool:
    вопрос → LLM (эндпоинт из .env) → SQL → исполнение на живой БД → результат.

Covers all three target DBs in BOTH arms, with the same bar in each: the query must
execute and return real rows.
  * `with_schema=True`  — baseline: the server pulls the schema into the prompt.
  * `with_schema=False` — the defended arm: schema comes from the adapter's weights,
    so this passes only against a vLLM serving the fine-tuned adapter
    (`db2model/kaggle_vllm_serve.ipynb`). Against the plain endpoint model it FAILS
    by design — the base model invents table names.

Needs the ssh tunnel (localhost:5444) and the LLM endpoint (LLM_* in .env).
Run:  PYTHONIOENCODING=utf-8 uv run --env-file .env python db2model/mcp_smoke_test.py

The endpoint model (Qwen3) reasons by default, so enable_thinking=false is injected
into every create() call — otherwise `content` comes back empty. The MCP server
itself would need the same flag when pointed at a Qwen3 model.
"""
import asyncio
import os
import sys

from autogen_ext.models.openai import OpenAIChatCompletionClient

from adv_text2sql.mcp_servers.text2sql_tool.src.text2sql_implementation import (
    Text2SQLGenerator,
)

DB_HOST = os.getenv("BENCHMARK_DB_URL", "localhost:5444")
# (db, вопрос с заведомым ответом) — по одному на каждую целевую базу.
CASES = [
    ("toxicology", "List 3 distinct bond types."),
    ("financial", "How many accounts are there in total?"),
    ("codebase_community", "How many users are there in total?"),
]


def make_client(model: str | None = None) -> OpenAIChatCompletionClient:
    client = OpenAIChatCompletionClient(
        model=model or os.environ["LLM_MODEL_NAME"],
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        temperature=0.0,
        model_info={"json_output": False, "function_calling": True, "vision": False,
                    "family": "unknown", "structured_output": False},
    )
    _orig = client.create

    async def _create(messages, **kw):
        ec = dict(kw.get("extra_create_args") or {})
        ec.setdefault("extra_body", {"chat_template_kwargs": {"enable_thinking": False}})
        kw["extra_create_args"] = ec
        return await _orig(messages, **kw)

    client.create = _create
    return client


def _first_line(sql: str | None) -> str:
    lines = (sql or "").splitlines()
    return lines[0][:90] if lines else "<пусто>"


def _agent(db: str, with_schema: bool, client) -> Text2SQLGenerator:
    uri = (f"postgresql+psycopg://{os.environ['DB_USER']}:{os.environ['DB_PASS']}"
           f"@{DB_HOST}/{db}")
    agent = Text2SQLGenerator(db_uri=uri, llm_client=client, with_schema=with_schema)
    agent.build()
    return agent


async def main() -> int:
    client = make_client()
    # Арму со схемой нужна ОБЩАЯ instruct-модель: дообученный адаптер на длинный
    # baseline-промпт не рассчитан. Когда vLLM отдаёт и базу, и адаптер (см.
    # kaggle_vllm_serve.ipynb), базу видно под именем `base` — тогда оба арма
    # честно проверяются на одном сервере.
    baseline_model = os.getenv("LLM_MODEL_NAME_WITH_SCHEMA")
    baseline_client = make_client(baseline_model) if baseline_model else client
    passed = 0

    print("=== with_schema=True (baseline-режим): ждём реальные строки ===")
    if baseline_model:
        print(f"    (модель арма со схемой: {baseline_model})")
    for db, question in CASES:
        agent = _agent(db, True, baseline_client)
        res = await agent.query(question, check_ambiguity=False, check_sql_query=False)
        ex = res.get("execution", {})
        ok = res.get("status") == "success" and ex.get("status") == "success" \
            and ex.get("row_count", 0) > 0
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {db}: {question}")
        print(f"       SQL: {_first_line(res.get('query'))}")
        print(f"       -> {ex.get('row_count', 0)} строк | {str(ex.get('results', ex.get('error')))[:90]}")

    print("\n=== with_schema=False (schema-less арм: знание в весах адаптера) ===")
    schema_less = 0
    for db, question in CASES:
        agent = _agent(db, False, client)
        res = await agent.query(question, check_ambiguity=False, check_sql_query=False)
        ex = res.get("execution", {})
        ok = res.get("status") == "success" and ex.get("status") == "success" \
            and ex.get("row_count", 0) > 0
        schema_less += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {db}: {question}")
        print(f"       SQL: {_first_line(res.get('query'))}")
        print(f"       -> {ex.get('row_count', 0)} строк | {str(ex.get('results', ex.get('error')))[:90]}")

    total = len(CASES)
    print(f"\nИТОГ: with_schema=True {passed}/{total} PASS; "
          f"schema-less {schema_less}/{total} PASS")
    if schema_less < total:
        print(
            "\nSchema-less арм требует ДООБУЧЕННЫЙ адаптер: LLM_MODEL_NAME=db2model и\n"
            "LLM_BASE_URL на vLLM из db2model/kaggle_vllm_serve.ipynb. На базовой модели\n"
            "эндпоинта без адаптера имена таблиц будут выдуманы — это ожидаемо и не баг."
        )
    return 0 if (passed == total and schema_less == total) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
