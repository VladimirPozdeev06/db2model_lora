"""Smoke-тест MCP-сервера как процесса: клиент ходит к нему по HTTP.

Отличие от `mcp_smoke_test.py`: тот импортирует `Text2SQLGenerator` напрямую и
проверяет путь «вопрос → LLM → SQL → БД», но сервер при этом не запускается вообще.
Здесь проверяется именно то, что требует DoD недель 5-6, — поднятый MCP-сервер,
зарегистрированный инструмент и ответ через протокол.

Сервер поднимают отдельным окном:

    PYTHONIOENCODING=utf-8 uv run --env-file .env python -m adv_text2sql.mcp_servers.text2sql_tool.main

Затем:

    PYTHONIOENCODING=utf-8 uv run --env-file .env python db2model/mcp_server_smoke_test.py

Код возврата 0 — инструмент вернул таблицу с данными; 1 — сервер отвечает, но запрос
не отработал (обычно недоступен LLM-эндпоинт); 2 — сервер не поднят.
"""

import argparse
import asyncio
import sys

from fastmcp import Client

DEFAULT_URL = "http://127.0.0.1:8000/mcp/"
DEFAULT_QUESTION = "How many accounts are there in total?"
TOOL_NAME = "text2sql"
ERROR_MARKERS = ("_Ошибка_", "_Ошибка выполнения_", "_Запрос неоднозначен_")


def extract_text(result: object) -> str:
    """Ответ инструмента → строка (форма результата зависит от версии fastmcp)."""
    content = getattr(result, "content", None)
    if content:
        return "\n".join(getattr(block, "text", str(block)) for block in content)
    data = getattr(result, "data", None)
    return str(data) if data is not None else str(result)


async def run_smoke(url: str, question: str) -> int:
    try:
        async with Client(url) as client:
            tools = await client.list_tools()
            names = [tool.name for tool in tools]
            print(f"сервер отвечает, инструменты: {names}")
            if TOOL_NAME not in names:
                print(f"[FAIL] инструмент {TOOL_NAME} не зарегистрирован")
                return 1

            print(f"\nвопрос: {question}")
            answer = extract_text(await client.call_tool(TOOL_NAME, {"user_query_text": question}))
    except Exception as exc:
        print(f"[FAIL] сервер не отвечает на {url}: {type(exc).__name__}: {exc}")
        print("Поднят ли он? См. docstring этого файла.")
        return 2

    print(f"ответ:\n{answer}")
    if any(answer.startswith(marker) for marker in ERROR_MARKERS):
        print("\n[FAIL] сервер жив, но запрос не отработал — обычно это недоступный LLM-эндпоинт")
        return 1

    print("\n[PASS] инструмент вернул данные")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="адрес MCP-сервера")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="вопрос к базе")
    args = parser.parse_args()
    return asyncio.run(run_smoke(args.url, args.question))


if __name__ == "__main__":
    sys.exit(main())
