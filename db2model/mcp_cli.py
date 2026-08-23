"""Интерактивная консоль к MCP-серверу text2sql: вопрос -> таблица результата.

Нужна, чтобы арм можно было ПОКАЗАТЬ. Сам MCP — протокол для агентов, интерфейса у
него нет; здесь минимальный клиент, который ходит к серверу тем же путём, что ходил бы
агент, и печатает ответ инструмента как есть.

Сервер поднимают отдельным окном:

    PYTHONIOENCODING=utf-8 uv run --env-file .env python -m adv_text2sql.mcp_servers.text2sql_tool.main

Затем:

    PYTHONIOENCODING=utf-8 uv run --env-file .env python db2model/mcp_cli.py
    PYTHONIOENCODING=utf-8 uv run --env-file .env python db2model/mcp_cli.py -q "How many accounts?"

Обслуживаемая база задаётся сервером (TARGET_DB в .env), а не здесь: адаптер знает
схему той базы, с которой сервер поднят.
"""

import argparse
import asyncio
import sys
import time

from fastmcp import Client

DEFAULT_URL = "http://127.0.0.1:8000/mcp/"
TOOL_NAME = "text2sql"


def extract_text(result: object) -> str:
    """Ответ инструмента -> строка (форма результата зависит от версии fastmcp)."""
    content = getattr(result, "content", None)
    if content:
        return "\n".join(getattr(block, "text", str(block)) for block in content)
    data = getattr(result, "data", None)
    return str(data) if data is not None else str(result)


async def ask_once(client: Client, question: str) -> None:
    started = time.perf_counter()
    answer = extract_text(await client.call_tool(TOOL_NAME, {"user_query_text": question}))
    print(answer)
    print(f"\n({time.perf_counter() - started:.1f} с)")


async def run(url: str, question: str | None) -> int:
    try:
        async with Client(url) as client:
            tools = [tool.name for tool in await client.list_tools()]
            if TOOL_NAME not in tools:
                print(f"на сервере нет инструмента {TOOL_NAME}, есть: {tools}")
                return 1

            if question:
                await ask_once(client, question)
                return 0

            print(f"подключено к {url}, инструмент: {TOOL_NAME}")
            print("Спрашивайте про базу обычным текстом. Пустая строка или Ctrl+C — выход.\n")
            while True:
                try:
                    question = input(">> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return 0
                if not question:
                    return 0
                await ask_once(client, question)
                print()
    except Exception as exc:
        print(f"сервер не отвечает на {url}: {type(exc).__name__}: {exc}")
        print("Поднят ли он? См. docstring этого файла.")
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Консоль к MCP-серверу text2sql")
    parser.add_argument("--url", default=DEFAULT_URL, help="адрес MCP-сервера")
    parser.add_argument("-q", "--question", default=None, help="разовый вопрос вместо диалога")
    args = parser.parse_args()
    return asyncio.run(run(args.url, args.question))


if __name__ == "__main__":
    sys.exit(main())
