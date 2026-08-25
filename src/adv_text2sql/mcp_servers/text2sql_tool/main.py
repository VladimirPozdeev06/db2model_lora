"""MCP-сервер инструмента `text2sql`.

Оборачивает победившую конфигурацию ветки db2model: вопрос на естественном языке →
SQL → исполнение на живой PostgreSQL → таблица результата. Какой арм обслуживается,
задаётся окружением:

* `TEXT2SQL_WITH_SCHEMA=false` — арм весов (LoRA): схема в промпт НЕ подаётся, знание
  о базе берётся из адаптера. `LLM_MODEL_NAME` должен указывать на обслуживаемый
  адаптер (например `db2model`), см. `db2model/kaggle_vllm_serve.ipynb`.
* `TEXT2SQL_WITH_SCHEMA=true` — baseline / RAG: схема инжектится в системный промпт.

Запуск (нужен ssh-туннель к БД и поднятый LLM-эндпоинт):

    PYTHONIOENCODING=utf-8 uv run --env-file .env python -m adv_text2sql.mcp_servers.text2sql_tool.main
"""

import logging
import os
from typing import Annotated

from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import Field
from tabulate import tabulate

from .src.text2sql_implementation import Text2SQLGenerator

logger = logging.getLogger(__name__)

DEFAULT_DB_HOST = "localhost:5444"
DEFAULT_TARGET_DB = "financial"
DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 8000

# Описание инструмента статическое и не зависит от LLM: прежняя версия звала
# `generate_description_text2sql`, которой в репозитории никогда не было, и сервер
# падал на импорте. Описание видит вызывающий агент, поэтому важны две вещи —
# что инструмент только читает и что возвращает готовую таблицу.
TOOL_DESCRIPTION = (
    "Отвечает на вопросы к базе данных на естественном языке: строит SQL-запрос, "
    "исполняет его и возвращает результат таблицей в формате markdown. "
    "Только чтение (SELECT), модифицирующие запросы отклоняются. "
    "Обслуживаемая база задаётся конфигурацией сервера."
)

server = FastMCP("text2sql_tool_server")

_agent: Text2SQLGenerator | None = None


def build_db_uri(target_db: str, db_host: str) -> str:
    """Собирает URI подключения из кред окружения."""
    return f"postgresql+psycopg://{os.environ['DB_USER']}:{os.environ['DB_PASS']}@{db_host}/{target_db}"


def build_llm_client(model_name: str) -> OpenAIChatCompletionClient:
    """Клиент к OpenAI-совместимому эндпоинту (эндпоинт МФТИ либо локальный vLLM)."""
    return OpenAIChatCompletionClient(
        model=model_name,
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        temperature=0.6,
        model_info={
            "json_output": False,
            "function_calling": True,
            "vision": False,
            "family": "unknown",
            "structured_output": False,
        },
    )


def build_agent(target_db: str, with_schema: bool, db_host: str) -> Text2SQLGenerator:
    """Готовый к работе генератор: подключение к БД поднято, системный промпт собран."""
    agent = Text2SQLGenerator(
        db_uri=build_db_uri(target_db, db_host),
        llm_client=build_llm_client(os.environ["LLM_MODEL_NAME"]),
        with_schema=with_schema,
    )
    agent.build()
    return agent


def get_agent() -> Text2SQLGenerator:
    """Агент, поднятый при старте сервера."""
    if _agent is None:
        raise RuntimeError("Агент не инициализирован: сервер запускают через main()")
    return _agent


def format_result(result: dict) -> str:
    """Ответ генератора → текст для вызывающего агента."""
    if result.get("status") == "ambiguous":
        return f"_Запрос неоднозначен_: {result.get('message') or 'уточните формулировку'}"

    if result.get("status") != "success":
        return f"_Ошибка_: {result.get('message', 'Выполнение запроса неуспешно')}"

    sql = result.get("query", "")
    sql_block = f"```sql\n{sql}\n```\n\n" if sql else ""

    execution = result.get("execution", {})
    if execution.get("status") != "success":
        return f"{sql_block}_Ошибка выполнения_: {execution.get('error', 'запрос не выполнен')}"

    rows = execution.get("results", [])
    table = tabulate(rows, headers="keys", tablefmt="pipe") if rows else "Нет данных"
    return f"{sql_block}{table}"


@server.tool(description=TOOL_DESCRIPTION)
async def text2sql(
    user_query_text: Annotated[
        str,
        Field(description="Текстовый запрос (напр.'все отчёты по западному округу за первый квартал')"),
    ],
) -> str:
    # Проверка неоднозначности и LLM-верификация выключены намеренно: первая режет EX
    # (нормальные вопросы объявляются неоднозначными), вторая удваивает вызовы LLM.
    result = await get_agent().query(user_query_text, check_ambiguity=False, check_sql_query=False)
    return format_result(result)


def main() -> None:
    global _agent

    load_dotenv()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

    target_db = os.getenv("TARGET_DB", DEFAULT_TARGET_DB)
    with_schema = os.getenv("TEXT2SQL_WITH_SCHEMA", "false").lower() in ("1", "true", "yes")
    db_host = os.getenv("BENCHMARK_DB_URL", DEFAULT_DB_HOST)

    logger.info("Инициализация инструмента (db=%s, with_schema=%s) ...", target_db, with_schema)
    _agent = build_agent(target_db, with_schema, db_host)
    logger.debug("Системная инструкция:\n\n%s", _agent.system_prompt)

    server.run(
        transport="http",
        host=os.getenv("MCP_HOST", DEFAULT_SERVER_HOST),
        port=int(os.getenv("MCP_PORT", DEFAULT_SERVER_PORT)),
        show_banner=False,
    )


if __name__ == "__main__":
    main()
