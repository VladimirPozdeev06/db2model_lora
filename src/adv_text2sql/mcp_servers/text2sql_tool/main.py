import asyncio
import os
from typing import Annotated
import logging

from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import Field
from .src.generate_tool_description import generate_description_text2sql
from .src.text2sql_implementation import Text2SQLGenerator
from tabulate import tabulate
# from unilog import setup_logging

load_dotenv()

logger = logging.getLogger(__name__)

server = FastMCP("text2sql_tool_server")

llm_client = OpenAIChatCompletionClient(
    model=os.environ["LLM_MODEL_NAME"],
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

# Which database this server serves, and how knowledge about it is supplied.
# WITH_SCHEMA=false is the winning LoRA arm: schema comes from the adapter weights,
# not the prompt (set LLM_MODEL_NAME to the served adapter, e.g. `db2model`).
# WITH_SCHEMA=true is the baseline / RAG arm: schema is injected into the prompt.
TARGET_DB = os.getenv("TARGET_DB", "financial")
WITH_SCHEMA = os.getenv("TEXT2SQL_WITH_SCHEMA", "false").lower() in ("1", "true", "yes")
DB_HOST = os.getenv("BENCHMARK_DB_URL", "localhost:5444")
db_uri = (
    f"postgresql+psycopg://{os.environ['DB_USER']}:{os.environ['DB_PASS']}"
    f"@{DB_HOST}/{TARGET_DB}"
)

logger.info("Инициализация инструмента (db=%s, with_schema=%s) ...", TARGET_DB, WITH_SCHEMA)
text2sql_agent = Text2SQLGenerator(
    db_uri=db_uri,
    llm_client=llm_client,
    with_schema=WITH_SCHEMA,
)
text2sql_agent.build()

logger.info("Генерация описания ...")

generated_description = None
if not generated_description:
    generated_description = asyncio.run(
        generate_description_text2sql(
            tool_description=(
                "Генератор SQL-запросов на естественном языке с контролем безопасности."
            ),
            text2sql_agent=text2sql_agent,
        )
    )

logger.info("Генерация описания завершена")
logger.info("Инициализация инструмента завершена")

logger.debug(f"Системная инструкция:\n\n{text2sql_agent.system_prompt}")


@server.tool(description=generated_description)
async def text2sql(
    user_query_text: Annotated[
        str,
        Field(
            description=(
                "Текстовый запрос (напр.'все отчёты по западному округу за первый квартал')"
            )
        ),
    ],
) -> str | None:
    global text2sql_agent

    result = await text2sql_agent.query(
        user_query_text,
        check_ambiguity=False,
        check_sql_query=False,
    )

    if result.get("status") == "ambiguous":
        return f"_Запрос неоднозначен_: {result.get('message') or 'уточните формулировку'}"

    if result.get("status") == "success":
        exec_info = result.get("execution", {})
        if exec_info.get("status") == "success":
            data = exec_info.get("results", [])
            return (
                tabulate(data, headers="keys", tablefmt="pipe") if data else "Нет данных"
            )
        return f"_Ошибка выполнения_: {exec_info.get('error', 'запрос не выполнен')}"

    return f"_Ошибка_: {result.get('message', 'Выполнение запроса неуспешно')}"


logger.info("Tool description: %s", text2sql.description)

if __name__ == "__main__":
    server.run(transport="http", host="0.0.0.0", port=8000, show_banner=False)
