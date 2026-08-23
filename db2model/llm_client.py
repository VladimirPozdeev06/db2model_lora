"""Клиент к OpenAI-совместимому LLM-эндпоинту (шлюз МФТИ либо локальный vLLM).

Вынесен из скриптов ветки: `generate_pairs`, `generate_reasoning`, `validate_profile`,
`measure_latency` и `mcp_smoke_test` строили его каждый по-своему, включая один и тот
же патч `enable_thinking`.

Учитель Qwen3.6-27B рассуждает по умолчанию, и тогда `content` в ответе приходит
ПУСТЫМ. Ключ `enable_thinking=false` это выключает; модели, которые его не знают,
поле игнорируют, поэтому он подставляется всегда.
"""

import os

from autogen_ext.models.openai import OpenAIChatCompletionClient

EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
MODEL_INFO = {
    "json_output": False,
    "function_calling": True,
    "vision": False,
    "family": "unknown",
    "structured_output": False,
}


def make_client(model: str | None = None, temperature: float = 0.0) -> OpenAIChatCompletionClient:
    """Клиент к эндпоинту из `.env`; `enable_thinking=false` подставляется в каждый вызов.

    Args:
        model: имя модели; по умолчанию `LLM_MODEL_NAME` из окружения.
        temperature: 0.0 для замеров, выше — для генерации разнообразных пар.
    """
    client = OpenAIChatCompletionClient(
        model=model or os.environ["LLM_MODEL_NAME"],
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        temperature=temperature,
        model_info=MODEL_INFO,
    )
    original_create = client.create

    async def create_with_thinking_off(messages, **kwargs):
        extra = dict(kwargs.get("extra_create_args") or {})
        extra.setdefault("extra_body", EXTRA_BODY)
        kwargs["extra_create_args"] = extra
        return await original_create(messages, **kwargs)

    client.create = create_with_thinking_off
    return client
