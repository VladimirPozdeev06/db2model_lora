"""Measure per-question generation latency of the two arms through the real MCP
path. The schema-less (knowledge-in-weights) arm sends ~90 prompt tokens vs ~288
for the schema-in-context baseline, so its prefill — and thus latency — is shorter.

This is the latency axis of the week-5 multi-criteria selection. Measured against
the endpoint's base Qwen3.6-27B (not the served 7B adapter) and includes network,
so the absolute numbers are indicative; the RELATIVE gap reflects the prompt-length
effect, which holds regardless of the served model.

Run: PYTHONIOENCODING=utf-8 uv run --env-file .env python db2model/measure_latency.py
Needs the ssh tunnel (schema build) and the LLM endpoint.
"""
import asyncio
import json
import os
import statistics
import time

from autogen_ext.models.openai import OpenAIChatCompletionClient

from adv_text2sql.mcp_servers.text2sql_tool.src.text2sql_implementation import (
    Text2SQLGenerator,
)

DB = "financial"
N = 15  # sample of eval questions
db_uri = (f"postgresql+psycopg://{os.environ['DB_USER']}:{os.environ['DB_PASS']}"
          f"@{os.getenv('BENCHMARK_DB_URL', 'localhost:5444')}/{DB}")


def make_client():
    c = OpenAIChatCompletionClient(
        model=os.environ["LLM_MODEL_NAME"], base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"], temperature=0.0,
        model_info={"json_output": False, "function_calling": True, "vision": False,
                    "family": "unknown", "structured_output": False})
    _orig = c.create

    async def _create(messages, **kw):
        ec = dict(kw.get("extra_create_args") or {})
        ec.setdefault("extra_body", {"chat_template_kwargs": {"enable_thinking": False}})
        kw["extra_create_args"] = ec
        return await _orig(messages, **kw)

    c.create = _create
    return c


async def measure(agent, questions) -> list[float]:
    times = []
    for q in questions:
        t = time.perf_counter()
        await agent.generate_sql(q)          # LLM call + local sanitize/validate
        times.append(time.perf_counter() - t)
    return times


async def main():
    bird = json.loads((__import__("pathlib").Path(__file__).parent
                       / "kaggle_input" / "bird_large.json").read_text(encoding="utf-8"))
    questions = [q["question"] for q in bird if q["db_id"] == DB][:N]
    client = make_client()

    out = {}
    for with_schema in (True, False):
        agent = Text2SQLGenerator(db_uri=db_uri, llm_client=client, with_schema=with_schema)
        agent.build()
        ts = await measure(agent, questions)
        arm = "baseline (схема в контексте)" if with_schema else "lora (знание в весах)"
        out[arm] = {"mean_s": round(statistics.mean(ts), 2),
                    "median_s": round(statistics.median(ts), 2),
                    "n": len(ts)}
        print(f"{arm}: mean {out[arm]['mean_s']}s | median {out[arm]['median_s']}s (n={len(ts)})")

    b = out["baseline (схема в контексте)"]["median_s"]
    l = out["lora (знание в весах)"]["median_s"]
    if l:
        print(f"\nschema-less быстрее в {b / l:.2f}× по медиане ({b}s -> {l}s)")
    (__import__("pathlib").Path(__file__).parent / "results" / "latency.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
