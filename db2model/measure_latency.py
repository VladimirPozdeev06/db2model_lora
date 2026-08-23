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
import statistics
import time

from adv_text2sql.mcp_servers.text2sql_tool.src.text2sql_implementation import (
    Text2SQLGenerator,
)

from llm_client import make_client
from utils import DB2MODEL_DIR, RESULTS_DIR, build_db_uri, dump_json, load_json

DB = "financial"
N = 15  # sample of eval questions


async def measure(agent, questions) -> list[float]:
    times = []
    for q in questions:
        t = time.perf_counter()
        await agent.generate_sql(q)  # LLM call + local sanitize/validate
        times.append(time.perf_counter() - t)
    return times


async def main() -> None:
    bird = load_json(DB2MODEL_DIR / "kaggle_input" / "bird_large.json")
    questions = [q["question"] for q in bird if q["db_id"] == DB][:N]
    client = make_client()
    db_uri = build_db_uri(DB)

    out = {}
    for with_schema in (True, False):
        agent = Text2SQLGenerator(db_uri=db_uri, llm_client=client, with_schema=with_schema)
        agent.build()
        ts = await measure(agent, questions)
        arm = "baseline (схема в контексте)" if with_schema else "lora (знание в весах)"
        out[arm] = {"mean_s": round(statistics.mean(ts), 2), "median_s": round(statistics.median(ts), 2), "n": len(ts)}
        print(f"{arm}: mean {out[arm]['mean_s']}s | median {out[arm]['median_s']}s (n={len(ts)})")

    baseline_median = out["baseline (схема в контексте)"]["median_s"]
    lora_median = out["lora (знание в весах)"]["median_s"]
    if lora_median:
        speedup = baseline_median / lora_median
        print(f"\nschema-less быстрее в {speedup:.2f}× по медиане ({baseline_median}s -> {lora_median}s)")
    dump_json(RESULTS_DIR / "latency.json", out)


if __name__ == "__main__":
    asyncio.run(main())
