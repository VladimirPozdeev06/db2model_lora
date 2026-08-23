"""Attach a short chain-of-thought to each training pair (Think2SQL, week-4 optional).

For every (question, gold SQL) in the training set the teacher writes a concise
2–4 step reasoning that leads to the query; the KNOWN gold SQL is then appended, so
the SQL label stays correct by construction (no execution filter needed). The
target the student learns becomes:

    <reasoning>
    SQL:
    <gold sql>

Training on this format tests whether explicit reasoning helps a small SFT model
(Think2SQL). Eval must extract the SQL after the `SQL:` marker — the reasoning
notebook does this.

Run: PYTHONIOENCODING=utf-8 uv run --env-file .env python db2model/generate_reasoning.py
Needs only the LLM endpoint (no DB / tunnel). ~1096 calls.
"""

import asyncio

from autogen_core.models import SystemMessage, UserMessage

from llm_client import EXTRA_BODY, make_client
from utils import DATASET_DIR, dump_json, load_json

OUT = DATASET_DIR / "train_reasoning.json"
CONCURRENCY = 8
TEMPERATURE = 0.3

SYSTEM = (
    "You are a PostgreSQL expert writing training data. Given a question about the "
    "database `{db_id}` and the correct SQL answer, write a SHORT reasoning (2–4 "
    "numbered steps) that leads from the question to that SQL: which tables/columns "
    "are needed, how they join, what filter/aggregation applies. Do NOT restate the "
    "SQL inside the reasoning. Keep it under 60 words."
)
USER = "question: {question}\ncorrect SQL: {sql}\n\nWrite only the reasoning steps."


async def one(client, sem, pair):
    async with sem:
        sys_msg = SYSTEM.format(db_id=pair["db_id"])
        user = USER.format(question=pair["question"], sql=pair["sql"])
        try:
            r = await client.create(
                [SystemMessage(content=sys_msg), UserMessage(source="user", content=user)],
                extra_create_args={"extra_body": EXTRA_BODY},
            )
            reasoning = (r.content or "").strip()
        except Exception as exc:
            reasoning = ""
            print("  ошибка:", str(exc)[:80])
        if not reasoning:
            return None
        return {
            "db_id": pair["db_id"],
            "question": pair["question"],
            "sql": pair["sql"],
            "reasoning": reasoning,
            "difficulty": pair.get("difficulty", "unknown"),
            "task": "reasoning_sql",
        }


async def main() -> None:
    pairs = load_json(DATASET_DIR / "train.json")
    client = make_client(temperature=TEMPERATURE)
    sem = asyncio.Semaphore(CONCURRENCY)

    out = []
    done = 0
    tasks = [one(client, sem, p) for p in pairs]
    for coro in asyncio.as_completed(tasks):
        r = await coro
        done += 1
        if r:
            out.append(r)
        if done % 100 == 0:
            print(f"  {done}/{len(pairs)} (собрано {len(out)})")

    dump_json(OUT, out)
    print(f"\nreasoning-пар: {len(out)}/{len(pairs)} -> {OUT}")
    if out:
        print("пример reasoning:\n", out[0]["reasoning"][:200])


if __name__ == "__main__":
    asyncio.run(main())
