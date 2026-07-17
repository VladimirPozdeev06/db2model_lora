"""Ask the teacher model for question/SQL pairs about one database, using the
profile collected by collect_profile.py as the only description of it.

Output is raw and unverified — every pair still has to survive filter_pairs.py.

Usage:
    uv run --env-file .env python db2model/generate_pairs.py toxicology 40
"""

import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

from autogen_core.models import SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from sqlglot import transpile

PROFILES = Path(__file__).parent / "profiles"
OUT_DIR = Path(__file__).parent / "raw"
PAIRS_PER_CALL = 8
N_FEWSHOT = 3
TEMPERATURE = 0.8
"""Diversity matters more than precision here: bad pairs get filtered out later,
but pairs that are all variations of one query cannot be recovered."""

FLAVOURS = [
    "simple lookups with a WHERE filter",
    "aggregations (COUNT, AVG, SUM, MAX) with GROUP BY",
    "joins across two or three tables following the foreign keys",
    "ranking questions using ORDER BY with LIMIT",
    "questions combining a join with an aggregate and a filter",
    "questions about ratios or percentages between two quantities",
]

SYSTEM = """You are a PostgreSQL expert building a training set for a smaller model.

Below is the complete profile of the database `{db_id}`: its tables, columns,
foreign keys, statistics and real values taken from the live database.

{profile}

Rules:
- Every table, column and literal you use MUST appear in the profile above.
  Inventing a value that is not in the data makes the pair useless.
- Follow the foreign keys in the profile when joining. Do not join on columns
  that are not linked there.
- Questions must sound like a human analyst asking, not like a description of SQL.
- The SQL must be valid PostgreSQL and must return at least one row.
- Every subquery in FROM must have an alias. PostgreSQL rejects it otherwise.
"""

USER = """Write {n} DIFFERENT question/SQL pairs about `{db_id}`.

Focus this batch on: {flavour}.
Prefer these tables where it makes sense: {tables}.

{fewshot}
Answer with a JSON array only, no prose:
[{{"question": "...", "sql": "...", "difficulty": "simple|moderate|challenging"}}]
"""


def _fewshot(db_id: str) -> str:
    """Real BIRD train pairs anchor the style. They are disjoint from bird_large
    and bird_small, so this cannot leak the evaluation set.

    They are stored in SQLite dialect and the teacher copies whatever it sees, so
    they have to be transpiled first — otherwise it emits SQLite-isms such as an
    unaliased subquery in FROM, which PostgreSQL rejects."""
    pairs = [i for i in json.load(open("data/train_queries.json")) if i["db_id"] == db_id]
    if not pairs:
        return ""
    picked = random.sample(pairs, min(N_FEWSHOT, len(pairs)))
    lines = []
    for p in picked:
        try:
            sql = transpile(p["SQL"], read="sqlite", write="postgres")[0]
        except Exception:
            continue
        lines.append(f'  Q: {p["question"]}\n  SQL: {sql}')
    if not lines:
        return ""
    return "Examples of the style expected:\n" + "\n".join(lines) + "\n"


def _parse(text: str) -> list[dict]:
    """Decode the first JSON array in the reply. A greedy regex would run from the
    first '[' to the last ']' anywhere in the text, swallowing prose or a second
    array; raw_decode stops at the end of the first well-formed value instead."""
    decoder = json.JSONDecoder()
    for start in (i for i, ch in enumerate(text) if ch == "["):
        try:
            items, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(items, list):
            return [
                it for it in items
                if isinstance(it, dict) and it.get("question") and it.get("sql")
            ]
    return []


async def generate(db_id: str, target: int) -> list[dict]:
    profile = (PROFILES / f"{db_id}.json").read_text(encoding="utf-8")
    tables = list(json.loads(profile)["tables"])

    client = OpenAIChatCompletionClient(
        model=os.environ["LLM_MODEL_NAME"],
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        temperature=TEMPERATURE,
        model_info={"json_output": False, "function_calling": False, "vision": False,
                    "family": "unknown", "structured_output": False},
    )
    system = SYSTEM.format(db_id=db_id, profile=profile)

    pairs: list[dict] = []
    n_calls = -(-target // PAIRS_PER_CALL)
    for i in range(n_calls):
        flavour = FLAVOURS[i % len(FLAVOURS)]
        focus = random.sample(tables, min(3, len(tables)))
        user = USER.format(
            n=PAIRS_PER_CALL, db_id=db_id, flavour=flavour,
            tables=", ".join(focus), fewshot=_fewshot(db_id),
        )
        result = await client.create(
            [SystemMessage(content=system), UserMessage(source="user", content=user)]
        )
        batch = _parse(result.content)
        for item in batch:
            item["db_id"] = db_id
            item["flavour"] = flavour
        pairs.extend(batch)
        print(f"  вызов {i + 1}/{n_calls}: получено {len(batch)} пар (всего {len(pairs)})")

    return pairs


async def main() -> None:
    db_id = sys.argv[1] if len(sys.argv) > 1 else "toxicology"
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    random.seed(0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = await generate(db_id, target)

    out = OUT_DIR / f"{db_id}.json"
    out.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{db_id}: {len(pairs)} сырых пар -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
