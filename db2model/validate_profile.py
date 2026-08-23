"""Check that a profile is good enough to write queries from.

The teacher sees the profile and nothing else — no database access — and has to
answer real BIRD train questions. Every answer is then executed against the live
database. If queries fail because a table, column or value is missing from the
profile, the profile is incomplete; if they fail on reasoning, it is not.

train_queries.json is disjoint from bird_large/bird_small, so nothing here leaks
into the evaluation set.

Usage:
    uv run --env-file .env python db2model/validate_profile.py toxicology 6
"""

import asyncio
import random
import re
import sys

from autogen_core.models import SystemMessage, UserMessage
from sqlalchemy import text

from llm_client import make_client
from utils import DATA_DIR, PROFILES_DIR, load_json, make_engine

TRAIN_QUERIES = DATA_DIR / "train_queries.json"

SYSTEM = """You are a PostgreSQL expert. Below is a complete profile of a database:
schema, foreign keys, statistics and real values.

{profile}

Write ONE PostgreSQL query answering the question. Return only SQL, no prose."""


def _clean(sql: str) -> str:
    return re.sub(r"^```(?:sql)?|```$", "", sql.strip(), flags=re.IGNORECASE).strip()


async def main() -> None:
    db_id = sys.argv[1] if len(sys.argv) > 1 else "toxicology"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    random.seed(0)

    profile = (PROFILES_DIR / f"{db_id}.json").read_text(encoding="utf-8")
    pairs = [i for i in load_json(TRAIN_QUERIES) if i["db_id"] == db_id]
    sample = random.sample(pairs, min(n, len(pairs)))

    client = make_client()

    engine = make_engine(db_id)

    system = SYSTEM.format(profile=profile)
    ok = 0
    with engine.connect() as conn:
        for item in sample:
            question = f"question: {item['question']}, evidence: {item['evidence']}"
            result = await client.create([SystemMessage(content=system), UserMessage(source="user", content=question)])
            sql = _clean(result.content)
            try:
                rows = conn.execute(text(sql)).fetchall()
                ok += 1
                print(f"[исполнился, строк {len(rows):>5}] {item['question'][:60]}")
            except Exception as exc:
                conn.rollback()
                print(f"[ошибка] {item['question'][:60]}")
                print(f"         {str(exc).splitlines()[0][:95]}")

    print(f"\n{db_id}: исполнилось {ok}/{len(sample)}")


if __name__ == "__main__":
    asyncio.run(main())
