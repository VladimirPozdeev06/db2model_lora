"""Execute every generated SQL query against the live database and keep only the
pairs that survive. This is the noise correction step: without it the target
model trains on queries that do not run.

A pair is dropped when the SQL is invalid, fails to execute, returns nothing, or
duplicates another pair. The reject rate is itself a result — it measures how
good the teacher is — so it gets reported, not hidden.

Usage:
    uv run --env-file .env python db2model/filter_pairs.py toxicology
"""

import re
import sys
from collections import Counter

from sqlalchemy import text
from sqlglot import Dialects, exp, parse_one

from utils import CLEAN_DIR, DB2MODEL_DIR, dump_json, load_json, make_engine

RAW_DIR = DB2MODEL_DIR / "raw"
OUT_DIR = CLEAN_DIR
STATEMENT_TIMEOUT_MS = 15000
"""A generated query can be accidentally quadratic; do not hang the whole run."""


def _normalize(sql: str) -> str:
    sql = re.sub(r"^```(?:sql)?|```$", "", sql.strip(), flags=re.IGNORECASE)
    return sql.strip().rstrip(";").strip()


def _is_trivial(sql: str) -> bool:
    """SELECT * FROM t with no filter teaches the model nothing."""
    try:
        parsed = parse_one(sql, dialect=Dialects.POSTGRES)
    except Exception:
        return False
    has_star = any(isinstance(n, exp.Star) for n in parsed.walk())
    has_logic = any(isinstance(n, (exp.Where, exp.Group, exp.Join, exp.Order, exp.AggFunc)) for n in parsed.walk())
    return has_star and not has_logic


def _fingerprint(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.lower()).strip()


def filter_db(db_id: str) -> tuple[list[dict], list[dict], Counter]:
    pairs = load_json(RAW_DIR / f"{db_id}.json")
    engine = make_engine(db_id, statement_timeout_ms=STATEMENT_TIMEOUT_MS)

    kept: list[dict] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    reasons: Counter = Counter()

    def drop(pair: dict, reason: str) -> None:
        reasons[reason] += 1
        rejected.append(pair | {"reject_reason": reason})

    with engine.connect() as conn:
        for pair in pairs:
            sql = _normalize(pair["sql"])

            try:
                parse_one(sql, dialect=Dialects.POSTGRES)
            except Exception:
                drop(pair, "не парсится")
                continue

            if _is_trivial(sql):
                drop(pair, "тривиальный")
                continue

            fp = _fingerprint(sql)
            if fp in seen:
                drop(pair, "дубликат")
                continue

            try:
                rows = conn.execute(text(sql)).fetchall()
            except Exception as exc:
                drop(pair | {"error": str(exc).splitlines()[0][:150]}, "не исполняется")
                continue
            finally:
                conn.rollback()

            if not rows:
                drop(pair, "пустой результат")
                continue

            seen.add(fp)
            pair["sql"] = sql
            pair["row_count"] = len(rows)
            kept.append(pair)

    return kept, rejected, reasons


def main() -> None:
    db_id = sys.argv[1] if len(sys.argv) > 1 else "toxicology"
    kept, rejected, reasons = filter_db(db_id)
    total = len(kept)
    raw_total = total + sum(reasons.values())
    if not raw_total:
        print(f"{db_id}: в {RAW_DIR / f'{db_id}.json'} нет пар — сначала generate_pairs.py")
        return

    out = OUT_DIR / f"{db_id}.json"
    dump_json(out, kept)
    rej = OUT_DIR / f"{db_id}.rejected.json"
    dump_json(rej, rejected)

    print(
        f"{db_id}: оставлено {total}/{raw_total} "
        f"({100 * total / raw_total:.0f}%), брак {100 - 100 * total / raw_total:.0f}%"
    )
    for reason, n in reasons.most_common():
        print(f"    {reason:<20} {n:>3}")
    print(f"  по сложности: {dict(Counter(p.get('difficulty') for p in kept))}")
    print(f"  -> {out}")
    print(f"  -> {rej} (брак для разбора)")


if __name__ == "__main__":
    main()
