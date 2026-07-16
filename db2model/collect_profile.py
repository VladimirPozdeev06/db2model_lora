"""Collect a compact profile of each target database: schema, foreign keys,
statistics and real values. The profile is what the teacher model gets in its
prompt, so it has to be complete enough to write a correct query from, and small
enough to fit in a context window.

Usage:
    uv run --env-file .env python db2model/collect_profile.py
    uv run --env-file .env python db2model/collect_profile.py financial toxicology

Needs the ssh tunnel to the benchmark database to be up.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import tiktoken
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

TARGET_DBS = ["financial", "toxicology", "codebase_community"]

OUT_DIR = Path(__file__).parent / "profiles"
SAMPLE_ROWS = 5
MAX_VALUE_LEN = 120
"""Long free text (forum posts, comments) would blow up the profile."""
LOW_CARDINALITY_MAX = 15
"""A column with few distinct values gets all of them listed, not just samples."""


def _truncate(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_VALUE_LEN:
        return value[:MAX_VALUE_LEN] + "..."
    return value


def _engine(db_id: str) -> Engine:
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASS"]
    host = os.getenv("BENCHMARK_DB_URL", "localhost:5444")
    return create_engine(
        f"postgresql+psycopg://{user}:{password}@{host}/{db_id}", pool_pre_ping=True
    )


def _row_estimate(conn, table: str) -> int:
    """reltuples is an estimate, but COUNT(*) on every table is too slow."""
    n = conn.execute(
        text("SELECT reltuples::bigint FROM pg_class WHERE relname = :t"),
        {"t": table},
    ).scalar()
    return max(int(n or 0), 0)


def _table_comment(conn, table: str) -> str | None:
    return conn.execute(
        text("SELECT obj_description(CAST(:t AS regclass), 'pg_class')"), {"t": table}
    ).scalar()


def _column_stats(conn, table: str) -> dict[str, dict]:
    rows = conn.execute(
        text(
            """
            SELECT attname, n_distinct, null_frac,
                   most_common_vals::text AS most_common_vals,
                   histogram_bounds::text AS histogram_bounds
            FROM pg_stats
            WHERE schemaname = 'public' AND tablename = :t
            """
        ),
        {"t": table},
    ).mappings()
    return {r["attname"]: dict(r) for r in rows}


def _column_values(conn, table: str, column: str) -> dict[str, Any]:
    """Real literals matter: without them the teacher invents values that are
    not in the database and every generated query dies on the execution filter."""
    try:
        distinct = conn.execute(
            text(
                f'SELECT DISTINCT "{column}" FROM "{table}" '
                f'WHERE "{column}" IS NOT NULL LIMIT :lim'
            ),
            {"lim": LOW_CARDINALITY_MAX + 1},
        ).fetchall()
    except Exception as exc:
        # A failed statement aborts the transaction, and every later query on this
        # connection would fail with "current transaction is aborted" — silently
        # producing a profile full of errors instead of one bad column.
        conn.rollback()
        return {"error": str(exc)[:80]}

    values = [_truncate(r[0]) for r in distinct]
    if len(values) <= LOW_CARDINALITY_MAX:
        return {"all_values": values}
    return {"sample_values": values[:SAMPLE_ROWS]}


def collect(db_id: str) -> dict:
    engine = _engine(db_id)
    inspector = inspect(engine)
    profile: dict[str, Any] = {"db_id": db_id, "tables": {}, "foreign_keys": []}

    with engine.connect() as conn:
        for table in inspector.get_table_names(schema="public"):
            stats = _column_stats(conn, table)
            pk = inspector.get_pk_constraint(table, schema="public")
            columns = []

            for col in inspector.get_columns(table, schema="public"):
                name = col["name"]
                st = stats.get(name, {})
                entry: dict[str, Any] = {
                    "name": name,
                    "type": str(col["type"]),
                    "nullable": col["nullable"],
                }
                if st.get("n_distinct") is not None:
                    entry["n_distinct"] = st["n_distinct"]
                if st.get("null_frac"):
                    entry["null_frac"] = round(float(st["null_frac"]), 3)
                if st.get("most_common_vals"):
                    entry["most_common_vals"] = _truncate(st["most_common_vals"])
                if st.get("histogram_bounds"):
                    entry["range"] = _truncate(st["histogram_bounds"])
                entry.update(_column_values(conn, table, name))
                columns.append(entry)

            profile["tables"][table] = {
                "columns": columns,
                "primary_key": pk.get("constrained_columns", []),
                "row_estimate": _row_estimate(conn, table),
                "comment": _table_comment(conn, table),
                "sample_rows": _sample_rows(conn, table),
            }

            for fk in inspector.get_foreign_keys(table, schema="public"):
                profile["foreign_keys"].append(
                    {
                        "from_table": table,
                        "from_columns": fk["constrained_columns"],
                        "to_table": fk["referred_table"],
                        "to_columns": fk["referred_columns"],
                    }
                )

    return profile


def _sample_rows(conn, table: str) -> list[dict]:
    try:
        result = conn.execute(text(f'SELECT * FROM "{table}" LIMIT :n'), {"n": SAMPLE_ROWS})
        cols = list(result.keys())
        return [{c: _truncate(v) for c, v in zip(cols, row)} for row in result.fetchall()]
    except Exception as exc:
        conn.rollback()
        return [{"error": str(exc)[:80]}]


def main() -> None:
    dbs = sys.argv[1:] or TARGET_DBS
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding("cl100k_base")

    for db_id in dbs:
        profile = collect(db_id)
        out = OUT_DIR / f"{db_id}.json"
        payload = json.dumps(profile, ensure_ascii=False, indent=2, default=str)
        out.write_text(payload, encoding="utf-8")

        n_fk = len(profile["foreign_keys"])
        print(
            f"{db_id:<24} таблиц: {len(profile['tables']):>2}  FK: {n_fk:>2}  "
            f"токенов: {len(enc.encode(payload)):>6}  -> {out}"
        )


if __name__ == "__main__":
    main()
