"""Collect the query-side half of the exploration signal: which tables real
queries actually touch, how the planner joins them, and what the server's own
access counters say. This closes the "logs and plans" bullet of the task card
without needing any server privileges we do not have.

Three sources, in order of how much we control them:

1. `pg_stat_user_tables` — the server's cumulative access counters (seq_scan,
   idx_scan, rows fetched). Readable by any user, no extension needed.
2. `EXPLAIN (FORMAT JSON)` over our own query corpus (real BIRD train pairs +
   the gold SQL of the evaluation split). EXPLAIN without ANALYZE does not run
   the query, so this is cheap and side-effect free. Gives join operators,
   scan methods and planner row estimates.
3. Static join keys from the same SQL via sqlglot — which columns the queries
   join on, independent of what the planner chose.

`pg_stat_statements` (real user query log) is probed and reported, but it needs
`shared_preload_libraries` set by the DBA, so treat a negative answer as final.

Usage:
    uv run --env-file .env python db2model/collect_query_stats.py
    uv run --env-file .env python db2model/collect_query_stats.py toxicology

Needs the ssh tunnel to the benchmark database to be up.
"""

import json
import sys
from collections import Counter
from typing import Any

import sqlglot
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlglot import expressions as exp

from utils import BIRD_EVAL_FILE, DATA_DIR, PROFILES_DIR, REPO_ROOT, TARGET_DBS, dump_json, load_json, make_engine

sys.path.insert(0, str(REPO_ROOT))
from benchmarks.evaluate_bird import sqlite_to_postgres  # noqa: E402

OUT_DIR = PROFILES_DIR
CORPUS_FILES = [
    DATA_DIR / "train_queries.json",
    BIRD_EVAL_FILE,
]
STATEMENT_TIMEOUT_MS = 15000

JOIN_NODES = {"Nested Loop", "Hash Join", "Merge Join"}


def _engine(db_id: str) -> Engine:
    """One engine per database — creating it inside a loop opens a TCP connection
    per query and kills the ssh forwarding."""
    return make_engine(db_id, statement_timeout_ms=STATEMENT_TIMEOUT_MS)


# --------------------------------------------------------------------------- #
# 1. server-side counters
# --------------------------------------------------------------------------- #


def table_access_counters(conn) -> list[dict]:
    rows = conn.execute(
        text(
            """
            SELECT relname,
                   seq_scan, seq_tup_read,
                   idx_scan, idx_tup_fetch,
                   n_live_tup
            FROM pg_stat_user_tables
            ORDER BY COALESCE(seq_scan, 0) + COALESCE(idx_scan, 0) DESC
            """
        )
    ).mappings()
    return [dict(r) for r in rows]


def stats_reset_at(conn) -> str | None:
    value = conn.execute(text("SELECT stats_reset FROM pg_stat_database WHERE datname = current_database()")).scalar()
    return str(value) if value else None


def pg_stat_statements_status(conn) -> dict[str, Any]:
    """The real user-query log. Requires the extension to be preloaded by the DBA,
    and a non-superuser sees other users' query text as <insufficient privilege>."""
    status: dict[str, Any] = {}
    for key, sql in (
        ("shared_preload_libraries", "SHOW shared_preload_libraries"),
        (
            "available",
            "SELECT installed_version FROM pg_available_extensions WHERE name = 'pg_stat_statements'",
        ),
        ("is_superuser", "SELECT current_setting('is_superuser')"),
        (
            "can_read_all_stats",
            "SELECT pg_has_role(current_user, 'pg_read_all_stats', 'member')",
        ),
    ):
        try:
            status[key] = conn.execute(text(sql)).scalar()
        except Exception as exc:
            conn.rollback()
            status[key] = f"error: {str(exc)[:80]}"

    try:
        status["rows_visible"] = conn.execute(text("SELECT count(*) FROM pg_stat_statements")).scalar()
    except Exception as exc:
        conn.rollback()
        status["rows_visible"] = f"error: {str(exc)[:80]}"

    return status


# --------------------------------------------------------------------------- #
# 2. plans
# --------------------------------------------------------------------------- #


def _leaf_relations(node: dict) -> list[str]:
    found = []
    if node.get("Relation Name"):
        found.append(node["Relation Name"])
    for child in node.get("Plans", []):
        found.extend(_leaf_relations(child))
    return found


def _walk_plan(node: dict, acc: dict) -> None:
    node_type = node.get("Node Type", "")
    relation = node.get("Relation Name")

    if relation:
        acc["tables"][relation] += 1
        acc["scans"][f"{relation}:{node_type}"] += 1

    if node_type in JOIN_NODES:
        children = node.get("Plans", [])
        if len(children) >= 2:
            left = set(_leaf_relations(children[0]))
            right = set(_leaf_relations(children[1]))
            for a in left:
                for b in right:
                    if a != b:
                        acc["plan_joins"]["|".join(sorted((a, b)))] += 1
            acc["join_operators"][node_type] += 1

    for child in node.get("Plans", []):
        _walk_plan(child, acc)


def explain(conn, sql: str) -> dict | None:
    """EXPLAIN without ANALYZE only plans the query, it does not execute it.

    Goes through the raw cursor on purpose: SQLAlchemy hands psycopg an empty
    parameter set, which turns on client-side placeholder parsing and makes any
    query containing a literal `%` (LIKE patterns, date formats) fail before it
    ever reaches the server.
    """
    with conn.connection.driver_connection.cursor() as cur:
        cur.execute("EXPLAIN (FORMAT JSON) " + sql)
        row = cur.fetchone()
    raw = row[0] if row else None
    payload = json.loads(raw) if isinstance(raw, str) else raw
    if not payload:
        return None
    return payload[0].get("Plan")


# --------------------------------------------------------------------------- #
# 3. static join keys
# --------------------------------------------------------------------------- #


def static_join_keys(sql: str) -> list[str]:
    """What the query itself says it joins on, regardless of the plan."""
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return []

    alias_to_table: dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        name = table.name
        alias_to_table[name] = name
        if table.alias:
            alias_to_table[table.alias] = name

    keys = []
    for eq in tree.find_all(exp.EQ):
        left, right = eq.this, eq.expression
        if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
            continue
        lt, rt = alias_to_table.get(left.table), alias_to_table.get(right.table)
        if not lt or not rt or lt == rt:
            continue
        pair = sorted([f"{lt}.{left.name}", f"{rt}.{right.name}"])
        keys.append(" = ".join(pair))
    return keys


# --------------------------------------------------------------------------- #


def load_corpus(db_id: str) -> list[dict]:
    corpus = []
    for path in CORPUS_FILES:
        if not path.exists():
            continue
        for item in load_json(path):
            if item.get("db_id") == db_id and item.get("SQL"):
                corpus.append({"source": path.name, "sql": item["SQL"]})
    return corpus


def collect(db_id: str) -> dict:
    corpus = load_corpus(db_id)
    acc = {
        "tables": Counter(),
        "scans": Counter(),
        "plan_joins": Counter(),
        "join_operators": Counter(),
        "static_joins": Counter(),
    }
    failures: list[dict] = []
    planned = 0

    engine = _engine(db_id)
    with engine.connect() as conn:
        conn.exec_driver_sql(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
        counters = table_access_counters(conn)
        reset_at = stats_reset_at(conn)
        pgss = pg_stat_statements_status(conn)

        for item in corpus:
            sql = sqlite_to_postgres(item["sql"])
            for key in static_join_keys(sql):
                acc["static_joins"][key] += 1
            try:
                plan = explain(conn, sql)
            except Exception as exc:
                conn.rollback()
                failures.append({"source": item["source"], "error": str(exc)[:120]})
                continue
            if plan:
                _walk_plan(plan, acc)
                planned += 1

    engine.dispose()

    return {
        "db_id": db_id,
        "corpus": {
            "queries": len(corpus),
            "planned": planned,
            "failed": len(failures),
            "files": [p.name for p in CORPUS_FILES if p.exists()],
        },
        "table_frequency": dict(acc["tables"].most_common()),
        "scan_methods": dict(acc["scans"].most_common()),
        "join_paths": dict(acc["plan_joins"].most_common()),
        "join_operators": dict(acc["join_operators"].most_common()),
        "join_keys": dict(acc["static_joins"].most_common()),
        "server_counters": {
            "stats_reset": reset_at,
            "pg_stat_user_tables": counters,
        },
        "pg_stat_statements": pgss,
        "failures": failures[:20],
    }


def main() -> None:
    dbs = sys.argv[1:] or TARGET_DBS
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for db_id in dbs:
        stats = collect(db_id)
        out = OUT_DIR / f"{db_id}.plans.json"
        dump_json(out, stats)

        c = stats["corpus"]
        top_table = next(iter(stats["table_frequency"].items()), ("-", 0))
        top_join = next(iter(stats["join_paths"].items()), ("-", 0))
        print(
            f"{db_id:<22} запросов: {c['planned']:>3}/{c['queries']:<3} "
            f"(ошибок {c['failed']})  таблиц: {len(stats['table_frequency']):>2}  "
            f"join-рёбер: {len(stats['join_paths']):>2}  -> {out}"
        )
        print(f"{'':22} чаще всего: {top_table[0]} ({top_table[1]}), связка {top_join[0]} ({top_join[1]})")


if __name__ == "__main__":
    main()
