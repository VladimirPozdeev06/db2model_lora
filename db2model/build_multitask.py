"""Derive the three ROUTE auxiliary tasks — schema linking, noise correction and
continuation writing — from the already-filtered question/SQL pairs in clean/.

The point of ROUTE is that the target model learns SQL generation better when it
is also trained to (a) name the tables/columns a question touches, (b) repair a
nearly-right query, and (c) finish a query given a valid head. All three signals
can be produced deterministically from a pair whose SQL already executes, so this
step needs neither the teacher nor the live database — it only reads clean/*.json
and the profiles.

Each derived example carries its own chat turns (`system`, `user`, `target`) so
the training notebook can format it without knowing the task. The main
`sql_generation` task keeps its lean `question`/`sql` shape and is tagged in
build_dataset.py, not here.

Usage:
    uv run python db2model/build_multitask.py
    uv run python db2model/build_multitask.py toxicology financial
"""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from sqlglot import Dialects, exp, parse_one

CLEAN_DIR = Path(__file__).parent / "clean"
PROFILES = Path(__file__).parent / "profiles"
OUT_DIR = Path(__file__).parent / "dataset"
TARGET_DBS = ["financial", "toxicology", "codebase_community"]
SEED = 0

# Keep the auxiliary tasks a minority of the mix: they are a regulariser, not the
# objective. ROUTE-style multitask helps most when SQL generation still dominates.
SCHEMA_LINK_FRACTION = 0.5
NOISE_FRACTION = 0.5
CONTINUATION_FRACTION = 0.5


def _columns_by_table(db_id: str) -> dict[str, list[str]]:
    profile = json.loads((PROFILES / f"{db_id}.json").read_text(encoding="utf-8"))
    return {t: [c["name"] for c in info["columns"]]
            for t, info in profile["tables"].items()}


def _referenced(sql: str) -> tuple[list[str], list[str]]:
    """Tables and columns a query touches, via the parsed AST. Columns are returned
    bare (no alias) because the training target is 'which real columns matter', and
    the aliases in the SQL are throwaway (T1, T2)."""
    tree = parse_one(sql, dialect=Dialects.POSTGRES)
    tables = sorted({t.name for t in tree.find_all(exp.Table) if t.name})
    columns = sorted({c.name for c in tree.find_all(exp.Column) if c.name})
    return tables, columns


def schema_linking(pair: dict, db_id: str) -> dict | None:
    """question -> the tables and columns needed to answer it. Nothing is invented:
    the answer is read straight off the gold SQL that already executed."""
    try:
        tables, columns = _referenced(pair["sql"])
    except Exception:
        return None
    if not tables or not columns:
        return None
    target = ("tables: " + ", ".join(tables)
              + "\ncolumns: " + ", ".join(columns))
    system = (f"You are a PostgreSQL expert for the database `{db_id}`. "
              "Name only the tables and columns needed to answer the question, "
              "as two lines:\ntables: ...\ncolumns: ...")
    return {
        "db_id": db_id,
        "task": "schema_linking",
        "system": system,
        "user": pair["question"],
        "target": target,
        "source": "derived",
    }


def _corrupt(sql: str, cols_by_table: dict[str, list[str]], rng: random.Random) -> str | None:
    """Swap one referenced column for a different real column from the same DB. The
    result stays syntactically valid and plausible — the model's job is to notice
    it does not match the question and restore the right one. This mirrors the
    teacher's dominant mistake (a wrong-but-existing column), not random garbage."""
    try:
        tree = parse_one(sql, dialect=Dialects.POSTGRES)
    except Exception:
        return None
    all_cols = sorted({c for cols in cols_by_table.values() for c in cols})
    col_nodes = [c for c in tree.find_all(exp.Column) if c.name]
    if not col_nodes or len(all_cols) < 2:
        return None
    victim = rng.choice(col_nodes)
    alternatives = [c for c in all_cols if c != victim.name]
    if not alternatives:
        return None
    victim.set("this", exp.to_identifier(rng.choice(alternatives)))
    broken = tree.sql(dialect=Dialects.POSTGRES)
    return broken if broken.strip().lower() != sql.strip().lower() else None


def noise_correction(pair: dict, db_id: str,
                     cols_by_table: dict[str, list[str]],
                     rng: random.Random) -> dict | None:
    """(question + a nearly-right SQL) -> the corrected SQL. The correct answer is
    the gold query that already passed the execution filter, so the label is clean
    by construction."""
    broken = _corrupt(pair["sql"], cols_by_table, rng)
    if not broken:
        return None
    system = (f"You are a PostgreSQL expert for the database `{db_id}`. "
              "The SQL below is close but wrong. Return only the corrected SQL.")
    user = f"question: {pair['question']}\nSQL: {broken}"
    return {
        "db_id": db_id,
        "task": "noise_correction",
        "system": system,
        "user": user,
        "target": pair["sql"],
        "source": "derived",
    }


_CLAUSE = re.compile(
    r"\s+(WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)\s+", re.IGNORECASE)


def continuation_writing(pair: dict, db_id: str) -> dict | None:
    """(question + the start of the SQL) -> the rest of it. Cut at a clause keyword
    so the prefix is a real query stem (SELECT ... FROM ...) and the completion is
    a meaningful tail — this trains the model to finish a query given a valid head,
    the third ROUTE auxiliary task. Queries with no cuttable clause are skipped."""
    sql = pair["sql"]
    cuts = [m.start() for m in _CLAUSE.finditer(sql)]
    if not cuts:
        return None
    cut = cuts[len(cuts) // 2]  # a stable middle boundary, not random
    prefix, suffix = sql[:cut].strip(), sql[cut:].strip()
    if not prefix or not suffix:
        return None
    system = (f"You are a PostgreSQL expert for the database `{db_id}`. "
              "Complete the SQL query. Return only the missing tail, nothing else.")
    user = f"question: {pair['question']}\nSQL so far: {prefix}"
    return {
        "db_id": db_id,
        "task": "continuation_writing",
        "system": system,
        "user": user,
        "target": suffix,
        "source": "derived",
    }


def derive(records: list[dict], seed: int = SEED) -> list[dict]:
    """The reusable entry point: given question/SQL records (each with db_id),
    return the auxiliary examples. build_dataset.py calls this on the *train* split
    only, so no validation question can leak into training through an aux task."""
    rng = random.Random(seed)
    out: list[dict] = []
    by_db: dict[str, list[dict]] = {}
    for r in records:
        by_db.setdefault(r["db_id"], []).append(r)

    for db_id, pairs in by_db.items():
        try:
            cols_by_table = _columns_by_table(db_id)
        except FileNotFoundError:
            print(f"  пропускаю {db_id}: нет профиля")
            continue

        n_link = round(len(pairs) * SCHEMA_LINK_FRACTION)
        n_noise = round(len(pairs) * NOISE_FRACTION)
        n_cont = round(len(pairs) * CONTINUATION_FRACTION)
        for pair in rng.sample(pairs, min(n_link, len(pairs))):
            ex = schema_linking(pair, db_id)
            if ex:
                out.append(ex)
        for pair in rng.sample(pairs, min(n_noise, len(pairs))):
            ex = noise_correction(pair, db_id, cols_by_table, rng)
            if ex:
                out.append(ex)
        for pair in rng.sample(pairs, min(n_cont, len(pairs))):
            ex = continuation_writing(pair, db_id)
            if ex:
                out.append(ex)
    return out


def _from_clean(db_ids: list[str]) -> list[dict]:
    records: list[dict] = []
    for db_id in db_ids:
        path = CLEAN_DIR / f"{db_id}.json"
        if not path.exists():
            print(f"  пропускаю {db_id}: нет {path}")
            continue
        for p in json.loads(path.read_text(encoding="utf-8")):
            records.append({"db_id": db_id, "question": p["question"], "sql": p["sql"]})
    return records


def main() -> None:
    """Standalone run derives from all of clean/ — for inspecting the aux tasks.
    The real pipeline calls derive() on the train split inside build_dataset.py."""
    db_ids = sys.argv[1:] or TARGET_DBS
    examples = derive(_from_clean(db_ids))
    if not examples:
        print("нечего собирать — сначала generate_pairs.py и filter_pairs.py")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "multitask.json"
    out.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"мультизадачных примеров: {len(examples)}")
    print("  по задаче:", dict(Counter(e["task"] for e in examples)))
    print("  по базе:  ", dict(Counter(e["db_id"] for e in examples)))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
