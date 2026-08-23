"""Train->eval leakage audit for the DB2Model SFT/LoRA branch.

Why this exists: the synthetic generator is seeded with real example values from
each database, so it can emit a training question that maps to the *same gold SQL
with the same literal constants* as an evaluation question. That is leakage — the
target model is then trained to produce the eval answer verbatim, inflating EX.

An earlier check only looked for >90% shared word 4-grams and reported "clean".
That threshold is far too strict for paraphrases and misses literal-level leaks.
This script instead compares the *normalized gold SQL, literals kept*: if a train
pair and an eval question share the same answer-query (modulo DISTINCT, aliases,
whitespace) on the same database, they return the same rows -> leak.

Usage:
    uv run python db2model/check_leakage.py                 # report
    uv run python db2model/check_leakage.py --write-clean   # also emit train_clean.json

Exit code is non-zero when leakage is found, so it can gate a pipeline.
"""

import argparse
import re
from collections import defaultdict

from utils import BIRD_EVAL_FILE, DATASET_DIR, TARGET_DBS, dump_json, load_json

TRAIN_FILE = DATASET_DIR / "train.json"
CLEAN_OUT = DATASET_DIR / "train_clean.json"


def norm_sql(s: str) -> str:
    """Normalize SQL for equality while KEEPING literals (same literals => same rows)."""
    s = s.lower()
    s = re.sub(r"```sql|```", " ", s)
    s = re.sub(r"\bdistinct\b", " ", s)
    s = re.sub(r"\bas\s+[a-z_]\w*", " ", s)  # drop alias declarations, any alias name
    s = re.sub(r"\b[a-z_]\w*\.", "", s)  # drop table qualifier before a column
    s = re.sub(r"\bas\b", " ", s)
    s = re.sub(r"\binner\s+join\b", "join", s)
    s = re.sub(r"[\s();]+", " ", s)
    s = re.sub(r"\s*([=<>,])\s*", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def load() -> tuple[list[dict], list[dict]]:
    """Читает вопросы оценки (только целевые базы) и обучающие пары."""
    gold = load_json(BIRD_EVAL_FILE)
    ev = [g for g in gold if g["db_id"] in TARGET_DBS]
    train = load_json(TRAIN_FILE)
    return ev, train


def find_leaks(ev: list[dict], train: list[dict]) -> tuple[dict[str, list[str]], set[int]]:
    """Return {eval_qid: [train questions]} and the set of leaking train indices."""
    eval_sql_by_db = defaultdict(dict)  # db -> {norm_sql: qid}
    for g in ev:
        eval_sql_by_db[g["db_id"]][norm_sql(g["SQL"])] = str(g["question_id"])

    contaminated = defaultdict(list)
    leak_train_idx = set()
    for i, t in enumerate(train):
        key = norm_sql(t["sql"])
        qid = eval_sql_by_db.get(t["db_id"], {}).get(key)
        if qid is not None:
            contaminated[qid].append(t["question"])
            leak_train_idx.add(i)
    return contaminated, leak_train_idx


def main() -> int:
    parser = argparse.ArgumentParser(description="Аудит утечки train -> eval")
    parser.add_argument("--write-clean", action="store_true", help=f"записать {CLEAN_OUT.name} без утекших пар")
    args = parser.parse_args()

    ev, train = load()
    contaminated, leak_train_idx = find_leaks(ev, train)
    ev_by_id = {str(g["question_id"]): g for g in ev}

    print(f"eval questions (target dbs): {len(ev)}")
    print(f"train pairs                : {len(train)}")
    print(f"CONTAMINATED eval questions: {len(contaminated)}  (exact gold-SQL twin, same literals)")
    for qid in sorted(contaminated, key=int):
        g = ev_by_id[qid]
        print(f"  q{qid} [{g['db_id']}] {g['question']}")
        for tq in contaminated[qid]:
            print(f"      <- train: {tq}")

    if args.write_clean:
        clean = [t for i, t in enumerate(train) if i not in leak_train_idx]
        dump_json(CLEAN_OUT, clean)
        print(f"\nwrote {CLEAN_OUT.name}: {len(train)} -> {len(clean)} (removed {len(leak_train_idx)} leaking pairs)")

    if contaminated:
        print(
            "\nLEAKAGE FOUND: report EX on the clean subset "
            f"(exclude {sorted(contaminated, key=int)}) or retrain on train_clean.json"
        )
        return 1
    print("\nno exact-SQL leakage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
