"""Assemble the filtered pairs into a training set: one file for training, one
for validation, plus a manifest describing exactly what went in.

The manifest exists because the training runs are the expensive part of this
project and their results are only meaningful if you can say which data produced
them. It records the input hashes, the split seed and the composition, so a run
can be traced back to its dataset.

Usage:
    uv run python db2model/build_dataset.py
    uv run python db2model/build_dataset.py toxicology financial
"""

import hashlib
import json
import random
import sys
from collections import Counter
from datetime import date
from pathlib import Path

from check_leakage import norm_sql

CLEAN_DIR = Path(__file__).parent / "clean"
OUT_DIR = Path(__file__).parent / "dataset"
EVAL_FILE = Path(__file__).resolve().parent.parent / "data" / "bird_large.json"
TARGET_DBS = ["financial", "toxicology", "codebase_community"]
VAL_FRACTION = 0.15
SEED = 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def drop_eval_leaks(records: list[dict]) -> tuple[list[dict], int]:
    """Remove synthetic pairs whose gold SQL matches an eval question's gold SQL
    (same db, same literals). The generator is seeded with real DB values, so it
    can reproduce an eval answer-query verbatim; training on it inflates EX. See
    check_leakage.py."""
    eval_sql = {}  # db -> set(norm_sql)
    for g in json.loads(EVAL_FILE.read_text(encoding="utf-8")):
        eval_sql.setdefault(g["db_id"], set()).add(norm_sql(g["SQL"]))
    kept = [r for r in records
            if norm_sql(r["sql"]) not in eval_sql.get(r["db_id"], set())]
    return kept, len(records) - len(kept)


def load(db_ids: list[str]) -> tuple[list[dict], dict]:
    records: list[dict] = []
    sources: dict[str, str] = {}

    for db_id in db_ids:
        path = CLEAN_DIR / f"{db_id}.json"
        if not path.exists():
            print(f"  пропускаю {db_id}: нет {path}")
            continue
        sources[db_id] = _sha256(path)
        for pair in json.loads(path.read_text(encoding="utf-8")):
            records.append({
                "db_id": db_id,
                "question": pair["question"],
                "sql": pair["sql"],
                "difficulty": pair.get("difficulty", "unknown"),
                "source": "synthetic",
            })
    return records, sources


def main() -> None:
    db_ids = sys.argv[1:] or TARGET_DBS
    records, sources = load(db_ids)
    if not records:
        print("нечего собирать — сначала generate_pairs.py и filter_pairs.py")
        return

    records, n_leaks = drop_eval_leaks(records)
    if n_leaks:
        print(f"деконтаминация: выкинул {n_leaks} пар, совпадающих с gold оценки")

    # Split per database, otherwise a small database can end up with no validation
    # rows at all and its own quality becomes invisible.
    rng = random.Random(SEED)
    train: list[dict] = []
    val: list[dict] = []
    for db_id in {r["db_id"] for r in records}:
        rows = [r for r in records if r["db_id"] == db_id]
        rng.shuffle(rows)
        cut = max(1, round(len(rows) * VAL_FRACTION))
        val.extend(rows[:cut])
        train.extend(rows[cut:])
    rng.shuffle(train)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "train.json").write_text(
        json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "val.json").write_text(
        json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "built": date.today().isoformat(),
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "n_train": len(train),
        "n_val": len(val),
        "eval_leaks_removed": n_leaks,
        "by_db": dict(Counter(r["db_id"] for r in records)),
        "by_difficulty": dict(Counter(r["difficulty"] for r in records)),
        "source_files": sources,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"train {len(train)} | val {len(val)}")
    print("по базам:      ", manifest["by_db"])
    print("по сложности:  ", manifest["by_difficulty"])
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    main()
