"""Build the dataset variants the ablation needs.

Each variant isolates one question:

    synth_50 / synth_143 / synth_347   сколько синтетики реально нужно
    unfiltered_347                     что даёт фильтр по исполнению
    real_143                           насколько синтетика хуже живых данных

`real_143` and `synth_143` are the same size on purpose: comparing 143 real pairs
against 347 synthetic ones would confound quality with quantity.

Usage:
    uv run python db2model/build_ablation.py
"""

import json
import random
from collections import Counter
from pathlib import Path

from sqlglot import transpile

HERE = Path(__file__).parent
CLEAN_DIR = HERE / "clean"
RAW_DIR = HERE / "raw"
OUT_DIR = HERE / "dataset"
TRAIN_QUERIES = Path("data/train_queries.json")
TARGET_DBS = ["financial", "toxicology", "codebase_community"]
SEED = 0

SYNTH_SIZES = [50, 143, 347]


def _load(directory: Path, source: str) -> list[dict]:
    records = []
    for db_id in TARGET_DBS:
        path = directory / f"{db_id}.json"
        for pair in json.loads(path.read_text(encoding="utf-8")):
            records.append({
                "db_id": db_id,
                "question": pair["question"],
                "sql": pair["sql"],
                "difficulty": pair.get("difficulty", "unknown"),
                "source": source,
            })
    return records


def real_pairs() -> list[dict]:
    """BIRD train pairs are stored in SQLite dialect; the target runs PostgreSQL,
    so they have to be transpiled or the model learns the wrong syntax.
    Disjoint from bird_large/bird_small — checked, no leakage into the metric."""
    records = []
    dropped = 0
    for item in json.loads(TRAIN_QUERIES.read_text(encoding="utf-8")):
        if item["db_id"] not in TARGET_DBS:
            continue
        try:
            sql = transpile(item["SQL"], read="sqlite", write="postgres")[0]
        except Exception:
            dropped += 1
            continue
        records.append({
            "db_id": item["db_id"],
            "question": item["question"],
            "sql": sql,
            "difficulty": item.get("difficulty", "unknown"),
            "source": "real",
        })
    if dropped:
        print(f"  real: {dropped} пар не транспилировались, выкинуты")
    return records


def write(name: str, rows: list[dict]) -> None:
    path = OUT_DIR / f"train_{name}.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    by_db = Counter(r["db_id"] for r in rows)
    print(f"{name:<16} {len(rows):>4} пар | {dict(by_db)} -> {path.name}")


def main() -> None:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Синтетика после фильтра — то, на чём уже обучались.
    synth = _load(CLEAN_DIR, "synthetic")
    rng.shuffle(synth)
    for size in SYNTH_SIZES:
        write(f"synth_{size}", synth[:size])

    # Та же синтетика, но без фильтра по исполнению: столько же пар, взятых из сырых.
    raw = _load(RAW_DIR, "unfiltered")
    rng.shuffle(raw)
    write("unfiltered_347", raw[:347])

    # Живые пары из BIRD train.
    real = real_pairs()
    rng.shuffle(real)
    write(f"real_{len(real)}", real)

    manifest = {
        "seed": SEED,
        "variants": {
            p.stem: len(json.loads(p.read_text(encoding="utf-8")))
            for p in sorted(OUT_DIR.glob("train_*.json"))
        },
    }
    (OUT_DIR / "ablation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {OUT_DIR / 'ablation_manifest.json'}")


if __name__ == "__main__":
    main()
