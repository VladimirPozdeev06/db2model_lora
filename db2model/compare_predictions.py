"""Compare two prediction files question-by-question to localise a baseline
disagreement between the two branches.

The baseline is deterministic (greedy, temperature=0), so two correct pipelines on
the same split must produce the same EX. When they do not, the divergence is
either in *generation* (different model / prompt / schema / evidence produce
different SQL) or in *evaluation* (same SQL scored differently). This tool answers
the first half without a database: it diffs the predicted SQL per question_id.

    predictions identical, EX differs  -> the evaluator/parser diverges
    predictions differ                 -> generation diverges (look at those Qs)

Prediction files are {question_id: sql}, as produced by extract_sql.py. To attach
db_id and split the report per database, the eval file (bird_large.json) is read
too.

Usage:
    uv run python db2model/compare_predictions.py mine.json theirs.json
    uv run python db2model/compare_predictions.py mine.json theirs.json --show 20
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

EVAL_FILE = Path(__file__).resolve().parent.parent / "data" / "bird_large.json"


def _norm(sql: str) -> str:
    """Whitespace- and case-insensitive, so cosmetic formatting is not counted as a
    disagreement. Literals are kept — a different constant is a real difference."""
    return re.sub(r"\s+", " ", (sql or "").strip().rstrip(";")).lower()


def _load(path: str) -> dict[str, str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    # Accept both {id: sql} and {id: {"query": sql}} shapes.
    out = {}
    for k, v in raw.items():
        out[str(k)] = v if isinstance(v, str) else (v.get("query") or v.get("sql") or "")
    return out


def main() -> None:
    argv = sys.argv[1:]
    show = 10
    if "--show" in argv:
        i = argv.index("--show")
        show = int(argv[i + 1])
        del argv[i:i + 2]  # drop the flag and its value before reading positionals
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 2:
        print("нужно два файла предсказаний: mine.json theirs.json")
        return

    a, b = _load(args[0]), _load(args[1])
    db_of = {str(q["question_id"]): q["db_id"]
             for q in json.loads(EVAL_FILE.read_text(encoding="utf-8"))}

    keys = sorted(set(a) & set(b), key=int)
    only_a, only_b = set(a) - set(b), set(b) - set(a)
    same = [k for k in keys if _norm(a[k]) == _norm(b[k])]
    diff = [k for k in keys if _norm(a[k]) != _norm(b[k])]

    print(f"файл A: {args[0]}  ({len(a)} предсказаний)")
    print(f"файл B: {args[1]}  ({len(b)} предсказаний)")
    print(f"общих вопросов: {len(keys)} | совпал SQL: {len(same)} | различается: {len(diff)}")
    if only_a or only_b:
        print(f"только в A: {len(only_a)} | только в B: {len(only_b)}")

    if not diff:
        print("\nПредсказания идентичны -> расхождение EX не в генерации, а в оценщике/парсере.")
        return

    by_db = Counter(db_of.get(k, "?") for k in diff)
    print(f"\nразличается по базам: {dict(by_db)}")
    print(f"\nпервые {min(show, len(diff))} расхождений (id | db):")
    for k in diff[:show]:
        print(f"\n  q{k}  [{db_of.get(k, '?')}]")
        print(f"    A: {a[k].strip()[:180]}")
        print(f"    B: {b[k].strip()[:180]}")


if __name__ == "__main__":
    main()
