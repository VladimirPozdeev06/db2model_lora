"""Разбивка EX/VES по базам (financial / toxicology / codebase_community).
Печатает ТОЛЬКО имена баз и числа. Запуск: uv run --env-file .env python db2model/score_by_db.py <файлы предсказаний...>
"""
import collections
import contextlib
import io
import os
import sys

from utils import BIRD_FILTERED_FILE, REPO_ROOT, RESULTS_DIR, load_json

sys.path.insert(0, str(REPO_ROOT))
from benchmarks.evaluate_bird import run_evaluation  # noqa: E402

gold = load_json(BIRD_FILTERED_FILE)
qid_db = {str(g["question_id"]): g["db_id"] for g in gold}
db_url = os.getenv("BENCHMARK_DB_URL", "localhost:5444")

files = sys.argv[1:]
agg = collections.defaultdict(lambda: {"n": 0, "correct": 0.0, "ves": 0.0})
for name in files:
    preds = load_json(RESULTS_DIR / name)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        rep = run_evaluation(preds, str(BIRD_FILTERED_FILE), db_url)
    for r in rep["results"]:
        db = qid_db.get(str(r["question_id"]), "?")
        agg[db]["n"] += 1
        agg[db]["correct"] += float(r["score"])
        agg[db]["ves"] += float(r.get("exec_ratio", 0.0))

nseeds = max(1, len(files))
print(f"\n=== разбивка по базам (сидов усреднено: {nseeds}) ===")
print(f"{'база':<22} {'вопросов':>9} {'EX%':>7} {'VES':>7}")
tot_n = tot_c = tot_v = 0.0
for db, a in sorted(agg.items()):
    n = a["n"]
    ex = 100.0 * a["correct"] / n if n else 0
    ves = 100.0 * a["ves"] / n if n else 0
    print(f"{db:<22} {n // nseeds:>9} {ex:>7.1f} {ves:>7.1f}")
    tot_n += n; tot_c += a["correct"]; tot_v += a["ves"]
print(f"{'ИТОГО':<22} {int(tot_n // nseeds):>9} {100.0 * tot_c / tot_n:>7.1f} {100.0 * tot_v / tot_n:>7.1f}")
