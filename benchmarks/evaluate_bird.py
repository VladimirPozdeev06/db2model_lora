import os
import json
import logging
import math
import time
import sqlglot
import decimal
import re

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict

logger = logging.getLogger(__name__)


def _time_query(conn, sql: str, repeats: int) -> float:
    """Median wall-clock execution time of a query over `repeats` runs.

    Median, not mean, so a single cold/contended run over the ssh tunnel does
    not dominate. Used only for VES, and only on queries already known correct.
    """
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        conn.execute(text(sql)).fetchall()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def sqlite_to_postgres(query: str) -> str:
    # Strip ```sql``` and comments first if needed
    query = re.sub(r"```sql|```", "", query, flags=re.IGNORECASE)
    query = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL).strip()

    # Transpile using sqlglot
    try:
        query_pg = sqlglot.transpile(query, read='sqlite', write='postgres')[0]
    except Exception as e:
        print("SQLGlot parse error:", e)
        query_pg = query

    # Division by zero is already handled by sqlglot's sqlite->postgres transpile,
    # which wraps every divisor in NULLIF(x, 0).

    return query_pg


def run_evaluation(
    predictions: Dict[str, str],
    answer_file: str,
    db_url: str,
    ves_repeats: int = 5,
):
    with open(answer_file, "r") as f:
        answer_file = json.load(f)

    gold_queries = {str(item["question_id"]): item for item in answer_file}

    results = []

    db_username = os.environ["DB_USER"]
    db_password = os.environ["DB_PASS"]

    all_predicted = {}
    all_gold = {}

    for question_id, predicted_sql in predictions.items():
        gold_query = gold_queries[question_id]

        gold_sql = gold_query["SQL"]
        db_id = gold_query["db_id"]
        difficulty = gold_query["difficulty"]

        # ---- ambiguous request processing ----
        if gold_sql == "ambiguous" or predicted_sql == "ambiguous":
            if gold_sql == "ambiguous" and predicted_sql == "ambiguous":
                score = 1
            else:
                score = 0

            results.append(
                {
                    "question_id": question_id,
                    "gold_sql": gold_sql,
                    "predicted_sql": predicted_sql,
                    "score": score,
                    "difficulty": difficulty,
                }
            )
            continue

        # ---- SQL execution and comparison with gold results ----
        db_uri = (
            f"postgresql+psycopg://{db_username}:{db_password}@{db_url}/{db_id}"
        )

        try:
            # predicted_sql = sqlite_to_postgres(predicted_sql)
            gold_sql = sqlite_to_postgres(gold_sql)
            # A predicted query can be accidentally quadratic (cross join, bad
            # subquery) and hang the whole evaluation; cap every statement.
            engine = create_engine(
                db_uri,
                connect_args={"options": "-c statement_timeout=30000"},
            )

            with engine.connect() as conn:

                gold_res = conn.execute(text(gold_sql)).fetchall()
                all_gold[question_id] = [list(row) for row in gold_res]

                pred_res = conn.execute(text(predicted_sql)).fetchall()
                all_predicted[question_id] = [list(row) for row in pred_res]

                score = set(pred_res) == set(gold_res)

                # VES (Valid Efficiency Score): defined only over correct
                # queries — how fast is the prediction relative to gold.
                # R = sqrt(t_gold / t_pred); wrong queries contribute 0.
                exec_ratio = 0.0
                if score:
                    t_gold = _time_query(conn, gold_sql, ves_repeats)
                    t_pred = _time_query(conn, predicted_sql, ves_repeats)
                    if t_pred > 0:
                        exec_ratio = math.sqrt(t_gold / t_pred)

            results.append(
                {
                    "question_id": question_id,
                    "gold_sql": gold_sql,
                    "predicted_sql": predicted_sql,
                    "score": score,
                    "difficulty": difficulty,
                    "exec_ratio": exec_ratio,
                }
            )

        except SQLAlchemyError as e:
            logger.info(
                f"Failed to process sql query for question {question_id}: '{predicted_sql}'"
            )
            results.append(
                {
                    "question_id": question_id,
                    "gold_sql": gold_sql,
                    "predicted_sql": predicted_sql,
                    "score": 0,
                    "difficulty": difficulty,
                    "error": str(e),
                }
            )

    with open("all_predicted_results.json", "w", encoding="utf-8") as f:
        json.dump(all_predicted, f, ensure_ascii=False, indent=2, default=lambda x: float(x) if isinstance(x, decimal.Decimal) else str(x))

    with open("all_gold_results.json", "w", encoding="utf-8") as f:
        json.dump(all_gold, f, ensure_ascii=False, indent=2, default=lambda x: float(x) if isinstance(x, decimal.Decimal) else str(x))


    # ---- accuracy calculation ----

    def accuracy(rows):
        if not rows:
            return 0.0
        return 100.0 * sum(r["score"] for r in rows) / len(rows)

    def ves(rows):
        # VES over all questions: correct queries contribute sqrt(t_gold/t_pred),
        # everything else contributes 0. Same denominator as EX, so VES <= EX
        # only when predictions are on average slower than gold.
        if not rows:
            return 0.0
        return 100.0 * sum(r.get("exec_ratio", 0.0) for r in rows) / len(rows)

    total_acc = accuracy(results)
    total_ves = ves(results)

    by_difficulty = {}
    ves_by_difficulty = {}
    for diff in set(r["difficulty"] for r in results):
        subset = [r for r in results if r["difficulty"] == diff]
        by_difficulty[diff] = accuracy(subset)
        ves_by_difficulty[diff] = ves(subset)

    false_ambiguous = sum(
        1 for r in results if r["predicted_sql"] == "ambiguous"
    )

    false_ambiguous_rate = (
        100.0 * false_ambiguous / len(results) if results else 0.0
    )

    print(results)
    report = {
        "overall_accuracy": total_acc,
        "accuracy_by_difficulty": by_difficulty,
        "overall_ves": total_ves,
        "ves_by_difficulty": ves_by_difficulty,
        "false_ambiguous": false_ambiguous,
        "false_ambiguous_rate": false_ambiguous_rate,
        "total": len(results),
        "results": results,
    }

    return report

def print_evaluation_report(report: dict):
    print("\n================ BIRD Benchmark Results ====================\n")

    # ---- overall ----
    print(f"Overall accuracy (EX): {report['overall_accuracy']:.2f}%")
    print(f"Overall VES          : {report.get('overall_ves', 0.0):.2f}%")
    print(f" Total queries : {report['total']}")

    # ---- difficulty ----
    print()
    print("EX / VES by difficulty:")
    ves_by_diff = report.get("ves_by_difficulty", {})
    for diff, acc in sorted(report["accuracy_by_difficulty"].items()):
        print(f"  {diff:<12}: EX {acc:5.2f}%   VES {ves_by_diff.get(diff, 0.0):5.2f}%")
    print()

    print(f"False ambiguous predicted : {report['false_ambiguous']}")
    print(f"False ambiguous rate      : {report['false_ambiguous_rate']:.2f}%")

    print("============================================================\n")
