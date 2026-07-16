import json
import os
import sys

from benchmarks.evaluate_bird import run_evaluation, print_evaluation_report

db_url = os.getenv("BENCHMARK_DB_URL", "localhost:5444")
predictions_file = sys.argv[1] if len(sys.argv) > 1 else "./query_results.json"
answer_file = sys.argv[2] if len(sys.argv) > 2 else "./data/bird_small.json"

with open(predictions_file, "r", encoding="utf-8") as f:
    predictions = json.load(f)

report = run_evaluation(predictions, answer_file, db_url)

print_evaluation_report(report)
