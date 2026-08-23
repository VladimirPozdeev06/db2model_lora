"""Pull the SQL out of a model reply.

An instruct model asked for "only SQL" often answers with a sentence, a fenced
code block, and a closing explanation. Stripping fences only at the ends leaves
the prose attached, the whole thing goes to the database as a query, and the
answer scores zero — a loss caused by the harness, not by the model.

This matters for fairness between arms: an untuned model narrates, a fine-tuned
one emits bare SQL, so a naive extractor silently penalises the baseline.

Usage:
    uv run python db2model/extract_sql.py db2model/results/query_results_baseline.json
"""

import argparse
import re
from pathlib import Path

from utils import dump_json, load_json

FENCED = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
STATEMENT = re.compile(r"\b(WITH|SELECT)\b", re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Return the first SQL statement in a reply, or the text itself if none."""
    text = text.strip()

    match = FENCED.search(text)
    if match:
        text = match.group(1).strip()

    # Prose before the query ("To find X, use:") — cut to where SQL starts.
    start = STATEMENT.search(text)
    if start:
        text = text[start.start() :]

    return text.strip().rstrip(";").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Вычистить SQL из ответов модели")
    parser.add_argument("predictions", type=Path, help="файл предсказаний {question_id: ответ}")
    args = parser.parse_args()

    predictions = load_json(args.predictions)
    cleaned = {qid: extract_sql(sql) for qid, sql in predictions.items()}

    changed = sum(1 for qid in predictions if predictions[qid] != cleaned[qid])
    out = args.predictions.with_name(args.predictions.stem + "_extracted.json")
    dump_json(out, cleaned)
    print(f"{args.predictions.name}: переписано {changed} из {len(predictions)} -> {out.name}")


if __name__ == "__main__":
    main()
