"""Точный замер prompt-токенов на запрос для армов ветки SFT/LoRA.

Определение (согласовано для обеих веток): токены промпта того единственного вызова,
что генерирует SQL, посчитанные ТОКЕНАЙЗЕРОМ САМОЙ target-модели (Qwen2.5-Coder), а не
cl100k. Промпт строится ровно как в `kaggle_infer_7b.ipynb` (функции `schema_text` и
`build_prompt`), которым и генерились предсказания лидерборда:

  system (schema-less) = "You are a PostgreSQL expert for the database `<db>`. Return only SQL."
  system (baseline)    = + "\n\nSchema:\n" + компактная схема table(col type, ...) из profiles/<db>.json
  user                 = "question: <q>, evidence (may be empty): <evidence>"
  prompt               = tokenizer.apply_chat_template([system, user], add_generation_prompt=True)

Каждый арм = ОДИН вызов на вопрос, так что «только генерация» = весь промпт вопроса.
Токенайзер Qwen2.5-Coder-3B и -7B Instruct идентичны (одна семья), поэтому число одно на оба.

ВАЖНО (правило №2): скрипт печатает ТОЛЬКО числа (счётчики токенов, mean/median/sd),
тексты вопросов и схем не выводятся.

Запуск:  python db2model/measure_tokens.py
Нужен `transformers`; токенайзер тянется с HuggingFace один раз (публичный).
"""

import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer

from utils import BIRD_EVAL_FILE, PROFILES_DIR, TARGET_DBS, RESULTS_DIR, dump_json, load_json

MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"  # токенайзер 3B и 7B Instruct идентичны

profiles = {db: load_json(PROFILES_DIR / f"{db}.json") for db in TARGET_DBS}


def schema_text(db: str) -> str:
    """Дословно из kaggle_infer_7b.ipynb: компактная схема table(col type, ...)."""
    lines = []
    for table, info in profiles[db]["tables"].items():
        cols = ", ".join(f"{c['name']} {c['type']}" for c in info["columns"])
        lines.append(f"{table}({cols})")
    return "\n".join(lines)


def build_messages(db: str, question: str, with_schema: bool):
    system = f"You are a PostgreSQL expert for the database `{db}`. Return only SQL."
    if with_schema:
        system += f"\n\nSchema:\n{schema_text(db)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    bird = load_json(BIRD_EVAL_FILE)
    qs = [q for q in bird if q["db_id"] in TARGET_DBS]

    arms = {
        "schema_less (lora / zeroshot)": False,  # with_schema=False
        "baseline (schema in context)": True,     # with_schema=True
    }

    out = {"model_tokenizer": MODEL, "definition": "prompt tokens of the single SQL-generation call, target-model tokenizer, apply_chat_template(add_generation_prompt=True)", "n_questions": len(qs), "arms": {}}
    for arm, with_schema in arms.items():
        counts = []
        per_db: dict[str, list[int]] = {db: [] for db in TARGET_DBS}
        for q in qs:
            question = f"question: {q['question']}, evidence (may be empty): {q['evidence']}"
            prompt = tok.apply_chat_template(
                build_messages(q["db_id"], question, with_schema),
                tokenize=False,
                add_generation_prompt=True,
            )
            # строка уже содержит спец-токены шаблона (<|im_start|> …) как текст,
            # поэтому add_special_tokens=False — считаем ровно то, что уходит на сервер.
            n = len(tok(prompt, add_special_tokens=False)["input_ids"])
            counts.append(n)
            per_db[q["db_id"]].append(n)
        out["arms"][arm] = {
            "mean": round(statistics.mean(counts), 1),
            "median": int(statistics.median(counts)),
            "sd": round(statistics.pstdev(counts), 1),
            "min": min(counts),
            "max": max(counts),
            "per_db_mean": {db: round(statistics.mean(v), 1) for db, v in per_db.items()},
        }

    dump_json(RESULTS_DIR / "token_budget.json", out)
    # печать только чисел
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
