"""Воспроизведение лидерборда одной командой (DoD недели 1).

Прогоняет сохранённые предсказания всех армов через единый оценщик и печатает
готовые markdown-таблицы: EX, VES, разбивку по сложности, среднее ± σ по сидам.

    PYTHONIOENCODING=utf-8 uv run --env-file .env python db2model/reproduce_leaderboard.py

Нужен ssh-туннель к БД (SQL исполняется на живой базе). Полный прогон — около двух
часов: слабые армы (zeroshot, baseline) упираются в statement_timeout=30s почти на
каждом вопросе. Чтобы пересчитать только часть:

    ... reproduce_leaderboard.py --only 7b            # арм(ы) по подстроке имени
    ... reproduce_leaderboard.py --group canon        # canon | curve
    ... reproduce_leaderboard.py --out results.json   # сохранить числа

σ считается по генеральной совокупности (как во всех отчётах ветки: делитель n, не n−1).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import pathlib
import statistics
import sys

from utils import BIRD_FILTERED_FILE, REPO_ROOT, RESULTS_DIR, dump_json, load_json

sys.path.insert(0, str(REPO_ROOT))

from benchmarks.evaluate_bird import run_evaluation  # noqa: E402

RESULTS = RESULTS_DIR
GOLD = BIRD_FILTERED_FILE

# (группа, метка, target, [файлы предсказаний]) — несколько файлов = сиды.
ARMS: list[tuple[str, str, str, list[str]]] = [
    ("canon", "zeroshot (контроль)", "3B", ["query_results_qwen27b_sqlonly_zeroshot.json"]),
    ("canon", "baseline (схема в контексте)", "3B", ["query_results_qwen27b_sqlonly_baseline.json"]),
    ("canon", "new_lora (+мультизадачи)", "3B", ["query_results_qwen27b_multitask_lora.json"]),
    (
        "canon",
        "qwen27b_sql_only (3 сида)",
        "3B",
        [
            "query_results_qwen27b_sqlonly_seed0.json",
            "query_results_qwen27b_sqlonly_seed1.json",
            "query_results_qwen27b_sqlonly_seed2.json",
        ],
    ),
    ("canon", "baseline (схема в контексте)", "7B", ["query_results_qwen27b_7b_baseline.json"]),
    ("canon", "reasoning (CoT)", "7B", ["query_results_reason_lora.json"]),
    (
        "canon",
        "qwen27b_sql_only (3 сида)",
        "7B",
        [
            "query_results_qwen27b_7b_seed0.json",
            "query_results_qwen27b_7b_seed1.json",
            "query_results_qwen27b_7b_seed2.json",
        ],
    ),
    # Мультизадачность ROUTE на 7B: закрывает дыру матрицы (на 3B мерилась одним
    # сидом и попала внутрь ±σ победителя). Файлов может ещё не быть — арм скипнется.
    (
        "canon",
        "multitask ROUTE (3 сида)",
        "7B",
        [
            "query_results_multitask7b_seed0.json",
            "query_results_multitask7b_seed1.json",
            "query_results_multitask7b_seed2.json",
        ],
    ),
    # Кривая объёма данных — адаптеры на старом учителе (Qwen-7B), target 3B.
    ("curve", "synth_50", "3B", ["query_results_synth_50.json"]),
    ("curve", "synth_143", "3B", ["query_results_synth_143.json"]),
    (
        "curve",
        "synth_347 (3 сида)",
        "3B",
        [
            "query_results_synth347_seed0.json",
            "query_results_synth347_seed1.json",
            "query_results_synth347_seed2.json",
        ],
    ),
    ("curve", "unfiltered_347 (без фильтра)", "3B", ["query_results_unfiltered_347.json"]),
    ("curve", "real_143 (живые пары BIRD)", "3B", ["query_results_real_143.json"]),
    (
        "curve",
        "synth_1171 (3 сида)",
        "3B",
        [
            "query_results_synth1171_seed0.json",
            "query_results_synth1171_seed1.json",
            "query_results_synth1171_seed2.json",
        ],
    ),
]


def evaluate(path: pathlib.Path, db_url: str) -> dict:
    """Один файл предсказаний -> отчёт. Внутренний дамп оценщика подавляем."""
    predictions = load_json(path)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        return run_evaluation(predictions, str(GOLD), db_url)


def pstdev(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def fmt(mean: float, sigma: float, seeds: int) -> str:
    return f"{mean:.2f} ± {sigma:.2f}" if seeds > 1 else f"{mean:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="подстрока имени файла предсказаний")
    ap.add_argument("--group", default="", choices=["", "canon", "curve"])
    ap.add_argument("--out", type=pathlib.Path, default=None, help="куда сохранить JSON с числами")
    args = ap.parse_args()

    if not GOLD.exists():
        print(f"НЕТ GOLD: {GOLD}", file=sys.stderr)
        return 2

    db_url = os.getenv("BENCHMARK_DB_URL", "localhost:5444")
    rows, collected = [], {}

    for group, label, target, files in ARMS:
        if args.group and group != args.group:
            continue
        paths = [RESULTS / f for f in files]
        if args.only and not any(args.only in f for f in files):
            continue
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            print(f"[skip] {label} {target}: нет файлов {missing}", file=sys.stderr)
            continue

        print(f"[{group}] {label} ({target}) ...", file=sys.stderr, flush=True)
        reps = [evaluate(p, db_url) for p in paths]
        ex = [r["overall_accuracy"] for r in reps]
        ves = [r.get("overall_ves", 0.0) for r in reps]
        diff = reps[0].get("accuracy_by_difficulty", {})

        row = {
            "group": group,
            "arm": label,
            "target": target,
            "seeds": len(reps),
            "ex_mean": statistics.fmean(ex),
            "ex_sigma": pstdev(ex),
            "ex_all": ex,
            "ves_mean": statistics.fmean(ves),
            "ves_sigma": pstdev(ves),
            "by_difficulty": {k: round(v, 2) for k, v in diff.items()},
            "files": files,
        }
        rows.append(row)
        collected[f"{label} [{target}]"] = row
        print(f"    EX {row['ex_mean']:.2f}  VES {row['ves_mean']:.2f}", file=sys.stderr, flush=True)

    for group, title in (("canon", "Канонический замер"), ("curve", "Кривая объёма данных")):
        sub = [r for r in rows if r["group"] == group]
        if not sub:
            continue
        print(f"\n## {title}\n")
        print("| Арм | target | EX | VES | сидов |")
        print("|---|---|---|---|---|")
        for r in sub:
            print(
                f"| {r['arm']} | {r['target']} | "
                f"{fmt(r['ex_mean'], r['ex_sigma'], r['seeds'])} | "
                f"{fmt(r['ves_mean'], r['ves_sigma'], r['seeds'])} | {r['seeds']} |"
            )

    if args.out:
        dump_json(args.out, collected)
        print(f"\nчисла сохранены: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
