"""Общее для скриптов db2model: пути, чтение/запись JSON, подключение к БД.

Скрипты ветки запускаются из корня репозитория (`uv run python db2model/<script>.py`),
поэтому каталог `db2model/` оказывается на `sys.path` и модуль импортируется как
`from utils import load_json`.

Базовые каталоги объявлены константами модуля; полный путь к конкретному файлу
собирается на месте вызова (в `main()` или в `argparse`), а не внутри функций.
"""

import json
import os
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

DB2MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = DB2MODEL_DIR.parent

DATA_DIR = REPO_ROOT / "data"
BIRD_EVAL_FILE = DATA_DIR / "bird_large.json"
BIRD_FILTERED_FILE = DATA_DIR / "bird_large_filtered.json"

CLEAN_DIR = DB2MODEL_DIR / "clean"
DATASET_DIR = DB2MODEL_DIR / "dataset"
PAIRS_DIR = DB2MODEL_DIR / "pairs"
PROFILES_DIR = DB2MODEL_DIR / "profiles"
RESULTS_DIR = DB2MODEL_DIR / "results"

# Три базы, на которых обучались адаптеры и снимался лидерборд.
TARGET_DBS = ("financial", "toxicology", "codebase_community")

DEFAULT_DB_HOST = "localhost:5444"
# БД доступна только через ssh-туннель. Без таймаутов оборванный туннель не роняет
# скрипт, а вешает его навсегда на установленном сокете (TCP keepalive — 2 часа).
CONNECT_TIMEOUT_SEC = 15
STATEMENT_TIMEOUT_MS = 30000


def load_json(path: Path) -> Any:
    """Читает JSON-файл в utf-8."""
    return json.loads(path.read_text(encoding="utf-8"))


def to_json_text(data: Any) -> str:
    """JSON-текст в том виде, в каком его пишет `dump_json`.

    `default=str` нужен профилям баз: в них попадают `Decimal` и `date` из pg_stats,
    которые иначе не сериализуются.
    """
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def dump_json(path: Path, data: Any) -> None:
    """Пишет JSON с отступами и кириллицей как есть; каталог создаётся при нужде."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json_text(data), encoding="utf-8")


def load_profile(db_id: str) -> dict:
    """Профиль базы, собранный `collect_profile.py`."""
    return load_json(PROFILES_DIR / f"{db_id}.json")


def build_db_uri(db_id: str, db_host: str | None = None) -> str:
    """URI подключения из кред окружения (`DB_USER`, `DB_PASS`, `BENCHMARK_DB_URL`)."""
    host = db_host or os.getenv("BENCHMARK_DB_URL", DEFAULT_DB_HOST)
    return f"postgresql+psycopg://{os.environ['DB_USER']}:{os.environ['DB_PASS']}@{host}/{db_id}"


def make_engine(
    db_id: str,
    db_host: str | None = None,
    statement_timeout_ms: int = STATEMENT_TIMEOUT_MS,
) -> Engine:
    """Подключение к базе с таймаутами на коннект и на отдельный запрос.

    `statement_timeout` задаётся параметром соединения, а не отдельным `SET`: после
    каждого упавшего запроса нужен rollback, а он откатил бы `SET`, сделанный внутри
    транзакции, и остаток прогона остался бы вообще без таймаута.
    """
    return create_engine(
        build_db_uri(db_id, db_host),
        connect_args={
            "options": f"-c statement_timeout={statement_timeout_ms}",
            "connect_timeout": CONNECT_TIMEOUT_SEC,
        },
        pool_pre_ping=True,
    )
