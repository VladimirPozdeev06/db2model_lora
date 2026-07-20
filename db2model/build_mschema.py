"""Render each database profile as an M-Schema text file — the canonical
knowledge-of-the-DB artifact required by the week-2 DoD.

M-Schema (from XiYan-SQL, https://arxiv.org/abs/2411.08599) is a compact,
semi-structured schema representation: per-table column lists with types, keys
and example values, plus foreign keys. It is the shared carrier of DB knowledge
that the other branches (baseline schema-in-prompt, light-RAG subgraph) draw from.

Input : db2model/profiles/<db>.json  (schema + FK + pg_stats + sample values)
Output: db2model/mschema/<db>.txt

Usage:
    uv run python db2model/build_mschema.py                 # all target dbs
    uv run python db2model/build_mschema.py toxicology
"""

import json
import sys
from pathlib import Path

PROFILE_DIR = Path(__file__).parent / "profiles"
OUT_DIR = Path(__file__).parent / "mschema"
TARGET_DBS = ["financial", "toxicology", "codebase_community"]
MAX_EXAMPLES = 3


def _as_list(vals):
    """pg_stats fields come either as a JSON list or a Postgres array literal
    string like '{-,+}'. Normalize both to a Python list."""
    if isinstance(vals, list):
        return vals
    if isinstance(vals, str):
        s = vals.strip()
        if s.startswith("{") and s.endswith("}"):
            s = s[1:-1]
        return [v.strip().strip('"') for v in s.split(",") if v.strip()]
    return []


def _examples(col: dict) -> list:
    for key in ("sample_values", "all_values", "most_common_vals"):
        vals = _as_list(col.get(key))
        if vals:
            return vals[:MAX_EXAMPLES]
    return []


def _fmt_val(v) -> str:
    return f"'{v}'" if isinstance(v, str) else str(v)


def render(profile: dict) -> str:
    lines = [f"【DB_ID】 {profile['db_id']}", "【Schema】"]

    for table, info in profile["tables"].items():
        pk = set(info.get("primary_key") or [])
        header = f"# Table: {table}"
        if info.get("comment"):
            header += f"  -- {info['comment']}"
        lines.append(header)
        lines.append("[")
        for col in info["columns"]:
            parts = [f"{col['name']}:{col['type']}"]
            if col["name"] in pk:
                parts.append("Primary Key")
            if col.get("nullable") is False:
                parts.append("NOT NULL")
            if col.get("comment"):
                parts.append(col["comment"])
            entry = ", ".join(parts)
            ex = _examples(col)
            if ex:
                entry += ". Examples: [" + ", ".join(_fmt_val(v) for v in ex) + "]"
            lines.append(f"({entry}),")
        lines.append("]")

    fks = profile.get("foreign_keys") or []
    if fks:
        lines.append("【Foreign keys】")
        for fk in fks:
            for fc, tc in zip(fk["from_columns"], fk["to_columns"]):
                lines.append(f"{fk['from_table']}.{fc} = {fk['to_table']}.{tc}")

    return "\n".join(lines) + "\n"


def main() -> None:
    dbs = sys.argv[1:] or TARGET_DBS
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for db in dbs:
        profile_path = PROFILE_DIR / f"{db}.json"
        if not profile_path.exists():
            print(f"  пропускаю {db}: нет {profile_path}")
            continue
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        out_path = OUT_DIR / f"{db}.txt"
        out_path.write_text(render(profile), encoding="utf-8")
        n_tables = len(profile["tables"])
        n_fk = len(profile.get("foreign_keys") or [])
        print(f"{db}: {n_tables} таблиц, {n_fk} FK -> {out_path}")


if __name__ == "__main__":
    main()
