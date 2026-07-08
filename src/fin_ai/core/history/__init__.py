"""
Question history — CRUD for vector-db-scoped question/answer history.

These functions were extracted from ``dashboard/utils.py`` so that the core
``processor`` module does not depend on the dashboard package.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def get_question_history_path(vector_db_name: str, history_dir: Path) -> Path:
    sanitized_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", vector_db_name).strip("_")
    if not sanitized_name:
        sanitized_name = "default"
    return history_dir / f"{sanitized_name}.json"


def load_question_history(vector_db_name: str, history_dir: Path) -> list[dict[str, Any]]:
    history_path = get_question_history_path(vector_db_name, history_dir)
    if not history_path.exists():
        return []

    try:
        return json.loads(history_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def save_question_history(vector_db_name: str, history_dir: Path, history: list[dict[str, Any]]) -> None:
    history_path = get_question_history_path(vector_db_name, history_dir)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(history, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def append_question_history(
    vector_db_name: str,
    history_dir: Path,
    entry: dict[str, Any],
    limit: int = 50,
) -> None:
    history = load_question_history(vector_db_name, history_dir)
    history.insert(0, entry)
    save_question_history(vector_db_name, history_dir, history[:limit])


def clear_question_history(vector_db_name: str, history_dir: Path) -> None:
    save_question_history(vector_db_name, history_dir, [])


def purge_vector_db_assets(vector_db_name: str, vector_db_dir: Path, history_dir: Path) -> list[Path]:
    import shutil

    deleted_paths: list[Path] = []

    candidates = [
        Path(vector_db_dir) / f"{vector_db_name}.faiss",
        Path(vector_db_dir) / f"{vector_db_name}.pdf",
        Path(vector_db_dir) / f"{vector_db_name}.embedding.json",
        Path(vector_db_dir) / vector_db_name / "images",
        get_question_history_path(vector_db_name, Path(history_dir)),
    ]

    for path in candidates:
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
        deleted_paths.append(path)

    return deleted_paths