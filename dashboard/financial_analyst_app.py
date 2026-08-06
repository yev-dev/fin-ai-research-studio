"""Financial Data Analysis — backend-aware Streamlit entry point.

This module serves two roles:
- when imported, it exposes the dashboard backend facade;
- when executed with ``streamlit run``, it launches the UI dashboard.
"""

from __future__ import annotations

import os
import sys
import re
from html import unescape
from pathlib import Path

from dashboard import PARENT_DIR, SRC_DIR

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.chdir(str(PARENT_DIR))

from fin_ai.core.embeddings import create_embeddings
from fin_ai.core.processor import (
    SUPPORTED_UPLOAD_TYPES,
    answer_question,
    build_agent_llm_config,
    build_query_source_configs,
    clear_history,
    discover_source_groups,
    fetch_models,
    filter_stores_by_source_groups,
    get_source_vector_stores,
    get_vector_db_names,
    load_history,
    load_vector_stores_for_query,
    process_uploaded_document,
    purge_vector_db,
    run_agent_task,
    save_history_entry,
)
from fin_ai.config.fin_ai import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDINGS_PROVIDER,
    DEFAULT_GITHUB_MODEL,
    DEFAULT_GITHUB_EMBEDDING_MODEL,
    DEFAULT_DEEPSEEK_MODEL,
    DEEPSEEK_BASE_URL,
    GITHUB_EMBEDDING_BASE_URL,
    OLLAMA_BASE_URL,
    PUBLISHED_RESEARCH_DIR,
    VECTOR_DB_DIR,
)
from fin_ai.core.providers import list_models
from fin_ai.core.query import format_source_citations
from fin_ai.core.rag import load_embedding_metadata
from fin_ai.core.request import known_providers, get_provider_config
from dashboard.utils import get_pdf_text
from fin_ai.agents.agent_library import library as agent_library
from fin_ai.agents.prompts_library import RESEARCH_ANALYSIS

__all__ = [
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDINGS_PROVIDER",
    "DEFAULT_GITHUB_MODEL",
    "DEFAULT_GITHUB_EMBEDDING_MODEL",
    "DEFAULT_DEEPSEEK_MODEL",
    "DEEPSEEK_BASE_URL",
    "GITHUB_EMBEDDING_BASE_URL",
    "OLLAMA_BASE_URL",
    "PUBLISHED_RESEARCH_DIR",
    "VECTOR_DB_DIR",
    "SUPPORTED_UPLOAD_TYPES",
    "RESEARCH_ANALYSIS",
    "agent_library",
    "answer_question",
    "build_agent_llm_config",
    "build_query_source_configs",
    "clear_history",
    "create_embeddings",
    "discover_source_groups",
    "fetch_models",
    "filter_stores_by_source_groups",
    "format_source_citations",
    "get_provider_config",
    "get_source_vector_stores",
    "get_vector_db_names",
    "known_providers",
    "list_models",
    "load_embedding_metadata",
    "load_history",
    "load_vector_stores_for_query",
    "process_uploaded_document",
    "purge_vector_db",
    "run_agent_task",
    "save_history_entry",
]


def _truncate_text(value: str, limit: int = 180) -> str:
    collapsed = re.sub(r"\s+", " ", value or "").strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def _humanize_report_title(file_path: Path) -> str:
    stem = file_path.stem
    stem = re.sub(r"_(\d{8}_\d{6})(?:_printable)?$", "", stem)
    stem = re.sub(r"[_-]+", " ", stem).strip()
    return stem or file_path.name


def _describe_published_document(file_path: Path) -> tuple[str, str]:
    title = _humanize_report_title(file_path)
    description = ""

    try:
        if file_path.suffix.lower() == ".pdf":
            page_texts = get_pdf_text(str(file_path))
            first_page_text = page_texts[0] if page_texts else ""
            description = _truncate_text(first_page_text)
        else:
            html_text = file_path.read_text(encoding="utf-8", errors="ignore")
            title_match = re.search(r"<title>(.*?)</title>", html_text, flags=re.IGNORECASE | re.DOTALL)
            if title_match:
                title = _truncate_text(unescape(title_match.group(1).strip()), 120)

            body_text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html_text)
            body_text = re.sub(r"(?s)<[^>]+>", " ", body_text)
            description = _truncate_text(unescape(body_text))
    except Exception:
        description = ""

    if not description:
        description = "Published research report"

    return title, description


def _mime_for_published_document(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in {".html", ".htm"}:
        return "text/html"
    return "application/octet-stream"


def _load_published_documents() -> list[dict[str, object]]:
    published_dir = Path(PUBLISHED_RESEARCH_DIR)
    if not published_dir.exists():
        return []

    candidates = [
        path for path in published_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".html", ".htm", ".pdf"}
    ]
    documents: list[dict[str, object]] = []
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True):
        title, description = _describe_published_document(path)
        documents.append({
            "path": path,
            "title": title,
            "description": description,
            "filename": path.name,
            "mtime": path.stat().st_mtime,
            "mime": _mime_for_published_document(path),
        })
    return documents


def _launch_dashboard() -> None:
    dashboard_path = Path(__file__).with_name("financial_analyst_dashboard.py")
    with dashboard_path.open(encoding="utf-8") as file_handle:
        dashboard_globals = {
            "__name__": "__main__",
            "__file__": str(dashboard_path),
            "__package__": "dashboard",
        }
        exec(compile(file_handle.read(), str(dashboard_path), "exec"), dashboard_globals, dashboard_globals)


if __name__ == "__main__":
    _launch_dashboard()
