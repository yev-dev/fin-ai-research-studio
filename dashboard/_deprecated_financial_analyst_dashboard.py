"""
Financial Data Analysis — business logic and external system communication.

This module contains all business logic, external API calls, and data
processing functions.  The presentation layer (Streamlit UI) lives in
``financial_analyst_app.py`` and imports from here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS

from dashboard import (
    DEFAULT_GITHUB_MODEL,
    DEFAULT_GITHUB_EMBEDDING_MODEL,
    GITHUB_EMBEDDING_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DEEPSEEK_BASE_URL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDINGS_PROVIDER,
    OLLAMA_BASE_URL,
    VECTOR_DB_DIR,
)
from dashboard.utils import (
    execute_python_code,
    extract_python_code,
    render_pdf_pages,
    render_csv_thumbnail,
    sanitize_generated_python_code,
)
from fin_ai.core.embeddings import create_embeddings
from fin_ai.core.processor import (
    SUPPORTED_UPLOAD_TYPES,
    answer_question,
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
    save_history_entry,
    build_agent_llm_config,
    run_agent_task,
)
from fin_ai.core.providers import list_models
from fin_ai.core.query import format_source_citations
from fin_ai.core.rag import load_embedding_metadata
from fin_ai.core.request import known_providers, get_provider_config
from fin_ai.agents.agent_library import library as agent_library


def looks_like_embedding_model(model_id: str) -> bool:
    """Heuristic filter for embedding-capable model identifiers."""
    mid = (model_id or "").strip().lower()
    return "embedding" in mid or "embed" in mid


def discover_vector_stores() -> tuple[dict[str, Any], list[str]]:
    """Discover available vector stores and return (source_vector_stores, vector_db_names)."""
    source_vector_stores = get_source_vector_stores()
    vector_db_names = get_vector_db_names(source_vector_stores)
    return source_vector_stores, vector_db_names


def get_local_model_options() -> tuple[list[str], list[str], str | None]:
    return [DEFAULT_CHAT_MODEL], [DEFAULT_EMBEDDING_MODEL], None


def get_cached_csv_thumbnail_path(csv_path: str, images_folder: str, source_mtime: float) -> str:
    del source_mtime
    return render_csv_thumbnail(csv_path, images_folder)


def get_cached_pdf_page_paths(pdf_path: str, images_folder: str, zoom: float, source_mtime: float) -> tuple[str, ...]:
    del source_mtime
    return tuple(str(p) for p in render_pdf_pages(pdf_path, images_folder, zoom=zoom))


def get_provider_configuration(provider_key: str) -> Any:
    """Get the provider configuration for a given provider key."""
    return get_provider_config(provider_key)


def get_known_providers() -> dict[str, Any]:
    """Return all known providers with their label→key mapping."""
    return known_providers()


def fetch_github_models(api_key: str) -> list[Any]:
    """Fetch available GitHub models."""
    return fetch_models("github", api_key=api_key)


def fetch_deepseek_models(api_key: str) -> list[Any]:
    """Fetch available DeepSeek models."""
    return fetch_models("deepseek", api_key=api_key)


def fetch_github_embedding_models(api_key: str) -> list[str]:
    """Fetch embedding-capable GitHub models."""
    try:
        emb_models = fetch_models("github", api_key=api_key)
        return [m.id for m in emb_models if looks_like_embedding_model(m.id)]
    except Exception:
        return []


def load_embeddings_for_query(
    vector_db_names: list[str],
    selected_emb_provider: str,
    selected_embedding_model: str,
    embeddings_base_url: str,
    embedding_github_token: str | None = None,
) -> Any | None:
    """Load embeddings for the first vector DB, falling back to configured provider."""
    if not vector_db_names:
        return None
    _github_token = os.environ.get("GITHUB_TOKEN", "")
    if selected_emb_provider == "github":
        _github_token = embedding_github_token or _github_token
    try:
        saved_emb_meta = load_embedding_metadata(vector_db_names[0])
        if saved_emb_meta:
            return create_embeddings(
                provider=saved_emb_meta["provider"],
                model=saved_emb_meta["model"],
                api_base=saved_emb_meta["base_url"],
                api_key=_github_token if saved_emb_meta["provider"] == "github" else None,
            )
        else:
            return create_embeddings(
                provider=selected_emb_provider,
                model=selected_embedding_model,
                api_base=embeddings_base_url,
                api_key=_github_token if selected_emb_provider == "github" else None,
            )
    except Exception:
        return None


def load_vector_stores(
    selected_query_vector_dbs: list[str],
    source_vector_stores: dict[str, Any],
    embeddings: Any,
) -> tuple[dict[str, FAISS], list[Any], list[str], dict[str, list[str]]]:
    """Load vector stores and build query configs. Returns (loaded_stores, configs, source_names, source_groups_map)."""
    loaded_stores = load_vector_stores_for_query(selected_query_vector_dbs, source_vector_stores, embeddings)
    query_source_configs = build_query_source_configs(loaded_stores, group_by="vector_db")
    available_source_names = [c.name for c in query_source_configs]
    source_groups_map = discover_source_groups(loaded_stores)
    available_source_groups = sorted(source_groups_map.keys())
    return loaded_stores, query_source_configs, available_source_names, available_source_groups


def filter_stores_by_groups(loaded_stores: dict[str, FAISS], selected_source_groups: list[str]) -> dict[str, FAISS]:
    """Filter loaded stores to only those in selected source groups."""
    return filter_stores_by_source_groups(loaded_stores, selected_source_groups)


def process_upload(
    file_binary: bytes,
    file_name: str,
    embedding_model: str,
    embedding_base_url: str,
    emb_provider: str,
    source_type: str,
    github_token: str | None = None,
) -> dict[str, Any]:
    """Process an uploaded document and store it in the vector DB."""
    return process_uploaded_document(
        file_binary=file_binary,
        file_name=file_name,
        embedding_model=embedding_model,
        embedding_base_url=embedding_base_url,
        emb_provider=emb_provider,
        source_type=source_type,
        github_token=github_token,
    )


def answer_query(
    question: str,
    active_configs: list[Any],
    provider: str,
    retrieval_mode: str,
    auto_truncate_prompt: bool,
    use_tools: bool,
) -> dict[str, Any]:
    """Answer a question using the RAG pipeline."""
    return answer_question(
        question,
        active_configs,
        provider=provider,
        system_prompt="You are a concise financial analysis assistant.",
        temperature=0.2,
        retrieval_mode=retrieval_mode,
        auto_truncate_prompt=auto_truncate_prompt,
        use_tools=use_tools,
    )


def run_agent(
    agent_name: str,
    prompt: str,
    llm_config: dict[str, Any],
    embedding_model: str,
    embedding_provider: str,
    embedding_base_url: str,
    chat_provider: str,
    is_publisher: bool,
    publisher_format: str,
    publisher_email: str,
) -> dict[str, Any]:
    """Run an agent task."""
    return run_agent_task(
        agent_name=agent_name,
        prompt=prompt,
        llm_config=llm_config,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        embedding_base_url=embedding_base_url,
        chat_provider=chat_provider,
        is_publisher=is_publisher,
        publisher_format=publisher_format,
        publisher_email=publisher_email,
    )


def build_llm_config(
    provider: str,
    model: str,
    ollama_base_url: str,
    github_endpoint: str,
    github_token: str,
    deepseek_base_url: str,
    deepseek_token: str,
) -> dict[str, Any]:
    """Build the LLM configuration dict for agent execution."""
    return build_agent_llm_config(
        provider=provider,
        model=model,
        ollama_base_url=ollama_base_url,
        github_endpoint=github_endpoint,
        github_token=github_token,
        deepseek_base_url=deepseek_base_url,
        deepseek_token=deepseek_token,
    )


def load_question_history(vector_db_names: list[str]) -> tuple[list[dict[str, Any]], str | None]:
    """Load question history from the first available vector DB."""
    if not vector_db_names:
        return [], None
    _history_db_name = vector_db_names[0]
    history = load_history(_history_db_name)
    return history, _history_db_name


def save_question_history_entry(
    history_db: str,
    entry: dict[str, Any],
) -> None:
    """Save a question history entry."""
    save_history_entry(history_db, entry)


def clear_question_history(history_db: str) -> None:
    """Clear question history for a given DB."""
    clear_history(history_db)


def purge_vector_database(db_name: str) -> bool:
    """Purge a vector database and return True if successful."""
    return purge_vector_db(db_name)


def get_source_citations(retrieval: Any, response_type: str) -> str | None:
    """Format source citations from retrieval results."""
    return format_source_citations(retrieval, response_type=response_type)


def get_agent_library() -> dict[str, Any]:
    """Return the agent library."""
    return agent_library


def get_yahoo_finance_tools() -> list[dict[str, Any]]:
    """Return the Yahoo Finance tools list."""
    from fin_ai.core.tools import YAHOO_FINANCE_TOOLS
    return YAHOO_FINANCE_TOOLS


def get_rag_source_store() -> Any:
    """Return a RAGSourceStore instance."""
    from fin_ai.core.rag import RAGSourceStore, discover_vector_stores_by_source
    return RAGSourceStore(), discover_vector_stores_by_source()


def filter_sources_by_valid_types(df_sources: Any, on_disk_stores: list[str]) -> Any:
    """Filter registered RAG sources to only include those with valid source types.

    Valid source types are defined in SUPPORTED_UPLOAD_TYPES (pdf, csv, json, html, url, etc.).
    """
    if df_sources is None or df_sources.empty or not on_disk_stores:
        return df_sources
    df_on_disk = df_sources[df_sources["name"].isin(on_disk_stores)].copy()
    if df_on_disk.empty:
        return df_on_disk
    # Only keep rows where source_type is in SUPPORTED_UPLOAD_TYPES
    valid_types = set(SUPPORTED_UPLOAD_TYPES)
    df_filtered = df_on_disk[df_on_disk["source_type"].isin(valid_types)]
    return df_filtered
