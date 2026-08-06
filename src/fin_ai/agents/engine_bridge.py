"""
Engine Bridge — connects the agentic framework to processor.

Provides callable tools that agents can use to query the local RAG
infrastructure (FAISS vector stores), discover available models/providers,
access financial data, and run structured multi-source queries — all powered
by ``fin_ai.core.processor``, ``fin_ai.core.providers``, ``fin_ai.core.query``,
``fin_ai.core.request``, and ``fin_ai.core.response``.

These functions are designed for ``autogen.register_function()`` —
each accepts simple string/int arguments and returns a string.

Usage (in workflow.py)::

    from fin_ai.agents.engine_bridge import query_local_rag, get_provider_info
    register_function(query_local_rag, caller=agent, executor=proxy, ...)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fin_ai.config.fin_ai import (
    VECTOR_DB_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDINGS_PROVIDER,
    OLLAMA_BASE_URL,
    GITHUB_EMBEDDING_BASE_URL,
)
from fin_ai.core.embeddings import create_embeddings
import logging

logger = logging.getLogger(__name__)

# Core module imports (used by tools below)
from fin_ai.core.providers import list_models, ModelInfo
from fin_ai.core.query import (
    SourceRetrieverConfig,
    MultiSourceQueryResult,
    MultiSourcePromptResult,
    format_source_citations,
    route_query_to_sources,
)
from fin_ai.core.request import (
    known_providers,
    resolve_model_name,
)



# ---------------------------------------------------------------------------
# Lazy-initialised engine state
# ---------------------------------------------------------------------------


@dataclass
class _EngineState:
    """Singleton holding initialised engine resources.

    Created lazily on first access.  Call ``_ensure_initialised()`` before
    using any bridge function.
    """

    embeddings: Any = None
    source_vector_stores: dict[str, Path] = field(default_factory=dict)
    available_dbs: list[str] = field(default_factory=list)
    source_configs: list = field(default_factory=list)
    loaded_stores: dict[str, Any] = field(default_factory=dict)
    chat_provider: str = "ollama"
    initialised: bool = False
    _last_retrieval: MultiSourceQueryResult | None = None
    _last_prompt_result: MultiSourcePromptResult | None = None


_state = _EngineState()


def _ensure_initialised(
    *,
    chat_provider: str = "ollama",
    embedding_model: str | None = None,
    embedding_provider: str | None = None,
    embedding_base_url: str | None = None,
    selected_dbs: list[str] | None = None,
    github_token: str | None = None,
) -> None:
    """Initialise the engine bridge state (idempotent).

    Parameters
    ----------
    chat_provider : str
        Provider for chat/answer generation (``"ollama"``, ``"github"``,
        ``"deepseek"``, ``"proxied_github"``, ``"proxied_deepseek"``).
    embedding_model : str, optional
        Override for the embedding model name.
    embedding_provider : str, optional
        Override for the embedding provider.
    embedding_base_url : str, optional
        Override for the embedding API base URL.  Defaults to the provider's
        standard endpoint (Ollama localhost or GitHub inference).
    selected_dbs : list[str], optional
        Specific vector DB names to load.  If None, loads all available.
    github_token : str, optional
        GitHub token for embeddings authentication.  Falls back to env var.
        Not needed for ``proxied_github``.
    """
    if _state.initialised:
        return

    from fin_ai.core.processor import (
        get_source_vector_stores,
        get_vector_db_names,
        load_vector_stores_for_query,
        build_query_source_configs,
    )

    _state.chat_provider = chat_provider

    emb_model = embedding_model or DEFAULT_EMBEDDING_MODEL
    emb_provider = embedding_provider or DEFAULT_EMBEDDINGS_PROVIDER
    emb_base = embedding_base_url or (
        GITHUB_EMBEDDING_BASE_URL if emb_provider == "github" else OLLAMA_BASE_URL
    )

    logger.info(
        "Initializing engine bridge (chat_provider=%s, embedding_model=%s, embedding_provider=%s)",
        chat_provider,
        emb_model,
        emb_provider,
    )

    # GitHub token is optional for embeddings — proxied providers don't need it
    _needs_gh_token = emb_provider == "github" and chat_provider not in ("proxied_github",)
    _state.embeddings = create_embeddings(
        provider=emb_provider,
        model=emb_model,
        api_base=emb_base,
        api_key=github_token if _needs_gh_token else None,
    )

    _state.source_vector_stores = get_source_vector_stores()
    _state.available_dbs = get_vector_db_names(_state.source_vector_stores)

    dbs_to_load = selected_dbs or _state.available_dbs
    if dbs_to_load:
        _state.loaded_stores = {}
        for db_name in dbs_to_load:
            # Try to load using saved embedding metadata first (avoids dim mismatch)
            try:
                from fin_ai.core.rag import load_embedding_metadata
                meta = load_embedding_metadata(db_name)
                if meta:
                    _needs_gh_token_meta = meta["provider"] == "github" and chat_provider not in ("proxied_github",)
                    db_emb = create_embeddings(
                        provider=meta["provider"],
                        model=meta["model"],
                        api_base=meta["base_url"],
                        api_key=(
                            github_token if _needs_gh_token_meta else None
                        ),
                    )
                else:
                    db_emb = _state.embeddings

                store = load_vector_stores_for_query(
                    selected_vector_db_names=[db_name],
                    source_vector_stores=_state.source_vector_stores,
                    embeddings=db_emb,
                )
                _state.loaded_stores.update(store)
            except ValueError as exc:
                import warnings as _w
                _w.warn(f"Skipping vector store '{db_name}': {exc}")
            except Exception as exc:
                import warnings as _w
                _w.warn(f"Failed to load vector store '{db_name}': {exc}")

        # Rebuild source configs only from successfully loaded stores
        _state.source_configs = build_query_source_configs(
            _state.loaded_stores, group_by="vector_db"
        )

    _state.initialised = True


def reset_engine_state() -> None:
    """Reset the cached engine state (force re-initialisation on next call)."""
    global _state
    _state = _EngineState()


# ---------------------------------------------------------------------------
# Callable tools for autogen agents
# ---------------------------------------------------------------------------


def query_local_rag(
    query: str,
    retrieval_mode: str = "ensemble",
) -> str:
    """Query the local FAISS vector stores for financial context.

    Searches indexed documents (annual reports, research reports, earnings
    transcripts, news articles) and returns the most relevant excerpts with
    an LLM-generated answer and source citations.

    Use this tool whenever you need factual context about a company that may
    be covered in uploaded documents.  The tool automatically searches all
    available vector stores.

    Parameters
    ----------
    query : str
        A specific, focused search query.  Examples:
        - "NVIDIA revenue growth in Q4 2024"
        - "risk factors mentioned in Shaw Communications report"
        - "key competitive advantages of the company"
    retrieval_mode : str
        Retrieval strategy: ``"ensemble"`` (default, combines all sources),
        ``"separate"`` (per-source), or ``"routed"`` (auto-selects best sources).
    """
    logger.info("query_local_rag: query=%s retrieval_mode=%s", query, retrieval_mode)
    _ensure_initialised()

    if not _state.source_configs:
        return (
            "[Local RAG] No vector stores available.  Upload documents via the "
            "Streamlit dashboard first: streamlit run dashboard/financial_analyst_dashboard.py"
        )

    from fin_ai.core.processor import answer_question

    result = answer_question(
        question=query,
        source_configs=_state.source_configs,
        provider=_state.chat_provider,
        system_prompt=(
            "You are a precise financial research assistant. "
            "Answer based ONLY on the retrieved context below. "
            "If the context does not contain relevant information, say so clearly. "
            "Cite specific data points and document sections when available."
        ),
        temperature=0.1,
        retrieval_mode=retrieval_mode,
        auto_truncate_prompt=True,
    )

    response = result.get("response", "")
    elapsed = result.get("elapsed", 0)
    llm_result = result.get("llm_result")

    # Extract citations from the underlying MultiSourcePromptResult
    if llm_result and llm_result.retrieval:
        _state._last_retrieval = llm_result.retrieval
        _state._last_prompt_result = llm_result
        citations = format_source_citations(
            llm_result.retrieval, response_type="Markdown"
        )
    else:
        citations = ""

    if not response.strip():
        return "[Local RAG] No relevant context found for this query."

    parts = [f"[Retrieved in {elapsed:.1f}s | mode={retrieval_mode}]", "", response]
    if citations:
        parts.extend(["", "---", citations])
    return "\n".join(parts)


def list_available_models(provider: str = "ollama") -> str:
    """List available LLM models for a given provider.

    Uses ``fin_ai.core.providers.list_models`` to query provider APIs
    directly (Ollama /api/tags, GitHub Models catalog).

    Parameters
    ----------
    provider : str
        One of ``"ollama"``, ``"github"``, or ``"deepseek"``.
        Default: ``"ollama"``.
    """
    models: list[ModelInfo] = list_models(provider)
    if not models:
        return f"No models found for provider '{provider}'."

    lines = [f"Available models for {provider}:"]
    for m in models:
        lines.append(f"  • {m.id}  ({m.name})")
    return "\n".join(lines)


def get_provider_info() -> str:
    """Return the current provider/connection status.

    Reports which chat and embedding providers are configured, which models
    are resolved for each, and lists all registered provider configs
    with their required and optional parameters.
    """
    logger.info("get_provider_info called")
    _ensure_initialised()

    lines = [
        "=== Provider Configuration ===",
        f"Chat provider:    {_state.chat_provider}",
        f"Chat model:       {resolve_model_name(_state.chat_provider)}",
        f"Embedding model:  {DEFAULT_EMBEDDING_MODEL}",
        f"Embedding prov:   {DEFAULT_EMBEDDINGS_PROVIDER}",
        "",
        "Registered provider configs:",
    ]
    for p_name, p_cfg in known_providers().items():
        required = ", ".join(p_cfg.required_params) if p_cfg.required_params else "none"
        optional = ", ".join(p_cfg.optional_params) if p_cfg.optional_params else "none"
        lines.append(f"  • {p_name}")
        lines.append(f"      label:     {p_cfg.label}")
        lines.append(f"      required:  {required}")
        lines.append(f"      optional:  {optional}")

    lines.append("")
    lines.append("Supported response formats (ResponseFactory):")
    from fin_ai.core.response import ResponseFactory as _RF
    for fmt in _RF.available():
        lines.append(f"  • {fmt}")

    lines.append("")
    lines.append("Available providers for model listing (providers.list_models):")
    from fin_ai.core.providers import _PROVIDER_MODEL_REGISTRY
    for p_name in sorted(_PROVIDER_MODEL_REGISTRY):
        lines.append(f"  • {p_name}")

    return "\n".join(lines)


def list_vector_stores() -> str:
    """List all local vector stores with their source documents.

    Returns names, sizes, and source file paths of indexed document stores.
    """
    _ensure_initialised()

    if not _state.available_dbs:
        return (
            "No vector stores found.  Upload documents via the "
            "Streamlit dashboard: streamlit run dashboard/financial_analyst_dashboard.py"
        )

    lines = [f"Local vector stores ({len(_state.available_dbs)} available):"]
    for name in _state.available_dbs:
        path = _state.source_vector_stores.get(name)
        size = ""
        if path and path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            size = f" ({size_mb:.1f} MB)"
        lines.append(f"  • {name}{size}")
    return "\n".join(lines)


def get_financial_snapshot(symbol: str) -> str:
    """Get a comprehensive financial snapshot for a ticker symbol.

    Retrieves stock info, company profile, income statement, balance sheet,
    cash flow, and analyst recommendations — all in one call.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol, e.g. ``"AAPL"``, ``"NVDA"``.
    """
    logger.info("get_financial_snapshot: symbol=%s", symbol)
    from fin_ai.core.tools import (
        get_stock_info,
        get_company_info,
        get_income_stmt,
        get_balance_sheet,
        get_cash_flow,
        get_analyst_recommendations,
    )

    results: dict[str, Any] = {}

    try:
        results["stock_info"] = get_stock_info(symbol)
    except Exception as exc:
        results["stock_info"] = {"error": str(exc)}

    try:
        results["company"] = get_company_info(symbol)
    except Exception as exc:
        results["company"] = {"error": str(exc)}

    try:
        results["income_stmt"] = get_income_stmt(symbol)
    except Exception as exc:
        results["income_stmt"] = {"error": str(exc)}

    try:
        results["balance_sheet"] = get_balance_sheet(symbol)
    except Exception as exc:
        results["balance_sheet"] = {"error": str(exc)}

    try:
        results["cash_flow"] = get_cash_flow(symbol)
    except Exception as exc:
        results["cash_flow"] = {"error": str(exc)}

    try:
        results["analyst_recs"] = get_analyst_recommendations(symbol)
    except Exception as exc:
        results["analyst_recs"] = {"error": str(exc)}

    return json.dumps(results, indent=2, default=str)


def get_source_citations() -> str:
    """Return formatted source citations from the most recent RAG query.

    Call this after ``query_local_rag`` to see which documents and sections
    contributed to the answer.  Uses ``fin_ai.core.query.format_source_citations``.
    """
    if _state._last_retrieval is None:
        return "No RAG query has been run yet.  Call query_local_rag first."

    citations = format_source_citations(
        _state._last_retrieval, response_type="Markdown"
    )
    return citations if citations else "No citations available from last query."


def query_with_routed_rag(
    query: str,
    max_sources: int = 3,
) -> str:
    """Query the local RAG with intelligent source routing.

    Uses ``fin_ai.core.query.route_query_to_sources`` to automatically
    select the most relevant vector stores for the query, then retrieves
    and answers.  More efficient than ensemble mode when many stores are
    indexed.

    Parameters
    ----------
    query : str
        The search query.
    max_sources : int
        Maximum number of vector stores to query.  Default: 3.
    """
    _ensure_initialised()

    if not _state.source_configs:
        return "[Routed RAG] No vector stores available."

    from fin_ai.core.processor import answer_question

    # Route query to best sources
    selected = route_query_to_sources(
        query, _state.source_configs, max_sources=max_sources
    )
    selected_names = [s.name for s in selected]

    if not selected:
        return "[Routed RAG] No relevant sources found for this query."

    result = answer_question(
        question=query,
        source_configs=selected,
        provider=_state.chat_provider,
        system_prompt=(
            "You are a precise financial research assistant. "
            "Answer based ONLY on the retrieved context below."
        ),
        temperature=0.1,
        retrieval_mode="separate",
        auto_truncate_prompt=True,
    )

    response = result.get("response", "")
    elapsed = result.get("elapsed", 0)
    llm_result = result.get("llm_result")

    if llm_result and llm_result.retrieval:
        _state._last_retrieval = llm_result.retrieval
        _state._last_prompt_result = llm_result
        citations = format_source_citations(
            llm_result.retrieval, response_type="Markdown"
        )
    else:
        citations = ""

    if not response.strip():
        return "[Routed RAG] No relevant context found."

    parts = [
        f"[Routed RAG: {elapsed:.1f}s | sources: {', '.join(selected_names)}]",
        "",
        response,
    ]
    if citations:
        parts.extend(["", "---", citations])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Research publishing tools — re-exported from fin_ai.core.tools
# ---------------------------------------------------------------------------

from fin_ai.core.tools import (  # noqa: E402
    publish_research_html,
    publish_research_pdf,
    publish_research_report,
    send_research_email,
)


# ---------------------------------------------------------------------------
# Convenience: initialise with specific settings
# ---------------------------------------------------------------------------


def init_engine(
    chat_provider: str = "ollama",
    embedding_model: str | None = None,
    embedding_provider: str | None = None,
    embedding_base_url: str | None = None,
    selected_dbs: list[str] | None = None,
    github_token: str | None = None,
) -> dict[str, Any]:
    """Initialise the engine bridge and return a status summary.

    Call this once at the start of a notebook or script to pre-warm the
    engine state.  Subsequent calls to ``query_local_rag`` etc. will reuse
    the cached state.

    Returns a dict with keys ``status``, ``provider``, ``num_stores``,
    ``available_dbs``, ``num_source_configs``.
    """
    _ensure_initialised(
        chat_provider=chat_provider,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        embedding_base_url=embedding_base_url,
        selected_dbs=selected_dbs,
        github_token=github_token,
    )
    return {
        "status": "initialised",
        "provider": _state.chat_provider,
        "num_stores": len(_state.loaded_stores),
        "available_dbs": _state.available_dbs,
        "num_source_configs": len(_state.source_configs),
    }


# ---------------------------------------------------------------------------
# Agent introspection tool
# ---------------------------------------------------------------------------


def list_agent_profiles(name_filter: str = "") -> str:
    """Return a structured summary of all registered agent profiles.

    Each agent's profile includes its purpose, tools, and capabilities.
    Useful for understanding which agent to use for a given task.

    Parameters
    ----------
    name_filter : str
        Optional agent name to filter by (case-insensitive, partial match).
        If empty, returns all agents.
    """
    from .agent_library import library

    lines: list[str] = ["Registered Agents:\n"]
    for agent_name, config in library.items():
        if name_filter and name_filter.lower() not in agent_name.lower():
            continue
        lines.append(f"## {agent_name}")
        profile = config.get("profile", "")
        # Extract a short description from the profile (first ~200 chars)
        short = profile.strip()[:200].replace("\n", " ").strip()
        lines.append(f"  Role: {short}...")
        tools = config.get("tools", [])
        if tools:
            lines.append(f"  Tools ({len(tools)}): {', '.join(tools)}")
        lines.append("")

    if not name_filter and not lines:
        return "No agents registered."
    if name_filter and len(lines) == 1:
        return f"No agents found matching '{name_filter}'."

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool name → callable map (for workflow.py registration)
# ---------------------------------------------------------------------------

# Public callables exported for agent tool registration.
# These are discovered by workflow._build_tool_map() alongside core.tools.
ENGINE_BRIDGE_TOOLS: dict[str, Any] = {
    "query_local_rag": query_local_rag,
    "query_with_routed_rag": query_with_routed_rag,
    "list_available_models": list_available_models,
    "get_provider_info": get_provider_info,
    "list_vector_stores": list_vector_stores,
    "get_financial_snapshot": get_financial_snapshot,
    "get_source_citations": get_source_citations,
    "publish_research_html": publish_research_html,
    "publish_research_pdf": publish_research_pdf,
    "publish_research_report": publish_research_report,
    "send_research_email": send_research_email,
    "list_agent_profiles": list_agent_profiles,
}
