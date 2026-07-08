"""
Request Processor — handles all RAG queries and agent interactions.

This is the central processing module used by the dashboard, notebooks,
and agent framework.  It consolidates:
- RAG query execution (vector store loading, question answering)
- Agent creation and execution (single-shot agent tasks)
- Provider configuration and model resolution
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from langchain_community.vectorstores import FAISS

import openai

from fin_ai.core.request import (
    ModelRequest,
    RequestPayload,
    get_provider_config,
)
from fin_ai.core.query import (
    SourceRetrieverConfig,
    query_with_multi_source_prompting,
    build_source_retriever_configs,
)
from fin_ai.core.embeddings import create_embeddings

logger = logging.getLogger(__name__)

from fin_ai.core.rag import (
    RAGSourceStore,
    create_or_load_vector_store,
    discover_vector_stores_by_source,
    get_markdown_splits,
    load_embedding_metadata,
    load_and_convert_document,
    save_embedding_metadata,
    _create_source_metadata,
)
from fin_ai.core.providers import list_models, ModelInfo
from fin_ai.core.tools import (
    YAHOO_FINANCE_TOOLS,
    execute_litellm_tool_call,
    extract_tool_calls,
    build_tool_aware_system_prompt,
)
from fin_ai.core.history import (
    append_question_history,
    load_question_history,
    purge_vector_db_assets,
)

# Config
from fin_ai.config.fin_ai import (
    VECTOR_DB_DIR,
    OLLAMA_BASE_URL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDINGS_PROVIDER,
    GITHUB_EMBEDDING_BASE_URL,
    QUESTION_HISTORY_DIR,
    SUPPORTED_UPLOAD_TYPES,
    SUPPORTED_SOURCE_SUFFIXES,
)


# ---------------------------------------------------------------------------
# Model listing
# ---------------------------------------------------------------------------


def fetch_models(
    provider: str,
    api_key: str = "",
    base_url: str = "",
) -> list[ModelInfo]:
    """Fetch available models for a provider.  Returns empty list on error."""
    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return list_models(provider, **kwargs)


def find_source_document(vector_db_name: str) -> Path | None:
    """Locate the original source file for a vector DB."""
    for suffix in SUPPORTED_SOURCE_SUFFIXES:
        candidate = Path(VECTOR_DB_DIR) / f"{vector_db_name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def get_source_vector_stores() -> dict[str, Path]:
    """Discover vector stores and return name → path mapping."""
    return discover_vector_stores_by_source(VECTOR_DB_DIR)


def get_vector_db_names(vector_stores: dict[str, Path]) -> list[str]:
    """Return only vector DB names that have a corresponding source document."""
    return [name for name in vector_stores if find_source_document(name) is not None]


# ---------------------------------------------------------------------------
# Document indexing
# ---------------------------------------------------------------------------


def process_uploaded_document(
    file_binary: bytes,
    file_name: str,
    embedding_model: str,
    embedding_base_url: str,
    emb_provider: str,
    source_type: str,
    temp_dir: Path | str | None = None,
    github_token: str | None = None,
) -> dict[str, Any]:
    """Index an uploaded document into a FAISS vector store."""
    from hashlib import md5

    upload_hash = md5(file_binary).hexdigest()
    suffix = Path(file_name).suffix.lower()
    base_name = Path(file_name).stem

    temp_upload_dir = Path(temp_dir or Path(VECTOR_DB_DIR) / "_temp_uploads")
    temp_upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_upload_dir / f"{upload_hash}{suffix}"
    if not temp_path.exists():
        temp_path.write_bytes(file_binary)

    start_time = perf_counter()

    markdown_content = load_and_convert_document(temp_path)
    document_metadata = _create_source_metadata(temp_path, source_type)
    document_metadata["source"] = file_name
    document_metadata["filename"] = file_name
    document_metadata["source_type"] = source_type
    document_metadata["file_size"] = len(file_binary)
    chunks = get_markdown_splits(markdown_content, metadata=document_metadata)

    embeddings = create_embeddings(
        provider=emb_provider,
        model=embedding_model,
        api_base=embedding_base_url,
        api_key=github_token if emb_provider == "github" else None,
    )
    vector_store = create_or_load_vector_store(base_name, chunks, embeddings)
    save_embedding_metadata(base_name, provider=emb_provider, model=embedding_model, base_url=embedding_base_url)

    vector_db_path = Path(VECTOR_DB_DIR) / f"{base_name}.faiss"
    vector_store.save_local(str(vector_db_path))

    source_path = Path(VECTOR_DB_DIR) / file_name
    source_path.write_bytes(file_binary)

    # Record in the RAGSourceStore
    rag_store = RAGSourceStore()
    rag_store.add_from_metadata(
        name=base_name,
        source_type=source_type,
        filename=file_name,
        file_size=len(file_binary),
        chunk_count=len(chunks),
        embedding_provider=emb_provider,
        embedding_model=embedding_model,
        embedding_base_url=embedding_base_url,
    )

    temp_path.unlink(missing_ok=True)

    return {
        "vector_db_name": base_name,
        "elapsed": perf_counter() - start_time,
        "doc_metadata": document_metadata,
        "rag_source": base_name,
    }


# ---------------------------------------------------------------------------
# Vector store loading
# ---------------------------------------------------------------------------


def load_vector_stores_for_query(
    selected_vector_db_names: list[str],
    source_vector_stores: dict[str, Path],
    embeddings: Any,
) -> dict[str, FAISS]:
    """Load FAISS vector stores with dimension sanity checks."""
    import faiss

    loaded: dict[str, FAISS] = {}

    # Probe embedding dimension once — shared across all stores
    _dimension: int | None = None

    for name in selected_vector_db_names:
        if name not in source_vector_stores:
            continue
        vs_path = source_vector_stores[name]
        if not vs_path.exists():
            continue
        faiss_load_dir = str(vs_path.parent) if vs_path.is_file() else str(vs_path)

        vs = FAISS.load_local(faiss_load_dir, embeddings=embeddings, allow_dangerous_deserialization=True)

        if _dimension is None:
            probe = embeddings.embed_query("dimension probe")
            _dimension = len(probe)

        if _dimension != vs.index.d:
            raise ValueError(
                f"Embedding dimension mismatch for vector store '{name}': "
                f"index={vs.index.d}, embedding={_dimension}"
            )

        loaded[name] = vs

    return loaded


def build_query_source_configs(
    loaded_stores: dict[str, FAISS],
    group_by: str = "vector_db",
) -> list[SourceRetrieverConfig]:
    """Build source retriever configs from loaded vector stores."""
    configs: list[SourceRetrieverConfig] = []
    for name, store in loaded_stores.items():
        configs.extend(
            build_source_retriever_configs(store, base_name=name, group_by=group_by, search_k=5)
        )
    return configs


def discover_source_groups(
    loaded_stores: dict[str, FAISS] | None = None,
) -> dict[str, list[str]]:
    """Discover source groups and the documents within each group.

    A *source group* is a ``source_type`` value (e.g. ``"pdf"``, ``"csv"``,
    ``"json"``, ``"html"``, ``"url"``).  Each group maps to the list of
    vector-store names whose documents share that ``source_type``.

    Uses the ``RAGSourceStore`` registry for fast lookup.  When *loaded_stores*
    is provided it is used as a fallback for any names not found in the
    registry, then the registry is updated.

    Returns a dict like ``{"pdf": ["NVDA_report", "Ball Report"],
    "csv": ["credit_card_transactions_sample"]}``.
    """
    rag_store = RAGSourceStore()
    groups = dict(rag_store.get_groups())

    # Back-fill from loaded stores for any names not yet in the registry
    if loaded_stores:
        for store_name in loaded_stores:
            if any(store_name in names for names in groups.values()):
                continue
            # Peek at the first document's metadata as fallback
            try:
                store = loaded_stores[store_name]
                if hasattr(store, "index_to_docstore_id"):
                    doc_ids = list(store.index_to_docstore_id.values())
                elif hasattr(store, "docstore") and hasattr(store.docstore, "_dict"):
                    doc_ids = list(store.docstore._dict.keys())
                else:
                    doc_ids = []
                first_id = doc_ids[0] if doc_ids else None
                source_type = "unknown"
                if first_id:
                    doc = store.docstore.search(first_id)
                    if doc and hasattr(doc, "metadata"):
                        source_type = doc.metadata.get("source_type", "unknown")
                groups.setdefault(source_type, []).append(store_name)
            except Exception:
                groups.setdefault("unknown", []).append(store_name)

    # Exclude fallback groups that don't represent valid source types
    unknown_groups = {"unknown"}
    return {k: v for k, v in groups.items() if k not in unknown_groups}


def filter_stores_by_source_groups(
    loaded_stores: dict[str, FAISS],
    selected_groups: list[str],
) -> dict[str, FAISS]:
    """Filter *loaded_stores* to only those belonging to the selected source groups."""
    groups = discover_source_groups(loaded_stores)
    allowed_names: set[str] = set()
    for group in selected_groups:
        allowed_names.update(groups.get(group, []))
    return {name: store for name, store in loaded_stores.items() if name in allowed_names}


# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------


def answer_question(
    question: str,
    source_configs: Sequence[SourceRetrieverConfig],
    *,
    provider: str,
    system_prompt: str = "You are a concise financial analysis assistant.",
    temperature: float = 0.2,
    retrieval_mode: str = "ensemble",
    auto_truncate_prompt: bool = True,
    use_tools: bool = False,
    response_format: str = "text",
) -> dict[str, Any]:
    """Run a RAG query against the selected vector stores."""
    start_time = perf_counter()

    effective_system = build_tool_aware_system_prompt(system_prompt) if use_tools else system_prompt

    llm_result = query_with_multi_source_prompting(
        question,
        source_configs,
        provider=provider,
        response_format=response_format,
        mode=retrieval_mode,
        system_prompt=effective_system,
        temperature=temperature,
        auto_truncate_prompt=auto_truncate_prompt,
        tools=YAHOO_FINANCE_TOOLS if use_tools else None,
    )

    llm_response = llm_result.response

    # Diagnostic: flag silent LLM failures
    if llm_response is None:
        logger.warning(
            "LLM returned no response for query — retrieval had %d sources, mode=%s",
            len(llm_result.source_results),
            llm_result.retrieval.mode,
        )

    if use_tools and llm_response is not None:
        first_message = llm_response.get_metadata().raw_response.choices[0].message
        tool_calls = extract_tool_calls(first_message)
        if tool_calls:
            follow_up_messages = _build_tool_follow_up(effective_system, llm_result.prompt, first_message, tool_calls)
            follow_up_payload = RequestPayload(
                prompt=question,
                system_prompt=effective_system,
                temperature=temperature,
                auto_truncate_prompt=auto_truncate_prompt,
                tools=YAHOO_FINANCE_TOOLS,
                messages=follow_up_messages,
            )
            requester = ModelRequest(provider=provider, format="text")
            llm_response = requester.client.send(follow_up_payload, response_class=requester.response_class)

    content = llm_response.content if llm_response else ""
    metadata = llm_response.get_metadata() if llm_response else None

    return {
        "response": content,
        "metadata": metadata,
        "llm_result": llm_result,
        "elapsed": perf_counter() - start_time,
    }


# ---------------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------------


def build_agent_llm_config(
    provider: str,
    model: str,
    *,
    ollama_base_url: str = "",
    github_endpoint: str = "",
    github_token: str = "",
    deepseek_base_url: str = "",
    deepseek_token: str = "",
) -> dict[str, Any]:
    """Build an AutoGen-compatible llm_config from provider settings.

    Parameters
    ----------
    provider : str
        ``"ollama"``, ``"github"``, ``"deepseek"``, ``"proxied_github"``,
        or ``"proxied_deepseek"``.
    model : str
        Model identifier (e.g. ``"llama3.1"``, ``"openai/gpt-4o"``).
    """
    cfg = get_provider_config(provider)

    if provider == "github":
        return {
            "config_list": [
                {"model": model, "base_url": github_endpoint, "api_key": github_token}
            ],
            "temperature": 0,
            "timeout": 120,
        }
    elif provider == "proxied_github":
        return {
            "config_list": [
                {"model": model, "base_url": github_endpoint, "api_key": ""}
            ],
            "temperature": 0,
            "timeout": 120,
        }
    elif provider in ("deepseek", "proxied_deepseek"):
        return {
            "config_list": [
                {"model": model, "base_url": deepseek_base_url, "api_key": deepseek_token}
            ],
            "temperature": 0,
            "timeout": 120,
        }
    else:
        base = ollama_base_url or OLLAMA_BASE_URL
        base = base.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        return {
            "config_list": [
                {"model": model, "base_url": base, "api_key": "ollama"}
            ],
            "temperature": 0,
            "timeout": 120,
        }


def run_agent_task(
    agent_name: str,
    prompt: str,
    llm_config: dict[str, Any],
    *,
    embedding_model: str = "",
    embedding_provider: str = "",
    embedding_base_url: str = "",
    chat_provider: str = "ollama",
    is_publisher: bool = False,
    publisher_format: str = "html",
    publisher_email: str = "",
) -> dict[str, Any]:
    """Run a single agent task and return the response.

    This is the primary entry point for programmatic agent interaction
    from the dashboard, notebooks, or any script.

    Parameters
    ----------
    agent_name : str
        Agent profile name (e.g. ``"Data_Analyst"``, ``"Research_Publisher"``).
    prompt : str
        Task prompt to send to the agent.
    llm_config : dict
        AutoGen-compatible LLM config (use ``build_agent_llm_config`` to build).
    embedding_model : str
        Embedding model for initialising the engine bridge.
    embedding_provider : str
        Embedding provider (``"ollama"`` or ``"github"``).
    embedding_base_url : str
        Embedding API base URL.
    chat_provider : str
        Chat provider for RAG queries.
    is_publisher : bool
        If True, uses ``SingleAssistant`` (no RAG); otherwise ``SingleAssistantRAG``.
    publisher_format : str
        Report format for Research_Publisher (``"html"`` or ``"pdf"``).
    publisher_email : str
        Optional email address to send the report to.

    Returns
    -------
    dict with keys ``response``, ``agent_name``, ``success``, ``error`` (if failed).
    """
    from fin_ai.agents import SingleAssistantRAG, SingleAssistant, init_engine
    from fin_ai.agents.engine_bridge import publish_research_report

    _github_token_for_bridge = (
        os.environ.get("GITHUB_TOKEN") or ""
        if chat_provider in ("github",)
        else None
    )

    # Initialise the engine bridge for local RAG
    init_engine(
        chat_provider=chat_provider,
        embedding_model=embedding_model or None,
        embedding_provider=embedding_provider or None,
        embedding_base_url=embedding_base_url or None,
        github_token=_github_token_for_bridge,
    )

    _retrieve_config = {
        "task": "qa",
        "vector_db": None,
        "docs_path": [],
        "chunk_token_size": 1000,
        "get_or_create": False,
        "collection_name": "processor_agent_rag",
        "must_break_at_empty_line": False,
        "customized_prompt": (
            "Context from local stores:\n{input_context}\n\n"
            "Query: {input_question}"
        ),
    }

    try:
        if is_publisher:
            agent = SingleAssistant(
                agent_name,
                llm_config=llm_config,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=8,
                code_execution_config=False,
            )
            # Append publishing instruction
            full_prompt = prompt.strip()
            if publisher_email:
                full_prompt += (
                    f"\n\nAfter analysis, publish as {publisher_format} and email to {publisher_email}."
                )
            else:
                full_prompt += f"\n\nAfter analysis, publish as {publisher_format}."
        else:
            agent = SingleAssistantRAG(
                agent_name,
                llm_config=llm_config,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=8,
                code_execution_config=False,
                retrieve_config=_retrieve_config,
                rag_description="Query local FAISS vector stores for financial context.",
            )
            full_prompt = prompt

        agent.chat(full_prompt)

        # Extract last message
        history = agent.user_proxy.chat_messages
        response = ""
        if history:
            last_agent = list(history.keys())[-1]
            msgs = history[last_agent]
            if msgs:
                response = msgs[-1].get("content", "")

        pub_result = None
        if is_publisher:
            try:
                pub_result = publish_research_report(
                    content=response or prompt,
                    title=f"{agent_name} Report",
                    format=publisher_format,
                    email=publisher_email,
                )
            except Exception:
                pass

        return {
            "response": response,
            "agent_name": agent_name,
            "success": True,
            "publication": pub_result,
        }

    except openai.APIConnectionError as exc:
        logger.exception("API connection error for agent '%s'", agent_name)
        return {
            "response": "",
            "agent_name": agent_name,
            "success": False,
            "error": (
                f"Cannot connect to the LLM provider. "
                f"Check that your model endpoint is running and reachable. "
                f"Details: {exc}"
            ),
        }
    except Exception as exc:
        logger.exception("Agent task '%s' failed", agent_name)
        return {
            "response": "",
            "agent_name": agent_name,
            "success": False,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Question history
# ---------------------------------------------------------------------------


def load_history(vector_db_name: str) -> list[dict[str, Any]]:
    return load_question_history(vector_db_name, QUESTION_HISTORY_DIR)


def save_history_entry(vector_db_name: str, entry: dict[str, Any]) -> None:
    append_question_history(vector_db_name, QUESTION_HISTORY_DIR, entry)


def clear_history(vector_db_name: str) -> None:
    from fin_ai.core.history import clear_question_history
    clear_question_history(vector_db_name, QUESTION_HISTORY_DIR)


def purge_vector_db(vector_db_name: str) -> list[Path]:
    deleted = purge_vector_db_assets(vector_db_name, Path(VECTOR_DB_DIR), QUESTION_HISTORY_DIR)
    # Also remove from the RAGSourceStore registry
    rag_store = RAGSourceStore()
    rag_store.remove(vector_db_name)
    return deleted


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------





def _build_tool_follow_up(
    system: str,
    prompt: str,
    first_message: Any,
    tool_calls: list[dict],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": first_message.content or "",
            "tool_calls": [
                {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments_text"]}}
                for tc in tool_calls
            ],
        },
    ]
    for tc in tool_calls:
        tool_result = execute_litellm_tool_call(tc["name"], tc["arguments"])
        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "name": tc["name"] or "unknown_tool",
            "content": json.dumps(tool_result),
        })
    return messages
