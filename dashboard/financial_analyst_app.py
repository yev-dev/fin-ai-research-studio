"""
Financial Data Analysis — application / business logic layer.

All non-UI operations live here so that ``financial_analyst_dashboard.py``
can focus purely on Streamlit rendering and event wiring.

Responsibilities
----------------
- Provider & credential resolution
- Model listing & selection
- Embedding configuration
- Vector store discovery / filtering / normalisation
- RAG query dispatch & history management
- Agent execution & publishing
- Database maintenance (purge, sync, upload)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
import logging
from functools import wraps
import sys

# Ensure the project's `src/` directory is on sys.path so `import fin_ai` works
# when running Streamlit from the repository root (common dev setup).
try:
    import fin_ai  # type: ignore
except Exception:
    try:
        _repo_root = Path(__file__).resolve().parents[1]
        _src_dir = _repo_root / "src"
        if _src_dir.exists():
            sys.path.insert(0, str(_src_dir))
        else:
            sys.path.insert(0, str(_repo_root))
    except Exception:
        pass

from fin_ai.config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDINGS_PROVIDER,
    DEFAULT_GITHUB_MODEL,
    DEFAULT_GITHUB_EMBEDDING_MODEL,
    DEFAULT_DEEPSEEK_MODEL,
    OLLAMA_BASE_URL,
    GITHUB_EMBEDDING_BASE_URL,
    DEEPSEEK_BASE_URL,
    VECTOR_DB_DIR,
)


# ---------------------------------------------------------------------------
# Startup environment fixes
# ---------------------------------------------------------------------------
# Ensure matplotlib has a writable configuration/cache dir to avoid repeated
# font-cache warnings on macOS and CI environments. Prefer a workspace-scoped
# directory under `VECTOR_DB_DIR` when available, otherwise fall back to /tmp.
try:
    _mpl_dir = os.environ.get("MPLCONFIGDIR")
    if not _mpl_dir:
        try:
            _mpl_dir = os.path.join(VECTOR_DB_DIR or ".", "matplotlib")
        except Exception:
            _mpl_dir = "/tmp/matplotlib"
        try:
            os.makedirs(_mpl_dir, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = _mpl_dir
        except Exception:
            # best-effort; continue even if we can't create the directory
            pass
except Exception:
    pass

# Reduce overly-verbose font_manager/info logs from matplotlib during import.
try:
    import logging as _logging

    _logging.getLogger("matplotlib.font_manager").setLevel(_logging.WARNING)
except Exception:
    pass


def _check_socks_dependency_for_litellm() -> None:
    """Log a helpful hint if a SOCKS proxy is configured but the socks
    dependency is missing. This avoids repeated runtime confusion from
    LiteLLM warnings about missing 'socksio'."""
    try:
        proxy = os.environ.get("ALL_PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
        if proxy and "socks" in proxy.lower():
            try:
                import socksio  # type: ignore

            except Exception:
                logging.getLogger("finai.app").warning(
                    "Detected a SOCKS proxy in the environment but 'socksio' is not importable. "
                    "Install with: pip install httpx[socks] to avoid LiteLLM fetch warnings.")
    except Exception:
        # never fail app startup for this check
        pass


# Run a quick, non-fatal dependency check at import time.
_check_socks_dependency_for_litellm()

# ---------------------------------------------------------------------------
# System-level defaults (re-exported for dashboard convenience)
# ---------------------------------------------------------------------------

from dashboard import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDINGS_PROVIDER,
    DEFAULT_GITHUB_MODEL,
    DEFAULT_GITHUB_EMBEDDING_MODEL,
    DEFAULT_DEEPSEEK_MODEL,
    OLLAMA_BASE_URL,
    GITHUB_EMBEDDING_BASE_URL,
    DEEPSEEK_BASE_URL,
    VECTOR_DB_DIR,
)

# ---------------------------------------------------------------------------
# Core module imports
# ---------------------------------------------------------------------------

from fin_ai.core.embeddings import create_embeddings
from fin_ai.core.processor import (
    SUPPORTED_UPLOAD_TYPES,
    answer_question,
    build_query_source_configs,
    clear_history,
    discover_source_groups,
    fetch_models,
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
from fin_ai.core.rag import RAGSourceStore, discover_vector_stores_by_source, load_embedding_metadata
from fin_ai.core.request import known_providers, get_provider_config
from fin_ai.agents.agent_library import library as agent_library
from fin_ai.agents.prompts_library import RESEARCH_ANALYSIS
from fin_ai.core.tools import publish_research_pdf, publish_research_html, publish_research_report

# ---------------------------------------------------------------------------
# Known source group types
# ---------------------------------------------------------------------------

KNOWN_SOURCE_GROUPS = {"pdf", "csv", "json", "html", "url"}

# ---------------------------------------------------------------------------
# 1.  Provider / credential resolution
# ---------------------------------------------------------------------------


def resolve_provider_label_map() -> tuple[dict[str, str], str, int, dict]:
    """Build the provider label→key mapping and pick the default.

    Returns
    -------
    (label_to_key, default_provider, default_index, all_configs)
    """
    all_cfgs = known_providers()
    label_to_key: dict[str, str] = {cfg.label: name for name, cfg in all_cfgs.items()}
    default_provider = os.getenv("DEFAULT_PROVIDER", "ollama").strip().lower()
    default_index = next(
        (i for i, k in enumerate(label_to_key) if label_to_key[k] == default_provider),
        0,
    )
    return label_to_key, default_provider, default_index, all_cfgs


def load_credentials(provider_key: str, api_key_value: str = "", api_base_value: str = "") -> dict[str, Any]:
    """Load env credentials and build the provider's connection dict.

    Parameters
    ----------
    provider_key : str
        Internal provider key (e.g. ``"ollama"``, ``"github"``).
    api_key_value, api_base_value : str
        Values typed by the user in the UI; override env defaults when set.

    Returns keys: ``provider_config``, ``api_key``, ``api_base``,
    ``proxy_port``, ``http_proxy_port``, ``https_proxy_port``.
    """
    pcfg = get_provider_config(provider_key)
    creds: dict[str, Any] = {
        "provider_config": pcfg,
        "api_key": "",
        "api_base": "",
        "proxy_port": None,
        "http_proxy_port": None,
        "https_proxy_port": None,
    }

    if "api_key" in pcfg.required_params or "api_key" in pcfg.optional_params:
        env_map = {"github": "GITHUB_TOKEN", "deepseek": "DEEPSEEK_TOKEN"}
        creds["api_key"] = api_key_value or os.getenv(env_map.get(provider_key, ""), "")

    if "api_base" in pcfg.optional_params:
        default_base = pcfg.default_base_url
        if provider_key in ("github", "proxied_github"):
            creds["api_base"] = api_base_value or os.getenv("GITHUB_ENDPOINT", default_base)
        elif provider_key in ("deepseek", "proxied_deepseek"):
            creds["api_base"] = api_base_value or os.getenv("DEEPSEEK_BASE_URL", default_base)
        else:
            creds["api_base"] = api_base_value or os.getenv("OLLAMA_ENDPOINT", default_base)

    return creds


def _parse_int_or_none(value: str | None) -> int | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def resolve_chat_base_url(provider_key: str, gh_endpoint: str, ds_base_url: str, ollama_base: str) -> str:
    return (
        gh_endpoint
        if provider_key in ("github", "proxied_github")
        else ds_base_url
        if provider_key in ("deepseek", "proxied_deepseek")
        else ollama_base
    )


def format_agent_prompt(template_key: str, asset: str | None = None) -> str:
    """Return a formatted prompt string for the given template key and asset.

    Falls back to the raw template key string if missing.
    """
    if template_key == "Custom":
        return ""
    tpl = RESEARCH_ANALYSIS.get(template_key)
    if tpl:
        try:
            return tpl.format(asset=(asset or "").strip() or "NVDA")
        except Exception:
            return tpl
    return str(template_key or "")


# ---------------------------------------------------------------------------
# 2.  Model listing (chat & embedding)
# ---------------------------------------------------------------------------


def fetch_chat_models(provider_key: str, api_key: str = "", api_base: str = "") -> list[str]:
    """Fetch chat models for the given provider; returns empty list on error."""
    listing_provider = provider_key.removeprefix("proxied_")
    kwargs: dict[str, str] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["base_url"] = api_base
    try:
        models = fetch_models(listing_provider, **kwargs)
        return [m.id for m in models if getattr(m, "id", "")]
    except Exception:
        return []


def _looks_like_embedding_model(model_id: str) -> bool:
    mid = (model_id or "").strip().lower()
    return "embedding" in mid or "embed" in mid


def fetch_embedding_models(provider_key: str, api_key: str, api_base: str) -> list[str]:
    """Fetch embedding-capable models for the provider."""
    listing_provider = provider_key.removeprefix("proxied_")
    kwargs: dict[str, str] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["base_url"] = api_base
    try:
        models = fetch_models(listing_provider, **kwargs)
        return [m.id for m in models if _looks_like_embedding_model(getattr(m, "id", ""))]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 3.  Embedding configuration
# ---------------------------------------------------------------------------


def resolve_embedding_provider_map() -> tuple[dict[str, str], str, int]:
    """Return (label_to_key, default_provider, default_index)."""
    label_to_key = {"Local Ollama": "ollama", "GitHub Models": "github"}
    default_provider = os.getenv("DEFAULT_EMBEDDINGS_PROVIDER", DEFAULT_EMBEDDINGS_PROVIDER).strip().lower()
    labels = list(label_to_key.keys())
    default_index = next(
        (i for i, k in enumerate(labels) if label_to_key[k] == default_provider), 0,
    )
    return label_to_key, default_provider, default_index


def create_embedding_instance(
    provider: str, model: str, api_base: str, api_key: str | None = None,
) -> Any:
    """Create an embedding instance.  Returns None on failure."""
    try:
        return create_embeddings(provider=provider, model=model, api_base=api_base, api_key=api_key)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 4.  Vector store discovery & source group normalisation
# ---------------------------------------------------------------------------


def normalise_source_groups(group_map: dict[str, list[str]]) -> dict[str, list[str]]:
    """Normalise source-group names to lower-case known types."""
    normalised: dict[str, list[str]] = {}
    for group, names in group_map.items():
        key = str(group).strip().lower()
        if key not in KNOWN_SOURCE_GROUPS:
            continue
        normalised.setdefault(key, []).extend(str(name).removesuffix(".faiss") for name in names)
    return {group: sorted(set(names)) for group, names in normalised.items()}


def resolve_source_environment(
    embedding_provider: str,
    embedding_model: str,
    embedding_base_url: str,
    github_token: str = "",
) -> dict[str, Any]:
    """Discover vector stores, build source groups, return an env dict.

    Returned keys: ``source_vector_stores``, ``vector_db_names``,
    ``source_groups_map_norm``, ``available_source_groups``,
    ``available_source_names``, ``embeddings``.
    """
    env: dict[str, Any] = {
        "source_vector_stores": {},
        "vector_db_names": [],
        "source_groups_map_norm": {},
        "available_source_groups": [],
        "available_source_names": [],
        "embeddings": None,
    }
    env["source_vector_stores"] = get_source_vector_stores()
    env["vector_db_names"] = get_vector_db_names(env["source_vector_stores"])
    if not env["vector_db_names"]:
        return env

    env["embeddings"] = create_embedding_instance(
        embedding_provider, embedding_model, embedding_base_url,
        github_token if embedding_provider == "github" else None,
    )
    if env["embeddings"] is None:
        return env

    try:
        group_map = discover_source_groups()
        env["source_groups_map_norm"] = normalise_source_groups(group_map)
        env["available_source_groups"] = sorted(env["source_groups_map_norm"].keys())
        available_names: set[str] = set()
        for names in env["source_groups_map_norm"].values():
            available_names.update(n for n in names if n in env["vector_db_names"])
        env["available_source_names"] = sorted(available_names) or list(env["vector_db_names"])
    except (ValueError, RuntimeError):
        pass

    return env


def build_source_configs_for_query(
    selected_dbs: list[str],
    source_vector_stores: dict[str, Path],
    embeddings: Any,
) -> list[Any]:
    """Load selected vector stores and build source configs for querying."""
    if not selected_dbs or not embeddings:
        return []
    loaded = load_vector_stores_for_query(selected_dbs, source_vector_stores, embeddings)
    return build_query_source_configs(loaded, group_by="vector_db")
# ---------------------------------------------------------------------------
# 5.  RAG query execution
# ---------------------------------------------------------------------------


def execute_rag_query(
    question: str,
    source_configs: list[Any],
    *,
    provider: str,
    model: str,
    api_base: str,
    api_key: str | None = None,
    proxy_port: int | None = None,
    http_proxy_port: int | None = None,
    https_proxy_port: int | None = None,
    system_prompt: str = "You are a concise financial analysis assistant.",
    temperature: float = 0.2,
    retrieval_mode: str = "ensemble",
    auto_truncate_prompt: bool = True,
    use_tools: bool = False,
) -> dict[str, Any]:
    """Execute a full RAG query and return the merged result dict."""
    return answer_question(
        question, source_configs,
        provider=provider, model=model, api_base=api_base, api_key=api_key,
        proxy_port=proxy_port, http_proxy_port=http_proxy_port, https_proxy_port=https_proxy_port,
        system_prompt=system_prompt, temperature=temperature,
        retrieval_mode=retrieval_mode, auto_truncate_prompt=auto_truncate_prompt,
        use_tools=use_tools,
    )


def manage_history(
    db_name: str,
    action: str = "load",
    entry: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Load / save / clear history entries for a vector DB.

    Actions: ``"load"`` (default), ``"save"``, ``"clear"``.
    """
    if action == "clear":
        clear_history(db_name)
        return []
    if action == "save" and entry is not None:
        save_history_entry(db_name, entry)
    return load_history(db_name)


# ---------------------------------------------------------------------------
# 6.  Agent execution & publishing
# ---------------------------------------------------------------------------


def build_llm_config_for_agent(
    provider: str,
    model: str,
    ollama_base_url: str = "",
    github_endpoint: str = "",
    github_token: str = "",
    deepseek_base_url: str = "",
    deepseek_token: str = "",
) -> dict[str, Any]:
    """Build an AutoGen LLM config dict for agent execution."""
    return build_agent_llm_config(
        provider=provider, model=model,
        ollama_base_url=ollama_base_url,
        github_endpoint=github_endpoint, github_token=github_token,
        deepseek_base_url=deepseek_base_url, deepseek_token=deepseek_token,
    )


def execute_agent_run(
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
    publisher_title: str = "",
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Run an agent task and return the result dict."""
    return run_agent_task(
        agent_name=agent_name, prompt=prompt, llm_config=llm_config,
        embedding_model=embedding_model, embedding_provider=embedding_provider,
        embedding_base_url=embedding_base_url, chat_provider=chat_provider,
        is_publisher=is_publisher, publisher_format=publisher_format,
        publisher_email=publisher_email, publisher_title=publisher_title,
        progress_callback=progress_callback,
    )


def publish_response(
    response_text: str,
    title: str,
    format: str = "html",
    email: str = "",
) -> str:
    """Publish an agent's response in the requested format.  Returns JSON."""
    cleaned = (response_text or "").strip()
    if cleaned.endswith("TERMINATE"):
        cleaned = cleaned[:-9].rstrip()
    return publish_research_report(content=cleaned, title=title, format=format, email=email)


def extract_publication_filepath(publication_str: str | None) -> str | None:
    """Extract the ``filepath`` from a publication JSON string."""
    if not publication_str:
        return None
    try:
        data = json.loads(publication_str) if isinstance(publication_str, str) else publication_str
        return data.get("publish", data).get("filepath")
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
# ---------------------------------------------------------------------------
# 7.  Database maintenance
# ---------------------------------------------------------------------------


def purge_vector_database(db_name: str) -> bool:
    """Permanently delete a vector database.  Returns True on success."""
    return purge_vector_db(db_name)


def sync_rag_source_store() -> int:
    """Sync RAG source store from disk.  Returns number of new sources added."""
    return RAGSourceStore().sync_from_disk()


def get_source_registry():
    """Return (dataframe_of_sources, on_disk_store_names)."""
    rag_store = RAGSourceStore()
    on_disk = discover_vector_stores_by_source()
    return rag_store.to_dataframe(), on_disk


def upload_and_process_document(
    file_binary: bytes,
    file_name: str,
    embedding_model: str,
    embedding_base_url: str,
    emb_provider: str,
    source_type: str,
    github_token: str | None = None,
) -> dict[str, Any]:
    """Upload a document and process it into the vector store."""
    return process_uploaded_document(
        file_binary=file_binary, file_name=file_name,
        embedding_model=embedding_model, embedding_base_url=embedding_base_url,
        emb_provider=emb_provider, source_type=source_type, github_token=github_token,
    )


# ---------------------------------------------------------------------------
# 8.  Utility helpers
# ---------------------------------------------------------------------------


def preview_filename(title: str, agent_name: str, ext: str) -> str:
    """Generate a safe preview filename for a report."""
    safe = "".join(c for c in (title or "") if c.isalnum() or c in (" ", "-", "_")).rstrip()
    prefix = (safe[:80] or "research_report").replace(" ", "_")
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{ext or 'html'}"


def known_embedding_providers() -> dict[str, str]:
    """Map user-facing embedding labels to provider keys."""
    return {"Local Ollama": "ollama", "GitHub Models": "github"}


# ---------------------------------------------------------------------------
# Re-export for dashboard convenience
# ---------------------------------------------------------------------------

__all__ = [
    # Provider
    "resolve_provider_label_map", "load_credentials", "_parse_int_or_none",
    "resolve_chat_base_url", "fetch_chat_models", "fetch_embedding_models",
    # Embeddings
    "resolve_embedding_provider_map", "create_embedding_instance",
    # Sources
    "normalise_source_groups", "resolve_source_environment",
    "build_source_configs_for_query",
    # RAG
    "execute_rag_query", "manage_history",
    # Agent
    "build_llm_config_for_agent", "execute_agent_run",
    "publish_response", "extract_publication_filepath",
    # Maintenance
    "purge_vector_database", "sync_rag_source_store",
    "get_source_registry", "upload_and_process_document",
    # Utils
    "preview_filename", "known_embedding_providers",
    "KNOWN_SOURCE_GROUPS",
    "DEFAULT_CHAT_MODEL", "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDINGS_PROVIDER",
    "DEFAULT_GITHUB_MODEL", "DEFAULT_GITHUB_EMBEDDING_MODEL",
    "DEFAULT_DEEPSEEK_MODEL",
    "OLLAMA_BASE_URL", "GITHUB_EMBEDDING_BASE_URL",
    "DEEPSEEK_BASE_URL", "VECTOR_DB_DIR",
    "agent_library", "RESEARCH_ANALYSIS",
]

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
LOG_DIR = Path(VECTOR_DB_DIR) if "VECTOR_DB_DIR" in globals() else Path("./logs")
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

LOG_FILE = LOG_DIR / "finai_app.log"
LOG_FILE_LEGACY = LOG_DIR / "fin_ai.log"
logger = logging.getLogger("finai.app")
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    # Primary file handler (new name)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # Legacy file handler for backwards compatibility
    try:
        fh2 = logging.FileHandler(LOG_FILE_LEGACY, encoding="utf-8")
        fh2.setFormatter(formatter)
        logger.addHandler(fh2)
    except Exception:
        # best-effort: ignore if legacy file can't be created
        pass
    # Also add console handler for interactive runs
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _enable_immediate_flush(logger_obj: logging.Logger) -> None:
    """Ensure all file/stream handlers flush after each emit to avoid
    buffering delays that can make recent entries invisible when a
    process crashes or Streamlit swallows exceptions.
    """
    try:
        for h in list(logger_obj.handlers):
            try:
                orig_emit = h.emit

                def _emit_and_flush(record, _orig_emit=orig_emit, _h=h):
                    _orig_emit(record)
                    try:
                        _h.flush()
                    except Exception:
                        pass

                h.emit = _emit_and_flush  # type: ignore
            except Exception:
                # non-fatal: skip handlers we can't wrap
                pass
    except Exception:
        pass


# Enable immediate flush for the configured logger to improve crash
# diagnostics visibility.
try:
    _enable_immediate_flush(logger)
except Exception:
    pass


# Global exception hook to capture any uncaught exceptions and persist a
# short traceback to `last_exception.log` under the same log directory.
def _global_excepthook(exc_type, exc_value, exc_tb):
    try:
        import traceback as _tb

        trace = _tb.format_exception(exc_type, exc_value, exc_tb)
        logger = logging.getLogger("finai.app")
        logger.error("Uncaught exception:\n%s", "".join(trace))
        try:
            last_file = LOG_DIR / "last_exception.log"
            with last_file.open("w", encoding="utf-8") as fh:
                fh.write("".join(trace))
        except Exception:
            pass
    except Exception:
        # Never let the exception hook itself raise
        pass


sys.excepthook = _global_excepthook


def _safefmt_args(args: tuple, kwargs: dict) -> str:
    # Simple safe formatting: redact common secret keys
    redacted = {}
    for k, v in kwargs.items():
        if k.lower() in ("api_key", "apikey", "token", "github_token", "deepseek_token"):
            redacted[k] = "<REDACTED>"
        else:
            try:
                redacted[k] = str(v)
            except Exception:
                redacted[k] = "<UNSERIALIZABLE>"
    arg_parts = [repr(a) for a in args]
    kw_parts = [f"{k}={v}" for k, v in redacted.items()]
    return ", ".join(arg_parts + kw_parts)


def log_call(level: str = "info"):
    def _decorator(fn):
        @wraps(fn)
        def _wrapped(*args, **kwargs):
            try:
                logger_method = getattr(logger, level, logger.info)
                logger_method(f"CALL {fn.__name__}({ _safefmt_args(args, kwargs) })")
            except Exception:
                pass
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                try:
                    logger.exception(f"EXCEPTION in {fn.__name__}: {e}")
                except Exception:
                    pass
                raise
            try:
                # Log a short result summary
                if isinstance(result, (str, int, float, bool)):
                    logger.info(f"RETURN {fn.__name__}: {result}")
                else:
                    logger.info(f"RETURN {fn.__name__}: <{type(result).__name__}>")
            except Exception:
                pass
            return result

        return _wrapped

    return _decorator


def log_ui_event(message: str, level: str = "info") -> None:
    """Log a UI-related event from the dashboard.

    Example: log_ui_event("dashboard.rendered")
    """
    try:
        method = getattr(logger, level, logger.info)
        method(f"UI: {message}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Instrument selected public functions with logging
# ---------------------------------------------------------------------------
try:
    fetch_chat_models = log_call()(fetch_chat_models)
    resolve_source_environment = log_call()(resolve_source_environment)
    execute_agent_run = log_call()(execute_agent_run)
except Exception:
    # Best-effort: if symbol names aren't present or wrapping fails, continue.
    pass



def main() -> None:
    """Entry point to run the Streamlit dashboard from this module.

    Importing `dashboard.financial_analyst_dashboard` executes the Streamlit
    UI code. This function exists so you can run Streamlit against this
    module (e.g. `streamlit run dashboard/financial_analyst_app.py`) and have
    the UI start correctly.
    """
    try:
        # Import the UI module; Streamlit will execute the top-level UI code.
        import dashboard.financial_analyst_dashboard as _ui  # noqa: F401
    except Exception as err:  # pragma: no cover - runtime error reporting
        print("Failed to import dashboard.financial_analyst_dashboard:", err)
        raise


if __name__ == "__main__":
    main()
