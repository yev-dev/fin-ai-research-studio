# streamlit run dashboard/financial_analyst_dashboard.py

"""
Financial Data Analysis — Streamlit presentation layer.

All business logic lives in ``fin_ai.core.dashboard_engine``.  This module
only contains Streamlit UI widgets and rendering helpers.
"""

from __future__ import annotations

import json
import os
import warnings
from hashlib import md5
from pathlib import Path
from time import perf_counter
from datetime import datetime

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Reduce noisy dependency logs from transformers backends we do not use directly.
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

WARNING_PATTERNS = [
    r"Accessing `__path__` from `.models.vilt.image_processing_vilt`.*",
    r"Accessing `__path__` from `.models.aria.image_processing_aria`.*",
    r".*Behavior may be different and this alias will be removed in future versions\..*",
    r".*Disabling PyTorch because PyTorch >= 2\.4 is required but found 2\.2\.2.*",
    r".*PyTorch was not found\. Models won't be available and only tokenizers, configuration and file/data utilities can be used\..*",
]
for pattern in WARNING_PATTERNS:
    warnings.filterwarnings("ignore", message=pattern)

import streamlit as st
import time
import re
from string import Formatter
import streamlit.components.v1 as components
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
    # find_source_document,
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
from fin_ai.core.tools import publish_research_pdf, publish_research_html
from fin_ai.agents.prompts_library import RESEARCH_ANALYSIS

# Agent library for sidebar listing
from fin_ai.agents.agent_library import library as agent_library

st.set_page_config(page_title="Financial Data Analysis", layout="wide")
SIDEBAR_PREVIEW_WIDTH = 320


def _looks_like_embedding_model(model_id: str) -> bool:
    """Heuristic filter for embedding-capable model identifiers."""
    mid = (model_id or "").strip().lower()
    return "embedding" in mid or "embed" in mid

# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=10)
def get_local_model_options() -> tuple[list[str], list[str], str | None]:
    return [DEFAULT_CHAT_MODEL], [DEFAULT_EMBEDDING_MODEL], None

@st.cache_data(show_spinner=False)
def get_cached_csv_thumbnail_path(csv_path: str, images_folder: str, source_mtime: float) -> str:
    del source_mtime
    return render_csv_thumbnail(csv_path, images_folder)

@st.cache_data(show_spinner=False)
def get_cached_pdf_page_paths(pdf_path: str, images_folder: str, zoom: float, source_mtime: float) -> tuple[str, ...]:
    del source_mtime
    return tuple(str(p) for p in render_pdf_pages(pdf_path, images_folder, zoom=zoom))


def _parse_int_or_none(value: str | None) -> int | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


KNOWN_SOURCE_GROUPS = {"pdf", "csv", "json", "html", "url"}


def _normalize_source_groups(group_map: dict[str, list[str]]) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for group, names in group_map.items():
        key = str(group).strip().lower()
        if key not in KNOWN_SOURCE_GROUPS:
            continue
        normalized.setdefault(key, []).extend(
            str(name).removesuffix(".faiss") for name in names
        )
    return {group: sorted(set(names)) for group, names in normalized.items()}


@st.cache_data(ttl=20)
def get_provider_model_ids(provider: str, api_key: str = "", base_url: str = "") -> list[str]:
    listing_provider = provider.removeprefix("proxied_")
    kwargs: dict[str, str] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    models = fetch_models(listing_provider, **kwargs)
    return [m.id for m in models if getattr(m, "id", "")]

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_response_output(response_text: str, response_type: str, panel_key: str) -> None:
    if response_type == "Plain Text":
        st.text(response_text)
        return

    if response_type == "Python Code":
        raw_code = extract_python_code(response_text)
        code = sanitize_generated_python_code(raw_code)
        if code != raw_code:
            st.caption("Auto-cleaned malformed Python syntax from model output.")
        st.code(code, language="python")

        with st.expander("Run in Streamlit (Pyodide/WebAssembly)", expanded=False):
            _render_pyodide_runner(code, panel_key=panel_key)

        confirm_key = f"confirm_python_{panel_key}"
        confirmed = st.checkbox("I understand this will execute model-generated Python locally.", key=confirm_key)
        if st.button("Execute Python", key=f"execute_python_{panel_key}"):
            if not confirmed:
                st.warning("Confirm execution before running Python code.")
                return
            stdout_text, stderr_text = execute_python_code(code)
            if stdout_text.strip():
                st.text(stdout_text)
            if stderr_text.strip():
                st.error(stderr_text)
            elif not stdout_text.strip():
                st.success("Python code executed successfully with no output.")
        return

    st.markdown(response_text.replace('$', '\\$'))


def render_source_citations(citations_text: str | None, response_type: str) -> None:
    if not citations_text:
        return
    st.caption("Source-level citations")
    if response_type == "Plain Text":
        st.text(citations_text)
    else:
        st.markdown(citations_text.replace('$', '\\$'))


def _render_pyodide_runner(initial_code: str, panel_key: str) -> None:
    code_json = json.dumps(initial_code)
    html = f"""
<div style="font-family: ui-monospace, SFMono-Regular, Menlo, monospace; border: 1px solid #ddd; border-radius: 8px; padding: 10px;">
    <div style="font-family: system-ui, -apple-system, Segoe UI, sans-serif; font-size: 14px; margin-bottom: 8px;">
        A WebAssembly-powered Python kernel backed by Pyodide
    </div>
    <textarea id="code_{panel_key}" style="width: 100%; height: 170px;"></textarea>
    <div style="margin-top: 8px; display: flex; gap: 8px; align-items: center;">
        <button id="run_{panel_key}">Run In Browser</button>
        <span id="status_{panel_key}" style="font-family: system-ui, -apple-system, Segoe UI, sans-serif; font-size: 12px; color: #444;"></span>
    </div>
    <pre id="output_{panel_key}" style="white-space: pre-wrap; margin-top: 10px; background: #f7f7f7; padding: 10px; border-radius: 6px; max-height: 180px; overflow: auto;"></pre>
</div>
<script>
    const initialCode = {code_json};
    const codeEl = document.getElementById("code_{panel_key}");
    const runEl = document.getElementById("run_{panel_key}");
    const statusEl = document.getElementById("status_{panel_key}");
    const outputEl = document.getElementById("output_{panel_key}");
    codeEl.value = initialCode;
    async function ensurePyodide() {{
        if (!window.__streamlitPyodideReady) {{
            statusEl.textContent = "Loading Pyodide runtime...";
            if (!window.loadPyodide) {{
                await new Promise((resolve, reject) => {{
                    const script = document.createElement("script");
                    script.src = "https://cdn.jsdelivr.net/pyodide/v0.27.5/full/pyodide.js";
                    script.onload = resolve;
                    script.onerror = reject;
                    document.head.appendChild(script);
                }});
            }}
            window.__streamlitPyodideReady = await window.loadPyodide();
        }}
        return window.__streamlitPyodideReady;
    }}
    runEl.onclick = async () => {{
        runEl.disabled = true; outputEl.textContent = "";
        try {{
            const pyodide = await ensurePyodide();
            statusEl.textContent = "Running...";
            const stdout = [], stderr = [];
            pyodide.setStdout({{ batched: (msg) => stdout.push(msg) }});
            pyodide.setStderr({{ batched: (msg) => stderr.push(msg) }});
            const result = await pyodide.runPythonAsync(codeEl.value);
            const chunks = [];
            if (stdout.length) chunks.push(stdout.join("\\n"));
            if (result !== undefined) chunks.push(String(result));
            if (stderr.length) chunks.push("\\n[stderr]\\n" + stderr.join("\\n"));
            outputEl.textContent = chunks.join("\\n\\n") || "(no output)";
            statusEl.textContent = "Done";
        }} catch (err) {{ outputEl.textContent = String(err); statusEl.textContent = "Failed"; }}
        finally {{ runEl.disabled = false; }}
    }};
</script>
"""
    components.html(html, height=460, scrolling=True)


def display_pdf_in_sidebar(pdf_path: str | Path, file_name: str) -> None:
    try:
        images_folder = Path(VECTOR_DB_DIR) / file_name / "images"
        source_path = Path(pdf_path)
        source_mtime = source_path.stat().st_mtime if source_path.exists() else 0.0
        for page_index, img_path in enumerate(
            get_cached_pdf_page_paths(str(source_path), str(images_folder), zoom=1.5, source_mtime=source_mtime),
            start=1,
        ):
            st.sidebar.image(str(img_path), caption=f"Page {page_index}", width=SIDEBAR_PREVIEW_WIDTH)
    except Exception as e:
        st.sidebar.error(f"Error loading PDF: {str(e)}")


def display_csv_in_sidebar(csv_path: str | Path, file_name: str) -> None:
    try:
        images_folder = Path(VECTOR_DB_DIR) / file_name / "images"
        source_path = Path(csv_path)
        source_mtime = source_path.stat().st_mtime if source_path.exists() else 0.0
        img = get_cached_csv_thumbnail_path(str(source_path), str(images_folder), source_mtime)
        if img:
            st.sidebar.image(img, caption="CSV Preview", width=SIDEBAR_PREVIEW_WIDTH)
        else:
            st.sidebar.info("Could not generate CSV preview.")
    except Exception as e:
        st.sidebar.error(f"Error loading CSV preview: {str(e)}")


# ---------------------------------------------------------------------------
# -- Page layout
# ---------------------------------------------------------------------------

st.title("FinAI Research Studio")

# Discover vector stores
source_vector_stores = get_source_vector_stores()
vector_db_names = get_vector_db_names(source_vector_stores)
available_chat_models, available_embedding_models, model_load_error = get_local_model_options()

# ---------------------------------------------------------------------------
# -- Reasoning -----------------------------------------------------------
# ---------------------------------------------------------------------------
st.sidebar.subheader("Reasoning")
st.sidebar.caption(
    "Choose the LLM that will analyse the retrieved context and generate "
    "answers. Select a provider, model, response format, and optional "
    "financial data tools."
)

# Build provider label→key mapping from ProviderConfig
_all_cfgs = known_providers()
provider_label_to_key = {cfg.label: name for name, cfg in _all_cfgs.items()}
default_provider = os.getenv("DEFAULT_PROVIDER", "ollama").strip().lower()
provider_labels = list(provider_label_to_key.keys())
default_provider_index = next(
    (i for i, k in enumerate(provider_labels) if provider_label_to_key[k] == default_provider), 0,
)
selected_provider_label = st.sidebar.selectbox("Select Provider", provider_labels, index=default_provider_index, key="chat_provider")
selected_provider = provider_label_to_key[selected_provider_label]
_pcfg = get_provider_config(selected_provider)

# Initialise provider-scoped vars with safe defaults
github_token = os.getenv("GITHUB_TOKEN", "")
github_endpoint = os.getenv("GITHUB_ENDPOINT", "https://models.github.ai/inference")
deepseek_token = os.getenv("DEEPSEEK_TOKEN", "")
deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL)
ollama_chat_base_url = os.getenv("OLLAMA_ENDPOINT", OLLAMA_BASE_URL)
proxy_port: int | None = None
http_proxy_port: int | None = None
https_proxy_port: int | None = None

# --- Show required / optional params based on ProviderConfig ---
if "api_key" in _pcfg.required_params or "api_key" in _pcfg.optional_params:
    _is_required = "api_key" in _pcfg.required_params
    _label = "API Key / Token" + (" *" if _is_required else "")
    _default_val = {"github": os.getenv("GITHUB_TOKEN", ""), "deepseek": os.getenv("DEEPSEEK_TOKEN", "")}.get(selected_provider, "")
    _value = st.sidebar.text_input(_label, value=_default_val, type="password", key=f"{selected_provider}_api_key")
    if selected_provider == "github":
        github_token = _value
    elif selected_provider == "deepseek":
        deepseek_token = _value

if "api_base" in _pcfg.optional_params:
    _default_base = _pcfg.default_base_url
    _current_base = os.getenv("GITHUB_ENDPOINT" if "github" in selected_provider else "DEEPSEEK_BASE_URL", _default_base) if selected_provider != "ollama" else os.getenv("OLLAMA_ENDPOINT", _default_base)
    _base_val = st.sidebar.text_input("API Base URL", value=_current_base, key=f"{selected_provider}_api_base")
    if selected_provider == "github" or selected_provider == "proxied_github":
        github_endpoint = _base_val
    elif selected_provider == "deepseek" or selected_provider == "proxied_deepseek":
        deepseek_base_url = _base_val
    elif selected_provider == "ollama":
        ollama_chat_base_url = _base_val

# Proxy ports: single-port mode (proxy_port) or split-port mode (http_proxy_port, https_proxy_port)
_show_proxy = "proxy_port" in _pcfg.optional_params or "http_proxy_port" in _pcfg.optional_params
if _show_proxy:
    with st.sidebar.expander("Proxy Settings", expanded=False):
        if "proxy_port" in _pcfg.optional_params:
            _default_proxy = os.getenv("PX_PROXY_PORT", "")
            proxy_port_val = st.text_input("Proxy Port", value=_default_proxy, key=f"{selected_provider}_proxy_port")
            proxy_port = _parse_int_or_none(proxy_port_val)
        if "http_proxy_port" in _pcfg.optional_params or "https_proxy_port" in _pcfg.optional_params:
            _default_http = os.getenv("PX_HTTP_PROXY_PORT", "")
            _default_https = os.getenv("PX_HTTPS_PROXY_PORT", "")
            http_proxy_port_val = st.text_input("HTTP Proxy Port", value=_default_http, key=f"{selected_provider}_http_proxy_port")
            https_proxy_port_val = st.text_input("HTTPS Proxy Port", value=_default_https, key=f"{selected_provider}_https_proxy_port")
            http_proxy_port = _parse_int_or_none(http_proxy_port_val)
            https_proxy_port = _parse_int_or_none(https_proxy_port_val)

# --- Model selection ---
if selected_provider == "github":
    with st.spinner("Fetching available GitHub models..."):
        display_model_options = get_provider_model_ids("github", api_key=github_token, base_url=github_endpoint)
    if not display_model_options:
        display_model_options = [os.getenv("GITHUB_MODEL", DEFAULT_GITHUB_MODEL)]
    default_chat_model = os.getenv("GITHUB_MODEL", DEFAULT_GITHUB_MODEL)
    default_chat_index = display_model_options.index(default_chat_model) if default_chat_model in display_model_options else 0
    selected_model = st.sidebar.selectbox("Select Model", display_model_options, index=default_chat_index, key="github_model")
elif selected_provider == "proxied_github":
    with st.spinner("Fetching available GitHub models..."):
        display_model_options = get_provider_model_ids("proxied_github", api_key=os.getenv("GITHUB_TOKEN", ""), base_url=github_endpoint)
    if not display_model_options:
        display_model_options = [os.getenv("GITHUB_MODEL", DEFAULT_GITHUB_MODEL)]
    default_chat_model = os.getenv("GITHUB_MODEL", DEFAULT_GITHUB_MODEL)
    default_chat_index = display_model_options.index(default_chat_model) if default_chat_model in display_model_options else 0
    selected_model = st.sidebar.selectbox("Select Model", display_model_options, index=default_chat_index, key="proxied_github_model")

elif selected_provider == "deepseek":
    with st.spinner("Fetching available DeepSeek models..."):
        deepseek_model_ids = get_provider_model_ids("deepseek", api_key=deepseek_token, base_url=deepseek_base_url)
    if not deepseek_model_ids:
        deepseek_model_ids = [os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)]
    default_ds = os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    default_ds_idx = deepseek_model_ids.index(default_ds) if default_ds in deepseek_model_ids else 0
    selected_model = st.sidebar.selectbox("Select Model", deepseek_model_ids, index=default_ds_idx, key="deepseek_model")

elif selected_provider == "proxied_deepseek":
    with st.spinner("Fetching available DeepSeek models..."):
        proxied_ds_models = get_provider_model_ids("proxied_deepseek", api_key=os.getenv("DEEPSEEK_TOKEN", ""), base_url=deepseek_base_url)
    if not proxied_ds_models:
        proxied_ds_models = [os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)]
    default_ds = os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    default_ds_idx = proxied_ds_models.index(default_ds) if default_ds in proxied_ds_models else 0
    selected_model = st.sidebar.selectbox("Select Model", proxied_ds_models, index=default_ds_idx, key="proxied_deepseek_model")

else:  # ollama
    default_chat_model = os.getenv("OLLAMA_MODEL", DEFAULT_CHAT_MODEL)
    with st.spinner("Fetching available Ollama models..."):
        ollama_model_ids = get_provider_model_ids("ollama", base_url=ollama_chat_base_url)
    display_ollama_models = ollama_model_ids or available_chat_models or [default_chat_model]
    default_chat_index = display_ollama_models.index(default_chat_model) if default_chat_model in display_ollama_models else 0
    selected_model = st.sidebar.selectbox("Model", display_ollama_models, index=default_chat_index, key="ollama_chat_model")

# Reset session state button
if st.sidebar.button("Reset Session", key="reset_session_btn"):
    keys_to_clear = [
        "agent_response",
        "agent_publication",
        "agent_response_debug",
        "agent_publication_filepath",
        "agent_trace",
        "latest_response",
        "latest_retrieval",
        "question_history",
        "agent_research_name_main",
        "agent_prompt_main",
        "agent_response_debug",
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    # Use a safe rerun that works across Streamlit versions
    def _safe_rerun():
        try:
            st.experimental_rerun()
            return
        except Exception:
            pass
        # Try raising the runtime RerunException used internally by Streamlit
        RerunException = None
        try:
            import importlib
            mod = importlib.import_module("streamlit.runtime.scriptrunner")
            RerunException = getattr(mod, "RerunException", None)
        except Exception:
            try:
                import importlib
                mod2 = importlib.import_module("streamlit.script_runner")
                RerunException = getattr(mod2, "RerunException", None)
            except Exception:
                RerunException = None
        if RerunException:
            raise RerunException()
        # Fallback: tweak query params to force a browser reload
        try:
            params = st.experimental_get_query_params() or {}
            params["_reset"] = int(time.time())
            st.experimental_set_query_params(**params)
            return
        except Exception:
            # Final fallback: meta refresh
            st.markdown('<meta http-equiv="refresh" content="0">', unsafe_allow_html=True)

    _safe_rerun()

response_type = st.sidebar.selectbox("Select Response Type", ["Plain Text", "Markdown", "Python Code"], index=1, key="response_type")
auto_truncate_prompt = st.sidebar.checkbox("Auto-truncate prompt (gpt-5 guard)", value=True)
use_tools = st.sidebar.checkbox("Enable function tools (financial data)", value=False)

if use_tools:
    with st.sidebar.expander("Available tools", expanded=False):
        from fin_ai.core.tools import YAHOO_FINANCE_TOOLS
        for tool in YAHOO_FINANCE_TOOLS:
            if tool.get("type") == "function":
                fn = tool["function"]
                st.sidebar.markdown(f"**`{fn['name']}`** — {fn.get('description', '')}")

if model_load_error:
    st.sidebar.warning(model_load_error)

# ---------------------------------------------------------------------------
# -- Embedding -----------------------------------------------------------
# ---------------------------------------------------------------------------
st.sidebar.caption(
    "Configure the model that converts your documents into vector "
    "representations. Embeddings are used both when indexing new documents "
    "and when retrieving relevant context during querying."
)

embeddings_provider_label_to_key = {"Local Ollama": "ollama", "GitHub Models": "github"}
default_emb_provider = os.getenv("DEFAULT_EMBEDDINGS_PROVIDER", DEFAULT_EMBEDDINGS_PROVIDER).strip().lower()
emb_labels = list(embeddings_provider_label_to_key.keys())
default_emb_idx = next((i for i, k in enumerate(emb_labels) if embeddings_provider_label_to_key[k] == default_emb_provider), 0)
selected_emb_provider_label = st.sidebar.selectbox("Select Embedding Provider", emb_labels, index=default_emb_idx, key="emb_provider_select")
selected_emb_provider = embeddings_provider_label_to_key[selected_emb_provider_label]

if selected_emb_provider == "github":
    embedding_github_base_url = st.sidebar.text_input(
        "GitHub Embedding Base URL",
        value=os.getenv("GITHUB_EMBEDDING_BASE_URL", GITHUB_EMBEDDING_BASE_URL),
        key="embedding_github_base_url",
    )
    embedding_github_token = st.sidebar.text_input("GitHub Token (Embeddings)", value=os.getenv("GITHUB_TOKEN", ""), type="password", key="embedding_github_token")
    with st.spinner("Fetching GitHub embedding models..."):
        embedding_model_ids = [
            m for m in get_provider_model_ids("github", api_key=embedding_github_token, base_url=embedding_github_base_url)
            if _looks_like_embedding_model(m)
        ]
    available_embedding_models_display = embedding_model_ids if embedding_model_ids else [DEFAULT_GITHUB_EMBEDDING_MODEL, "openai/text-embedding-3-large"]
    if not embedding_model_ids:
        st.sidebar.warning(
            "No embedding-capable GitHub models were detected from the catalog. "
            "Using safe defaults to avoid 400 errors from /embeddings."
        )
    default_emb_model_idx = available_embedding_models_display.index(DEFAULT_GITHUB_EMBEDDING_MODEL) if DEFAULT_GITHUB_EMBEDDING_MODEL in available_embedding_models_display else 0
    embeddings_base_url = embedding_github_base_url
else:
    ollama_emb_endpoint = st.sidebar.text_input("Ollama Endpoint (Embeddings)", value=os.getenv("OLLAMA_ENDPOINT", OLLAMA_BASE_URL), key="embedding_ollama_endpoint")
    with st.spinner("Fetching available Ollama embedding models..."):
        ollama_embedding_models = [
            m for m in get_provider_model_ids("ollama", base_url=ollama_emb_endpoint)
            if _looks_like_embedding_model(m)
        ]
    available_embedding_models_display = ollama_embedding_models or available_embedding_models or [DEFAULT_EMBEDDING_MODEL]
    default_emb_model_idx = available_embedding_models_display.index(DEFAULT_EMBEDDING_MODEL) if DEFAULT_EMBEDDING_MODEL in available_embedding_models_display else 0
    embeddings_base_url = ollama_emb_endpoint

selected_embedding_model = st.sidebar.selectbox("Select Embedding Model", available_embedding_models_display, index=default_emb_model_idx, key="embedding_model")

st.sidebar.divider()

# -- Active runtime config --------------------------------------------------
_active_chat_base = (
    github_endpoint
    if selected_provider in ("github", "proxied_github")
    else deepseek_base_url
    if selected_provider in ("deepseek", "proxied_deepseek")
    else ollama_chat_base_url
)
st.sidebar.caption(
    "\n".join(
        [
            "Active Runtime Config",
            f"Chat: {selected_provider} | {selected_model}",
            f"Chat Base: {_active_chat_base}",
            f"Embeddings: {selected_emb_provider} | {selected_embedding_model}",
            f"Embedding Base: {embeddings_base_url}",
        ]
    )
)

# -- Previous Questions -----------------------------------------------------
if vector_db_names:
    _history_db_name = vector_db_names[0]
    st.session_state["question_history"] = load_history(_history_db_name)
    st.session_state["history_vector_db"] = _history_db_name

    with st.sidebar.expander("Previous Questions", expanded=False):
        history = st.session_state.get("question_history", [])
        if not history:
            st.sidebar.caption("No saved question history yet.")
        else:
            if st.sidebar.button("Clear History", key=f"clear_history_{_history_db_name}"):
                clear_history(_history_db_name)
                st.session_state["question_history"] = []
                st.rerun()
            for idx, item in enumerate(history[:10], start=1):
                with st.sidebar.expander(
                    f"Q{idx}: {item['question'][:80]}{'...' if len(item['question']) > 80 else ''}",
                    expanded=False,
                ):
                    st.sidebar.caption(
                        f"Model: {item['chat_model']} | "
                        f"Embedding: {item.get('embedding_model', 'N/A')} | "
                        f"Type: {item.get('response_type', 'Markdown')} | "
                        f"Mode: {item.get('retrieval_mode', 'ensemble')} | "
                        f"Time: {item.get('answer_seconds', 0):.2f}s"
                    )

    # -- Maintenance ---------------------------------------------------------
    with st.sidebar.expander("Maintenance", expanded=False):
        from fin_ai.core.rag import RAGSourceStore, discover_vector_stores_by_source

        st.sidebar.caption("Registered RAG Sources")
        rag_store = RAGSourceStore()
        # Only show sources that have a FAISS index on disk
        on_disk_stores = discover_vector_stores_by_source()
        df_sources = rag_store.to_dataframe()
        if not df_sources.empty and on_disk_stores:
            df_on_disk = df_sources[df_sources["name"].isin(on_disk_stores)].copy()
            if not df_on_disk.empty:
                # Only display known source types to avoid confusing/invalid
                # entries that may have been recorded with an unexpected
                # source_type value. Normalize to lowercase/trimmed strings
                # so variants like 'PDF' or ' pdf ' are handled.
                KNOWN_SOURCE_GROUPS = ["pdf", "csv", "json", "html", "url"]
                # Create a normalized column for comparison but keep the
                # original values for display.
                df_on_disk = df_on_disk.assign(
                    _source_type_norm=df_on_disk["source_type"].astype(str).str.lower().str.strip()
                )
                df_on_disk_known = df_on_disk[df_on_disk["_source_type_norm"].isin(KNOWN_SOURCE_GROUPS)].copy()
                if not df_on_disk_known.empty:
                    st.sidebar.dataframe(
                        df_on_disk_known[["name", "source_type", "chunk_count", "embedding_model"]],
                        use_container_width=True,
                        hide_index=True,
                    )
                    total = len(df_on_disk_known)
                    total_chunks = df_on_disk_known["chunk_count"].sum()
                    st.sidebar.caption(f"{total} FAISS index(es) · {int(total_chunks):,} chunk(s)")
                else:
                    st.sidebar.caption("No registered RAG sources with known source types found. Consider syncing or re-registering sources.")
            else:
                st.sidebar.caption("No FAISS index found for registered sources.")
        elif on_disk_stores and df_sources.empty:
            st.sidebar.caption(f"{len(on_disk_stores)} FAISS index(es) found — sync to register.")
        else:
            st.sidebar.caption("No FAISS indexes found.")

        sync_col, _ = st.sidebar.columns([1, 2])
        with sync_col:
            if st.sidebar.button("Sync Sources", key="rag_sync_btn"):
                added = rag_store.sync_from_disk()
                if added:
                    st.sidebar.success(f"Added {added} new source(s) from disk.")
                    st.rerun()
                else:
                    st.sidebar.info("All sources already registered.")

        st.sidebar.divider()

        st.sidebar.warning("This permanently deletes a vector DB and all related files.")
        _purge_db = st.sidebar.selectbox("Select DB to purge", vector_db_names, key="purge_db_select")
        confirm_purge = st.sidebar.checkbox(f"Confirm purge of '{_purge_db}'", key="confirm_purge_main")
        if st.sidebar.button("Purge Vector DB", type="secondary", disabled=not confirm_purge, key="purge_db_btn"):
            deleted = purge_vector_db(_purge_db)
            if deleted:
                st.session_state["question_history"] = []
                st.session_state.pop("latest_response", None)
                st.success(f"Purged vector DB '{_purge_db}'.")
                st.rerun()
            else:
                st.sidebar.warning(f"No files found to purge for '{_purge_db}'.")

# -- Agents (main panel) ---------------------------------------------------
# ---------------------------------------------------------------------------
# Initialise agent UI state with safe defaults (may be overridden below)
agent_submit = False
agent_rag_query = ""
selected_agent = ""
agent_format = "html"
agent_email = ""
agent_prompt_template = "Custom"
agent_prompt_asset = "NVDA"

st.subheader("Agent Workflows")
st.caption(
    "Run agentic research workflows with configurable LLM backends. "
    "Agents can access market data tools (offline/online) and indexed "
    "RAG documents when available."
)

agent_col1, agent_col2 = st.columns([2, 1])
with agent_col1:
    selected_agent = st.selectbox(
        "Select Agent Profile",
        [n for n in agent_library],
        index=0,
        key="agent_profile_main",
    )
with agent_col2:
    agent_format = st.selectbox(
        "Report format",
        ["html", "pdf"],
        index=0,
        key="agent_format_main",
    )

with st.expander("Prompt Controls", expanded=True):
    agent_prompt_template = st.selectbox(
        "Template",
        ["Custom", *RESEARCH_ANALYSIS.keys()],
        index=0,
        key="agent_prompt_template_main",
    )
    # Make asset symbol optional and default to empty so templates are not
    # pre-populated unexpectedly. Users can fill parameters exposed below.
    agent_prompt_asset = st.text_input(
        "Asset Symbol (optional)",
        value=st.session_state.get("agent_prompt_asset_main", ""),
        key="agent_prompt_asset_main",
        help="Optional: used to populate template parameters like {asset}.",
    )

    # Pre-fill research name with a helpful default if not provided
    default_research_name = st.session_state.get("agent_research_name_main", "")
    if not default_research_name:
        date_str = datetime.utcnow().strftime("%Y%m%d")
        asset_part = (agent_prompt_asset or "").strip() or "asset"
        default_research_name = f"{selected_agent}*{asset_part}*{date_str}"

    agent_research_name = st.text_input(
        "Research Name (optional)",
        value=default_research_name,
        key="agent_research_name_main",
        help="Optional title to use when publishing research reports.",
    )

    # Preview expected generated filename/link for the chosen title
    def _safe_title_for_filename(title: str) -> str:
        safe = "".join(c for c in (title or "") if c.isalnum() or c in (" ", "-", "_")).rstrip()
        return (safe[:80] or "research_report").replace(" ", "_")

    chosen_title = agent_research_name.strip() or f"{selected_agent} Report"
    preview_prefix = _safe_title_for_filename(chosen_title)
    preview_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    preview_ext = agent_format or "html"
    preview_filename = f"{preview_prefix}_{preview_ts}.{preview_ext}"
    st.caption(f"Preview filename: {preview_filename}")

    if agent_prompt_template != "Custom":
        # Safely lookup the template; fall back to the raw selection string if missing
        template_text = RESEARCH_ANALYSIS.get(agent_prompt_template)
        if template_text:
            # Discover placeholder field names used in the template (e.g. 'asset')
            formatter = Formatter()
            field_names = [fname for _, fname, _, _ in formatter.parse(template_text) if fname]

            # For each placeholder, expose a small input so users can provide values
            params: dict[str, str] = {}
            for fname in field_names:
                key = f"agent_param_{fname}"
                default_val = st.session_state.get(key, "")
                # Use a single-line text input for parameters; allow empty values
                val = st.text_input(f"Parameter: {fname}", value=default_val, key=key)
                if isinstance(val, str) and val.strip():
                    params[fname] = val.strip()

            # Also include the optional top-level asset input if present
            if agent_prompt_asset and agent_prompt_asset.strip():
                params.setdefault("asset", agent_prompt_asset.strip())

            # Safe partial formatter: replace placeholders only for provided params
            def _partial_format(tpl: str, mapping: dict[str, str]) -> str:
                def _repl(m):
                    name = m.group(1)
                    return mapping.get(name, "{" + name + "}")
                return re.sub(r"\{(\w+)\}", _repl, tpl)

            formatted = _partial_format(template_text, params)
            st.session_state["agent_prompt_main"] = formatted
        else:
            st.session_state["agent_prompt_main"] = agent_prompt_template

    agent_rag_query = st.text_area(
        "Agent Task Prompt",
        placeholder='e.g. "Analyse NVDA financials and competitive position"',
        key="agent_prompt_main",
        height=180,
    )

agent_col_a, agent_col_b, agent_col_c = st.columns([1, 1, 3])
with agent_col_a:
    agent_submit = st.button(" Run Agent", key="run_agent_main", type="primary", use_container_width=True)
with agent_col_b:
    if st.button("Clear Output", key="clear_agent_main", use_container_width=True):
        st.session_state.pop("agent_response", None)
        st.session_state.pop("agent_publication", None)
        st.session_state.pop("latest_response", None)
        st.session_state.pop("latest_retrieval", None)
        st.rerun()
with agent_col_c:
    agent_email = st.text_input(
        "Email report (optional)",
        placeholder="analyst@firm.com",
        key="agent_email_main",
    )

st.divider()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# -- RAG Query (main panel) ------------------------------------------------
# ---------------------------------------------------------------------------

# Initialise shared variables with sensible defaults
_selected_source_groups = st.session_state.get("query_source_groups", [])
_selected_query_vector_dbs = st.session_state.get("query_vector_dbs", vector_db_names[:1])
_retrieval_mode = st.session_state.get("retrieval_mode", "ensemble")
selected_query_vector_dbs = vector_db_names[:1] if not vector_db_names else _selected_query_vector_dbs
retrieval_mode = _retrieval_mode
selected_source_names = []
available_source_names = []
available_source_groups: list[str] = []
source_groups_map_norm: dict[str, list[str]] = {}
query_source_configs = []
loaded_stores: dict[str, FAISS] = {}

# Load embeddings and vector stores for RAG querying
embeddings = None
if vector_db_names:
    _github_token = os.environ.get("GITHUB_TOKEN", "")
    if selected_emb_provider == "github":
        try:
            _github_token = embedding_github_token or _github_token
        except NameError:
            pass

    try:
        embeddings = create_embeddings(
            provider=selected_emb_provider,
            model=selected_embedding_model,
            api_base=embeddings_base_url,
            api_key=_github_token if selected_emb_provider == "github" else None,
        )
    except Exception as e:
        st.sidebar.error(f"Embeddings error: {e}")

    try:
        if embeddings is None:
            raise RuntimeError("No embeddings available — check the sidebar for errors.")
        # Build available source groups/names from the RAG registry so the
        # UI shows all registered documents, not only those currently
        # loaded into memory (loading all stores can be expensive).
        source_groups_map_norm = _normalize_source_groups(discover_source_groups())
        available_source_groups = sorted(source_groups_map_norm.keys())

        # Available source names should reflect what's registered and also
        # exist as vector DBs on disk (vector_db_names).
        available_source_names = sorted(
            {
                name
                for names in source_groups_map_norm.values()
                for name in names
                if name in vector_db_names
            }
        )
        if not available_source_names:
            available_source_names = vector_db_names[:]
    except (ValueError, RuntimeError) as e:
        st.sidebar.error(str(e))
        loaded_stores = {}
        query_source_configs = []
        available_source_names = []
        available_source_groups = []
        source_groups_map_norm = {}

# RAG Query
# Keep the RAG query controls collapsible until the user submits a question
if "rag_query_expanded" not in st.session_state:
    st.session_state["rag_query_expanded"] = False
# Expand if there's a previous response
rag_expanded = bool(st.session_state.get("rag_query_expanded", False)) or bool(st.session_state.get("latest_response"))
with st.expander("RAG Query", expanded=rag_expanded):
    # Upload document inline (always visible — even when vector_db is empty)
    with st.expander("Upload New Document", expanded=not bool(vector_db_names)):
        selected_source_type = st.selectbox("Type of Source Document", SUPPORTED_UPLOAD_TYPES, index=0, key="source_type_rag")
        uploaded_file = st.file_uploader("Upload a document for analysis", type=SUPPORTED_UPLOAD_TYPES, key="upload_rag")
        if uploaded_file:
            binary = uploaded_file.getvalue()
            if st.button("Process Document and Store in Vector DB", key="process_rag"):
                with st.spinner("Processing document..."):
                    _gh_tok = os.environ.get("GITHUB_TOKEN", "")
                    if selected_emb_provider == "github":
                        try:
                            _gh_tok = embedding_github_token or _gh_tok
                        except NameError:
                            pass
                    result_upload = process_uploaded_document(
                        file_binary=binary,
                        file_name=uploaded_file.name,
                        embedding_model=selected_embedding_model,
                        embedding_base_url=embeddings_base_url,
                        emb_provider=selected_emb_provider,
                        source_type=selected_source_type,
                        github_token=_gh_tok if selected_emb_provider == "github" else None,
                    )
                    st.success("Document processed and stored in the vector database.")
                    st.caption(f"Document processing completed in {result_upload['elapsed']:.2f} seconds.")
                    st.rerun()

    # --- Source selection & querying (only when stores exist) ---
    if vector_db_names:

        # --- Source Groups (top-level filter: pdf / csv / json / html / url) ---
        if available_source_groups:
            # Use the canonical known types as the selectable options, but keep
            # sensible defaults based on previous state or what is actually
            # available in the registry.
            options = sorted(KNOWN_SOURCE_GROUPS)
            # Prefer session-selected groups if they are valid known types,
            # otherwise prefer groups that are both known and present in the registry.
            default_selection = [g for g in (_selected_source_groups or available_source_groups) if g in options]
            if not default_selection:
                default_selection = [g for g in options if g in available_source_groups] or options[:]

            selected_source_groups = st.multiselect(
                "Source Groups",
                options,
                default=default_selection,
                key="query_source_groups",
                help="Select document type(s) to search within.",
            )

            # Filter available documents to only those in selected groups using the
            # normalized registry map so casing differences don't hide items.
            if selected_source_groups:
                allowed_names: set[str] = set()
                for g in selected_source_groups:
                    allowed_names.update(source_groups_map_norm.get(str(g).lower().strip(), []))
                group_filtered_names = [name for name in available_source_names if name in allowed_names]
            else:
                group_filtered_names = available_source_names[:]
            filtered_doc_names = sorted(group_filtered_names)
        else:
            selected_source_groups = []
            filtered_doc_names = available_source_names or vector_db_names[:]

        # --- Query Vector Documents (filtered by source groups) ---
        selected_query_vector_dbs = st.multiselect(
            "Query Vector Documents",
            filtered_doc_names,
            default=selected_query_vector_dbs,
            key="query_vector_dbs",
        )

        # Retrieval mode below source groups
        retrieval_mode = st.selectbox("Retrieval Mode", ["ensemble", "separate", "routed"], index=0, key="retrieval_mode")

        # Question input
        question = ""
        submit_clicked = False
        question = st.text_input("Enter your question:", placeholder="e.g., What is the company's revenue for the quarter?", key="question_input")
        submit_clicked = st.button("Submit Question")

        # If the user submitted a question, ensure the expander remains open
        if submit_clicked:
            st.session_state["rag_query_expanded"] = True

        # -- Submit and answer ------------------------------------------------------
        if submit_clicked and question:
            if not selected_query_vector_dbs:
                st.error("Select at least one document before submitting a question.")
            else:
                active_stores = load_vector_stores_for_query(selected_query_vector_dbs, source_vector_stores, embeddings)
                active_configs = build_query_source_configs(active_stores, group_by="vector_db")
                if not active_configs:
                    st.error("The selected documents are not available. Please re-select.")
                else:
                    with st.spinner("Answering your question..."):
                        result = answer_question(
                            question,
                            active_configs,
                            provider=selected_provider,
                            model=selected_model,
                            api_base=(
                                github_endpoint
                                if selected_provider in ("github", "proxied_github")
                                else deepseek_base_url
                                if selected_provider in ("deepseek", "proxied_deepseek")
                                else ollama_chat_base_url
                            ),
                            api_key=(
                                github_token
                                if selected_provider == "github"
                                else deepseek_token
                                if selected_provider == "deepseek"
                                else None
                            ),
                            proxy_port=proxy_port,
                            http_proxy_port=http_proxy_port,
                            https_proxy_port=https_proxy_port,
                            system_prompt="You are a concise financial analysis assistant.",
                            temperature=0.2,
                            retrieval_mode=retrieval_mode,
                            auto_truncate_prompt=bool(auto_truncate_prompt),
                            use_tools=use_tools,
                        )

                        llm_response = result.get("response")
                        metadata = result.get("metadata")
                        if not llm_response:
                            st.error("Model returned no response.")
                            st.stop()

                        if metadata and metadata.prompt_truncated:
                            st.warning(f"Prompt was truncated to fit gpt-5 input limits ({metadata.prompt_tokens_before_guard} -> {metadata.prompt_tokens_after_guard} tokens before sending).")

                        st.session_state["latest_response"] = {"question": question, "answer": llm_response, "response_type": response_type}

                    # Store the retrieval result for citation formatting
                    _llm_result = result.get("llm_result")
                    if _llm_result and _llm_result.retrieval:
                        st.session_state["latest_retrieval"] = _llm_result.retrieval

                    _history_db = selected_query_vector_dbs[0] if selected_query_vector_dbs else "default"
                    save_history_entry(_history_db, {
                        "question": question,
                        "answer": llm_response,
                        "vector_db": _history_db,
                        "chat_model": selected_model,
                        "provider": selected_provider,
                        "embedding_model": selected_embedding_model,
                        "response_type": response_type,
                        "answer_seconds": result["elapsed"],
                    })
                    st.session_state["question_history"] = load_history(_history_db)

# ---------------------------------------------------------------------------
# -- Agent execution & display ---------------------------------------------
# ---------------------------------------------------------------------------

if agent_submit and agent_rag_query.strip():
    with st.spinner(f"Running {selected_agent} agent..."):
        # Progress UI elements
        progress_bar = st.progress(0)
        status_text = st.empty()

        def _progress_cb(pct: int, msg: str | None = None) -> None:
            try:
                progress_bar.progress(min(max(int(pct or 0), 0), 100))
            except Exception:
                pass
            try:
                if msg is not None:
                    status_text.text(msg)
            except Exception:
                pass
        # For proxied providers, don't pass token (proxy handles auth)
        _effective_gh_token = github_token if selected_provider == "github" else ""
        _effective_ds_token = deepseek_token if selected_provider == "deepseek" else ""
        agent_llm_config = build_agent_llm_config(
            provider=selected_provider,
            model=selected_model,
            ollama_base_url=ollama_chat_base_url,
            github_endpoint=github_endpoint,
            github_token=_effective_gh_token,
            deepseek_base_url=deepseek_base_url,
            deepseek_token=_effective_ds_token,
        )
        result = run_agent_task(
            agent_name=selected_agent,
            prompt=agent_rag_query.strip(),
            llm_config=agent_llm_config,
            embedding_model=selected_embedding_model,
            embedding_provider=selected_emb_provider,
            embedding_base_url=embeddings_base_url,
            chat_provider=selected_provider,
            is_publisher=(selected_agent == "Research_Publisher"),
            publisher_format=agent_format,
            publisher_email=agent_email.strip(),
            publisher_title=agent_research_name.strip(),
            progress_callback=_progress_cb,
        )
        # Ensure UI shows completion
        try:
            progress_bar.progress(100)
            status_text.text("Agent run complete")
        except Exception:
            pass
        if result["success"]:
            st.session_state["agent_response"] = result["response"]
            # Store structured debug info for Agent Debug panel
            st.session_state["agent_response_debug"] = {
                "trace": result.get("trace"),
                "raw_history": result.get("raw_history"),
                "stdout": result.get("stdout"),
                "stderr": result.get("stderr"),
            }
            if result.get("publication"):
                st.session_state["agent_publication"] = result["publication"]
                # Extract filepath from publication result and store for quick preview/linking
                try:
                    _pub = json.loads(result["publication"]) if isinstance(result["publication"], str) else result["publication"]
                    _pub_info = _pub.get("publish", _pub)
                    _fp = _pub_info.get("filepath")
                    if _fp:
                        st.session_state["agent_publication_filepath"] = _fp
                except Exception:
                    pass
            if result.get("trace"):
                st.session_state["agent_trace"] = result["trace"]

            # ------------------------------------------------------------------
            # Publish the agent's response in the report format selected in the
            # UI ("pdf"/"html").  This makes the dashboard always honour the
            # chosen format for ANY agent profile — not only Research_Publisher —
            # whenever the Run Agent button is clicked and instructions were
            # provided.  The generated file is stored for preview/download below.
            # ------------------------------------------------------------------
            try:
                _report_title = agent_research_name.strip() or f"{selected_agent} Report"
                # Mirror run_agent_task's cleanup of the AutoGen terminate marker.
                _report_content = result["response"].strip()
                if _report_content.endswith("TERMINATE"):
                    _report_content = _report_content[: -len("TERMINATE")].rstrip()

                if agent_format == "pdf":
                    _publication = publish_research_pdf(_report_content, title=_report_title)
                else:
                    _publication = publish_research_html(_report_content, title=_report_title)

                st.session_state["agent_publication"] = _publication
                _pub_data = json.loads(_publication)
                _pub_filepath = _pub_data.get("filepath")
                if _pub_filepath:
                    st.session_state["agent_publication_filepath"] = _pub_filepath
            except Exception as _pub_error:  # noqa: BLE001 - surface to the UI
                st.session_state["agent_publication"] = json.dumps(
                    {"status": "error", "error": str(_pub_error)}
                )
        else:
            st.session_state["agent_response"] = f"Agent error: {result['error']}"
            if result.get("trace"):
                st.session_state["agent_trace"] = result["trace"]

# ---------------------------------------------------------------------------
# -- Shared output box (Agents + RAG) --------------------------------------
# ---------------------------------------------------------------------------

# Display RAG latest response in shared expander
latest_response = st.session_state.get("latest_response")
with st.expander("Communication Output", expanded=True):
    rag_tab, agent_tab = st.tabs(["RAG Response", "Agent Response"])
    with rag_tab:
        if latest_response:
            render_response_output(latest_response["answer"], latest_response.get("response_type", "Markdown"), panel_key="latest_response")
            _retrieval = st.session_state.get("latest_retrieval")
            if _retrieval:
                citations = format_source_citations(_retrieval, response_type=latest_response.get("response_type", "Markdown"))
                if citations:
                    render_source_citations(citations, latest_response.get("response_type", "Markdown"))
        else:
            st.info("Submit a question in the RAG Query section above to see results here.")
    with agent_tab:
        agent_response = st.session_state.get("agent_response")
        agent_publication = st.session_state.get("agent_publication")
        agent_publication_filepath = st.session_state.get("agent_publication_filepath")
        agent_trace = st.session_state.get("agent_trace")
        if st.button("Refresh Agent Response", key="refresh_agent_response_btn", use_container_width=True):
            st.rerun()
        if agent_response or agent_publication or agent_publication_filepath:
            # If a publication file exists, offer to view it inline
            # Prefer explicit filepath stored in session state
            filepath = None
            if agent_publication_filepath:
                filepath = agent_publication_filepath
            elif agent_publication:
                try:
                    pub_data = json.loads(agent_publication) if isinstance(agent_publication, str) else agent_publication
                    pub_info = pub_data.get("publish", pub_data)
                    filepath = pub_info.get("filepath", "")
                except (json.JSONDecodeError, KeyError, TypeError):
                    filepath = None

            if filepath:
                if Path(filepath).exists():
                    with st.expander("Published Report Preview", expanded=True):
                        # Link to open in browser
                        st.markdown(f"**Published:** [`{filepath}`](file://{filepath})")
                        # Read and render HTML inline
                        try:
                            html_content = Path(filepath).read_text(encoding="utf-8")
                            components.html(html_content, height=800, scrolling=True)
                        except Exception:
                            st.info(f"Report saved at {filepath} (cannot render inline).")
                            # Download button (serves bytes, MIME-aware)
                            try:
                                file_bytes = Path(filepath).read_bytes()
                                fname = Path(filepath).name
                                lower = fname.lower()
                                if lower.endswith(".pdf"):
                                    mime = "application/pdf"
                                elif lower.endswith(".html") or lower.endswith(".htm"):
                                    mime = "text/html"
                                else:
                                    mime = "application/octet-stream"
                                st.download_button("Download report", data=file_bytes, file_name=fname, mime=mime)
                            except Exception:
                                pass
                else:
                    st.info(f"Report saved at {filepath}.")
            elif agent_publication:
                st.caption("Publication Result:")
                st.code(agent_publication, language="json")
            # Agent debug panel
            agent_debug = st.session_state.get("agent_response_debug")
            if agent_debug:
                with st.expander("Agent Debug (raw)", expanded=False):
                    try:
                        st.subheader("Trace")
                        st.code(agent_debug.get("trace") or "(no trace)")
                        st.subheader("Stdout / Stderr")
                        st.code((agent_debug.get("stdout") or "") + "\n" + (agent_debug.get("stderr") or ""))
                        st.subheader("Raw History")
                        st.json(agent_debug.get("raw_history") or {})
                    except Exception:
                        st.write(agent_debug)
            # Show raw agent response in a collapsible section
            if agent_response:
                with st.expander("Raw Agent Response", expanded=not bool(agent_publication)):
                    render_response_output(agent_response, "Markdown", panel_key="agent_response_tab")
            if agent_trace:
                with st.expander("LLM Trace", expanded=False):
                    st.text(agent_trace)
        else:
            st.info("Run an agent in the Agent Workflows section above to see results here.")
