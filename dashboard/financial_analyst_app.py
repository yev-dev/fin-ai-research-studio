# streamlit run dashboard/financial_analyst_app.py

"""
Financial Data Analysis — Streamlit presentation layer.

This module contains only Streamlit UI widgets, layout, and rendering.
All business logic and external system communication are imported directly
from ``fin_ai.core.processor``, ``fin_ai.core.rag``, ``fin_ai.core.request``,
``fin_ai.core.providers``, and ``fin_ai.agents.engine_bridge``.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("LITELLM_DISABLE_COST_MAP", "true")
os.environ.setdefault("LITELLM_LOG_LEVEL", "ERROR")

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
import streamlit.components.v1 as components

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
    sanitize_generated_python_code,
    render_pdf_pages,
    render_csv_thumbnail,
)
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
from fin_ai.core.rag import load_embedding_metadata, RAGSourceStore, discover_vector_stores_by_source
from fin_ai.core.request import known_providers, get_provider_config
from fin_ai.core.embeddings import create_embeddings
from fin_ai.agents.agent_library import library as agent_library
from fin_ai.core.tools import YAHOO_FINANCE_TOOLS

st.set_page_config(page_title="Financial Data Analysis", layout="wide")
SIDEBAR_PREVIEW_WIDTH = 320


# ---------------------------------------------------------------------------
# Rendering helpers (pure presentation)
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
            render_pdf_pages(str(source_path), str(images_folder), zoom=1.5),
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
        img = render_csv_thumbnail(str(source_path), str(images_folder))
        if img:
            st.sidebar.image(img, caption="CSV Preview", width=SIDEBAR_PREVIEW_WIDTH)
        else:
            st.sidebar.info("Could not generate CSV preview.")
    except Exception as e:
        st.sidebar.error(f"Error loading CSV preview: {str(e)}")


# ---------------------------------------------------------------------------
# Helper: looks like embedding model
# ---------------------------------------------------------------------------

def looks_like_embedding_model(model_id: str) -> bool:
    """Heuristic filter for embedding-capable model identifiers."""
    mid = (model_id or "").strip().lower()
    return "embedding" in mid or "embed" in mid


# ---------------------------------------------------------------------------
# -- Page layout
# ---------------------------------------------------------------------------

st.title("FinAI Research Studio")

# Discover vector stores
source_vector_stores = get_source_vector_stores()
vector_db_names = get_vector_db_names(source_vector_stores)

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
    if selected_provider == "github":
        github_endpoint = _base_val
    elif selected_provider == "deepseek":
        deepseek_base_url = _base_val

# --- Model selection ---
if selected_provider == "github":
    try:
        with st.spinner("Fetching available GitHub models..."):
            gh_models = fetch_models("github", api_key=github_token)
        display_model_options = [m.id for m in gh_models] or [os.getenv("GITHUB_MODEL", DEFAULT_GITHUB_MODEL)]
    except Exception:
        display_model_options = [os.getenv("GITHUB_MODEL", DEFAULT_GITHUB_MODEL)]
    default_chat_model = os.getenv("GITHUB_MODEL", DEFAULT_GITHUB_MODEL)
    default_chat_index = display_model_options.index(default_chat_model) if default_chat_model in display_model_options else 0
    selected_model = st.sidebar.selectbox("Select Model", display_model_options, index=default_chat_index, key="github_model")

elif selected_provider == "deepseek":
    try:
        with st.spinner("Fetching available DeepSeek models..."):
            ds_models = fetch_models("deepseek", api_key=deepseek_token)
        deepseek_model_ids = [m.id for m in ds_models]
    except Exception:
        deepseek_model_ids = []
    if deepseek_model_ids:
        default_ds = os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
        default_ds_idx = deepseek_model_ids.index(default_ds) if default_ds in deepseek_model_ids else 0
        selected_model = st.sidebar.selectbox("Select Model", deepseek_model_ids, index=default_ds_idx, key="deepseek_model")
    else:
        selected_model = st.sidebar.text_input("Model", value=os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL), key="deepseek_model_fallback")

else:  # ollama
    default_chat_model = os.getenv("OLLAMA_MODEL", DEFAULT_CHAT_MODEL)
    from dashboard.utils import load_local_model_options
    available_chat_models, available_embedding_models, model_load_error = load_local_model_options(
        OLLAMA_BASE_URL, DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL
    )
    try:
        default_chat_index = available_chat_models.index(default_chat_model) if default_chat_model in available_chat_models else 0
    except ValueError:
        default_chat_index = 0
    selected_model = st.sidebar.selectbox("Model", available_chat_models, index=default_chat_index, key="ollama_chat_model")

response_type = st.sidebar.selectbox("Select Response Type", ["Plain Text", "Markdown", "Python Code"], index=1, key="response_type")
auto_truncate_prompt = st.sidebar.checkbox("Auto-truncate prompt (gpt-5 guard)", value=True)
use_tools = st.sidebar.checkbox("Enable function tools (financial data)", value=False)

if use_tools:
    with st.sidebar.expander("Available tools", expanded=False):
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
    embedding_github_token = st.sidebar.text_input("GitHub Token (Embeddings)", value=os.getenv("GITHUB_TOKEN", ""), type="password", key="embedding_github_token")
    try:
        emb_models = fetch_models("github", api_key=embedding_github_token)
        embedding_model_ids = [m.id for m in emb_models if looks_like_embedding_model(m.id)]
    except Exception:
        embedding_model_ids = []
    available_embedding_models_display = embedding_model_ids if embedding_model_ids else [DEFAULT_GITHUB_EMBEDDING_MODEL, "openai/text-embedding-3-large"]
    if not embedding_model_ids:
        st.sidebar.warning(
            "Using safe defaults to avoid 400 errors from /embeddings."
        )
    default_emb_model_idx = available_embedding_models_display.index(DEFAULT_GITHUB_EMBEDDING_MODEL) if DEFAULT_GITHUB_EMBEDDING_MODEL in available_embedding_models_display else 0
    embeddings_base_url = os.getenv("GITHUB_EMBEDDING_BASE_URL", GITHUB_EMBEDDING_BASE_URL)
else:
    ollama_emb_endpoint = st.sidebar.text_input("Ollama Endpoint (Embeddings)", value=os.getenv("OLLAMA_ENDPOINT", OLLAMA_BASE_URL), key="embedding_ollama_endpoint")
    available_embedding_models_display = available_embedding_models
    default_emb_model_idx = available_embedding_models_display.index(DEFAULT_EMBEDDING_MODEL) if DEFAULT_EMBEDDING_MODEL in available_embedding_models_display else 0
    embeddings_base_url = ollama_emb_endpoint

selected_embedding_model = st.sidebar.selectbox("Select Embedding Model", available_embedding_models_display, index=default_emb_model_idx, key="embedding_model")

st.sidebar.divider()

# -- Previous Questions -----------------------------------------------------
if vector_db_names:
    history, _history_db_name = load_history(vector_db_names[0]), vector_db_names[0]
    st.session_state["question_history"] = history
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
        rag_store = RAGSourceStore()
        on_disk_stores = list(discover_vector_stores_by_source(VECTOR_DB_DIR).keys())

        st.sidebar.caption("Registered RAG Sources")
        df_sources = rag_store.to_dataframe()
        if not df_sources.empty and on_disk_stores:
            # Only show sources with valid source types (pdf, csv, json, html, url, etc.)
            valid_types = set(SUPPORTED_UPLOAD_TYPES)
            df_on_disk = df_sources[df_sources["name"].isin(on_disk_stores)].copy()
            if not df_on_disk.empty:
                df_filtered = df_on_disk[df_on_disk["source_type"].isin(valid_types)]
                if not df_filtered.empty:
                    st.sidebar.dataframe(
                        df_filtered[["name", "source_type", "chunk_count", "embedding_model"]],
                        use_container_width=True,
                        hide_index=True,
                    )
                    total = len(df_filtered)
                    total_chunks = df_filtered["chunk_count"].sum()
                    st.sidebar.caption(f"{total} FAISS index(es) · {int(total_chunks):,} chunk(s)")
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

agent_rag_query = st.text_area(
    "Agent Task Prompt",
    placeholder='e.g. "Analyse NVDA financials and competitive position"',
    key="agent_prompt_main",
    height=80,
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
query_source_configs = []
loaded_stores: dict = {}

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
        saved_emb_meta = load_embedding_metadata(vector_db_names[0])
        if saved_emb_meta:
            embeddings = create_embeddings(
                provider=saved_emb_meta["provider"],
                model=saved_emb_meta["model"],
                api_base=saved_emb_meta["base_url"],
                api_key=_github_token if saved_emb_meta["provider"] == "github" else None,
            )
        else:
            embeddings = create_embeddings(
                provider=selected_emb_provider,
                model=selected_embedding_model,
                api_base=embeddings_base_url,
                api_key=_github_token if selected_emb_provider == "github" else None,
            )
    except Exception:
        embeddings = None

    if embeddings is None:
        st.sidebar.error("No embeddings available — check the sidebar for errors.")
    else:
        try:
            loaded_stores = load_vector_stores_for_query(
                selected_query_vector_dbs, source_vector_stores, embeddings
            )
            query_source_configs = build_query_source_configs(loaded_stores, group_by="vector_db")
            available_source_names = [c.name for c in query_source_configs]
            source_groups_map = discover_source_groups(loaded_stores)
            available_source_groups = sorted(source_groups_map.keys())
        except (ValueError, RuntimeError) as e:
            st.sidebar.error(str(e))
            loaded_stores = {}
            query_source_configs = []
            available_source_names = []
            available_source_groups = []

st.subheader("RAG Query")

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
        selected_source_groups = st.multiselect(
            "Source Groups",
            available_source_groups,
            default=(
                _selected_source_groups
                if _selected_source_groups
                else available_source_groups
            ),
            key="query_source_groups",
            help="Select document type(s) to search within.",
        )
        # Filter available documents to only those in selected groups
        group_filtered_stores = filter_stores_by_source_groups(loaded_stores, selected_source_groups)
        filtered_doc_names = sorted(group_filtered_stores.keys())
    else:
        selected_source_groups = []
        filtered_doc_names = available_source_names

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

    # -- Submit and answer ------------------------------------------------------
    if submit_clicked and question:
        if not query_source_configs:
            st.error("No vector database sources are available. Please load vector databases first.")
        elif not selected_query_vector_dbs:
            st.error("Select at least one document before submitting a question.")
        else:
            active_configs = [c for c in query_source_configs if c.name in selected_query_vector_dbs]
            if not active_configs:
                st.error("The selected documents are not available. Please re-select.")
            else:
                with st.spinner("Answering your question..."):
                    result = answer_question(
                        question,
                        active_configs,
                        provider=selected_provider,
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
                    st.session_state["question_history"] = load_history(vector_db_names[0])

# ---------------------------------------------------------------------------
# -- Agent execution & display ---------------------------------------------
# ---------------------------------------------------------------------------

if agent_submit and agent_rag_query.strip():
    with st.spinner(f"Running {selected_agent} agent..."):
        _effective_gh_token = github_token if selected_provider == "github" else ""
        _effective_ds_token = deepseek_token if selected_provider == "deepseek" else ""
        agent_llm_config = build_agent_llm_config(
            provider=selected_provider,
            model=selected_model,
            ollama_base_url=OLLAMA_BASE_URL,
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
        )
        if result["success"]:
            st.session_state["agent_response"] = result["response"]
            if result.get("publication"):
                st.session_state["agent_publication"] = result["publication"]
        else:
            st.session_state["agent_response"] = f"Agent error: {result['error']}"

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
        if agent_response or agent_publication:
            # If a publication file exists, offer to view it inline
            if agent_publication:
                try:
                    pub_data = json.loads(agent_publication) if isinstance(agent_publication, str) else agent_publication
                    pub_info = pub_data.get("publish", pub_data)
                    filepath = pub_info.get("filepath", "")
                    if filepath and Path(filepath).exists():
                        with st.expander("Published Report Preview", expanded=True):
                            # Link to open in browser
                            st.markdown(f"**Published:** [`{filepath}`](file://{filepath})")
                            # Read and render HTML inline
                            html_content = Path(filepath).read_text(encoding="utf-8")
                            components.html(html_content, height=800, scrolling=True)
                    elif filepath:
                        st.info(f"Report saved at `{filepath}`.")
                except (json.JSONDecodeError, KeyError, TypeError):
                    st.caption("Publication Result:")
                    st.code(agent_publication, language="json")
            # Show raw agent response in a collapsible section
            if agent_response:
                with st.expander("Raw Agent Response", expanded=not bool(agent_publication)):
                    render_response_output(agent_response, "Markdown", panel_key="agent_response_tab")
        else:
            st.info("Run an agent in the Agent Workflows section above to see results here.")