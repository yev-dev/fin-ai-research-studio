import os
import json

import streamlit as st

from dashboard import DEFAULT_GITHUB_MODEL, DEFAULT_DEEPSEEK_MODEL
from fin_ai.core.providers import list_models
from fin_ai.core.request import create_llm_client, RequestPayload, known_providers
from fin_ai.core.response import ResponseFactory
from fin_ai.core.tools import (
    YAHOO_FINANCE_TOOLS,
    execute_litellm_tool_call,
    extract_tool_calls,
    build_tool_aware_system_prompt,
)


st.set_page_config(page_title="LiteLLM Chat", layout="wide")
st.title("LiteLLM Provider Chat")
st.caption("Query providers using fin_ai.core request/response wrappers.")

provider_label_to_key = {cfg.label: name for name, cfg in known_providers().items()}

with st.sidebar:
    st.header("Settings")
    default_provider = os.getenv("DEFAULT_PROVIDER", "ollama").strip().lower()
    provider_labels = list(provider_label_to_key.keys())
    default_provider_index = next(
        (i for i, k in enumerate(provider_labels) if provider_label_to_key[k] == default_provider),
        0,
    )
    provider_label = st.selectbox("Provider", provider_labels, index=default_provider_index)
    provider = provider_label_to_key[provider_label]

    response_format = st.selectbox(
        "Response Format",
        ResponseFactory.available(),
        index=ResponseFactory.available().index("markdown") if "markdown" in ResponseFactory.available() else 0,
        help="Rendering style implemented by fin_ai.core.response wrappers.",
    )

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.2, step=0.1)
    max_tokens = st.number_input("Max Tokens (optional)", min_value=0, value=0, step=32)
    auto_truncate_prompt = st.checkbox(
        "Auto-truncate prompt for strict-input models (e.g. gpt-5)",
        value=True,
        help="When enabled, oversized prompts are clipped before sending to strict-input models like gpt-5.",
    )
    # Proxy port is no longer part of the UI – removed.

    # ---- Dynamic params + model listing based on selected provider ----
    _pcfg = known_providers().get(provider)

    if "api_key" in (_pcfg.required_params if _pcfg else ()):
        _default_val = {"github": os.getenv("GITHUB_TOKEN", ""), "deepseek": os.getenv("DEEPSEEK_TOKEN", "")}.get(provider, "")
        st.text_input("API Key / Token", value=_default_val, type="password", key=f"{provider}_api_key")

    if _pcfg and _pcfg.optional_params:
        # Proxy provider names removed – use direct connection only
        _default_endpoint = {
            "ollama": os.getenv("OLLAMA_ENDPOINT", _pcfg.default_base_url),
            "github": os.getenv("GITHUB_ENDPOINT", _pcfg.default_base_url),
            "deepseek": os.getenv("DEEPSEEK_BASE_URL", _pcfg.default_base_url),
        }.get(provider, _pcfg.default_base_url)
        st.text_input("API Base URL", value=_default_endpoint, key=f"{provider}_api_base")

    # --- Model selection ---
    if provider == "github":
        _token_for_list = st.session_state.get("github_api_key", os.getenv("GITHUB_TOKEN", ""))
        with st.spinner("Fetching available GitHub models..."):
            gh_models = list_models("github", api_key=_token_for_list)
        gh_model_ids = [m.id for m in gh_models]
        if gh_model_ids:
            default_gh = os.getenv("GITHUB_MODEL", DEFAULT_GITHUB_MODEL)
            default_gh_idx = gh_model_ids.index(default_gh) if default_gh in gh_model_ids else 0
            st.selectbox("Model", gh_model_ids, index=default_gh_idx, key="github_model")
        else:
            st.text_input("Model", value=os.getenv("GITHUB_MODEL", DEFAULT_GITHUB_MODEL), key="github_model")
            st.caption("No models found. Check your token or enter a model name manually.")
    elif provider == "proxied_github":
        st.text_input("Model", value=os.getenv("GITHUB_MODEL", DEFAULT_GITHUB_MODEL), key="proxied_github_model")
    elif provider == "deepseek":
        _ds_token = os.getenv("DEEPSEEK_TOKEN", "")
        with st.spinner("Fetching available DeepSeek models..."):
            ds_models = list_models("deepseek", api_key=_ds_token)
        ds_model_ids = [m.id for m in ds_models]
        if ds_model_ids:
            default_ds = os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
            default_ds_idx = ds_model_ids.index(default_ds) if default_ds in ds_model_ids else 0
            st.selectbox("Model", ds_model_ids, index=default_ds_idx, key="deepseek_model")
        else:
            st.text_input("Model", value=os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL), key="deepseek_model")
            st.caption("No models found. Check your token or enter a model name manually.")
    elif provider == "proxied_deepseek":
        st.text_input("Model", value=os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL), key="proxied_deepseek_model")
    else:  # ollama
        _ol_endpoint = st.session_state.get("ollama_api_base", os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"))
        with st.spinner("Fetching available Ollama models..."):
            ol_models = list_models("ollama", base_url=_ol_endpoint)
        ol_model_ids = [m.id for m in ol_models]
        if ol_model_ids:
            default_ol = os.getenv("OLLAMA_MODEL", "llama3.1")
            default_ol_idx = ol_model_ids.index(default_ol) if default_ol in ol_model_ids else 0
            st.selectbox("Model", ol_model_ids, index=default_ol_idx, key="ollama_model")
        else:
            st.text_input("Model", value=os.getenv("OLLAMA_MODEL", "llama3.1"), key="ollama_model")
            st.caption("No models found. Is Ollama running?")

system_prompt = st.text_area(
    "System Prompt",
    value="You are a helpful financial analysis assistant.",
    height=100,
)
user_prompt = st.text_area(
    "User Prompt",
    value="Use tools to summarize company info and latest analyst recommendations for AAPL.",
    height=140,
)

effective_system_prompt = build_tool_aware_system_prompt(system_prompt)
with st.expander("Effective System Prompt (with tool guidance)", expanded=False):
    st.code(effective_system_prompt)

if st.button("Send", type="primary"):
    if not user_prompt.strip():
        st.error("User Prompt cannot be empty.")
    else:
        # Build client params from session state
        _api_key = None
        if provider == "github":
            _api_key = st.session_state.get("github_token", "") or None
        elif provider == "deepseek":
            _api_key = os.getenv("DEEPSEEK_TOKEN", "") or None

        raw_proxy_port = st.session_state.get("proxy_port_input", "").strip()
        proxy_port = None
        if raw_proxy_port:
            try:
                proxy_port = int(raw_proxy_port)
            except ValueError:
                st.error("Proxy Port must be a valid integer.")
                st.stop()

        payload = RequestPayload(
            prompt=user_prompt.strip(),
            system_prompt=effective_system_prompt,
            temperature=float(temperature),
            max_tokens=int(max_tokens) if max_tokens > 0 else None,
            proxy_port=proxy_port,
            auto_truncate_prompt=bool(auto_truncate_prompt),
            tools=YAHOO_FINANCE_TOOLS,
        )

        try:
            client = create_llm_client(
                provider=provider,
                api_key=_api_key,
                proxy_port=proxy_port,
            )
            response = client.send(payload, response_class=ResponseFactory.get(response_format))

            first_raw = response.get_metadata().raw_response
            first_message = first_raw.choices[0].message
            tool_calls = extract_tool_calls(first_message)

            if tool_calls:
                follow_up_messages = []
                if payload.system_prompt:
                    follow_up_messages.append({"role": "system", "content": payload.system_prompt})
                follow_up_messages.append({"role": "user", "content": payload.prompt})
                follow_up_messages.append(
                    {
                        "role": "assistant",
                        "content": first_message.content or "",
                        "tool_calls": [
                            {
                                "id": tool_call["id"],
                                "type": "function",
                                "function": {
                                    "name": tool_call["name"],
                                    "arguments": tool_call["arguments_text"],
                                },
                            }
                            for tool_call in tool_calls
                        ],
                    }
                )

                for tool_call in tool_calls:
                    tool_result = execute_litellm_tool_call(tool_call["name"], tool_call["arguments"])
                    follow_up_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": tool_call["name"] or "unknown_tool",
                            "content": json.dumps(tool_result),
                        }
                    )

                follow_up_payload = RequestPayload(
                    prompt=payload.prompt,
                    system_prompt=payload.system_prompt,
                    temperature=payload.temperature,
                    max_tokens=payload.max_tokens,
                    proxy_port=payload.proxy_port,
                    auto_truncate_prompt=payload.auto_truncate_prompt,
                    tools=YAHOO_FINANCE_TOOLS,
                    messages=follow_up_messages,
                )
                response = client.send(
                    follow_up_payload,
                    response_class=ResponseFactory.get(response_format),
                )
        except Exception as exc:
            st.exception(exc)
        else:
            st.subheader("Response")
            if response_format == "markdown":
                st.markdown(response.render())
            else:
                st.text(response.render())

            metadata = response.get_metadata()
            if metadata.prompt_truncated:
                before = metadata.prompt_tokens_before_guard
                after = metadata.prompt_tokens_after_guard
                st.warning(
                    f"Prompt was truncated to fit gpt-5 input limits"
                    f" ({before} -> {after} tokens before sending)."
                )

            from dataclasses import asdict as _asdict
            metadata_dict = _asdict(metadata)
            metadata_dict.pop("raw_response", None)

            st.subheader("Metadata")
            st.json(metadata_dict)

            with st.expander("Raw Response", expanded=False):
                st.text(str(metadata.raw_response))
