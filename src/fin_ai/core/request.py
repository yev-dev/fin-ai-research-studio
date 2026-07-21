"""
LLM request handling — provider configs, LiteLLM client, and factory.

Every provider has a ``ProviderConfig`` that declares:
- Required parameters (raised on missing)
- Optional parameters (used when provided)
- How to construct the LiteLLM model string and ``api_base`` URL

The factory :func:`create_llm_client` builds a ready-to-use :class:`LiteLLMClient`
from a provider name and explicit keyword arguments — no implicit env-var
reading outside of sensible defaults.

Usage::

    from fin_ai.core.request import create_llm_client, ProviderConfig, known_providers

    # Build a client for any provider with explicit params
    client = create_llm_client(
        provider="ollama",
        model="llama3.1",
        api_base="http://localhost:11434",
    )
    # Proxied GitHub — only needs proxy ports, no token
    client = create_llm_client(
        provider="proxied_github",
        model="openai/gpt-4o",
        http_proxy_port=8080,
        https_proxy_port=8443,
    )
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Any, Type

import litellm
import tiktoken
from litellm import completion
from fin_ai.core.response import ConsoleModelResponse, ModelResponse, Provider, ResponseFactory, ResponseMetadata

litellm.drop_params = True

MODEL_INPUT_TOKEN_LIMITS = {
    "gpt-5": 4000,
    "gpt-4.1-mini": 8000,
}

MODEL_SAFE_INPUT_BUDGETS = {
    "gpt-5": 3600,
    "gpt-4.1-mini": 7200,
}

# ---------------------------------------------------------------------------
# Provider configuration — declares what each provider needs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderConfig:
    """Declares the params a provider requires and how to build LiteLLM arguments.

    Each registered provider has one instance that documents its interface.
    """

    #: Human-readable label (shown in UI dropdowns etc.)
    label: str

    #: Parameter names that the caller MUST provide.
    required_params: tuple[str, ...] = ()

    #: Parameter names that the caller MAY provide (e.g. token, proxy port).
    optional_params: tuple[str, ...] = ()

    #: Default ``api_base`` URL (used when none is supplied).
    default_base_url: str = ""

    #: How to construct the LiteLLM model string given the ``model`` kwarg.
    #: ``"direct"`` = use the model string as-is; ``"ollama/{model}"`` = prepend prefix.
    model_format: str = "direct"  # "direct" | "ollama/{model}" | "deepseek/{model}"

    def build_model_string(self, model: str) -> str:
        if self.model_format == "direct":
            return model
        return self.model_format.format(model=model)

    def build_api_base(self, api_base: str | None) -> str:
        return api_base or self.default_base_url


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_PROVIDER_CONFIGS: dict[str, ProviderConfig] = {
    "ollama": ProviderConfig(
        label="Local Ollama",
        optional_params=("api_base", "proxy_port"),
        default_base_url="http://192.168.1.188:11434",
        model_format="ollama/{model}",
    ),
    "github": ProviderConfig(
        label="GitHub Models",
        required_params=("api_key",),
        optional_params=("api_base", "proxy_port"),
        default_base_url="https://models.github.ai/inference",
        model_format="direct",
    ),
    "deepseek": ProviderConfig(
        label="DeepSeek Models",
        required_params=("api_key",),
        optional_params=("api_base", "proxy_port"),
        default_base_url="https://api.deepseek.com/v1",
        model_format="deepseek/{model}",
    ),
    "proxied_github": ProviderConfig(
        label="GitHub Models (Proxy)",
        required_params=(),
        optional_params=("api_base", "http_proxy_port", "https_proxy_port"),
        default_base_url="https://models.github.ai/inference",
        model_format="direct",
    ),
    "proxied_deepseek": ProviderConfig(
        label="DeepSeek Models (Proxy)",
        required_params=(),
        optional_params=("api_base", "http_proxy_port", "https_proxy_port"),
        default_base_url="https://api.deepseek.com/v1",
        model_format="deepseek/{model}",
    ),
}


def get_provider_config(provider: str) -> ProviderConfig:
    """Return the :class:`ProviderConfig` for *provider*, or raise ``ValueError``."""
    try:
        return _PROVIDER_CONFIGS[provider]
    except KeyError:
        available = ", ".join(sorted(_PROVIDER_CONFIGS))
        raise ValueError(f"Unknown provider {provider!r}. Available: {available}.")


def known_providers() -> dict[str, ProviderConfig]:
    """Return a copy of the provider config registry."""
    return dict(_PROVIDER_CONFIGS)


def register_provider_config(provider: str, config: ProviderConfig) -> None:
    """Register (or override) a provider config."""
    _PROVIDER_CONFIGS[provider] = config


# ---------------------------------------------------------------------------
# Model name resolution (kept for backward compat, delegates to config)
# ---------------------------------------------------------------------------


def resolve_model_name(provider: Provider) -> str:
    """Resolve the LiteLLM model string for *provider*.

    Uses legacy env-var fallbacks for backward compatibility. Prefer using
    :func:`create_llm_client` with an explicit ``model`` argument.
    """
    cfg = get_provider_config(provider)
    if provider == "ollama":
        return cfg.build_model_string(os.getenv("OLLAMA_MODEL", "llama3.2"))
    if provider == "github":
        return os.getenv("GITHUB_MODEL", "openai/gpt-4o")
    if provider == "deepseek":
        return cfg.build_model_string(os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    if provider.startswith("proxied_"):
        base = provider.removeprefix("proxied_")
        env_key = f"{base.upper()}_MODEL"
        return os.getenv(env_key, "openai/gpt-4o" if "github" in provider else "deepseek-chat")
    raise ValueError(f"Unknown provider {provider!r}.")


def get_model_input_token_limit(model: str) -> int | None:
    normalized = model.lower()
    for model_key, limit in MODEL_INPUT_TOKEN_LIMITS.items():
        if model_key in normalized:
            return limit
    return None


def get_model_safe_input_budget(model: str) -> int | None:
    normalized = model.lower()
    for model_key, budget in MODEL_SAFE_INPUT_BUDGETS.items():
        if model_key in normalized:
            return budget
    return None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback heuristic: roughly 4 characters per token for English prose.
        return max(1, len(text) // 4)


def count_message_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(count_tokens(str(message.get("content", ""))) for message in messages)


def trim_text_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text

    left = 0
    right = len(text)
    best = ""

    while left <= right:
        mid = (left + right) // 2
        candidate = text[:mid]
        token_count = count_tokens(candidate)
        if token_count <= max_tokens:
            best = candidate
            left = mid + 1
        else:
            right = mid - 1

    return best.rstrip()


@dataclass(slots=True)
class RequestPayload:
    prompt: str
    system_prompt: str | None = "You are a helpful financial analysis assistant."
    temperature: float = 0.2
    max_tokens: int | None = None
    proxy_port: int | None = None
    http_proxy_port: int | None = None
    https_proxy_port: int | None = None
    auto_truncate_prompt: bool = True
    tools: list[dict[str, Any]] | None = None
    messages: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class PromptGuardResult:
    messages: list[dict[str, Any]]
    truncated: bool = False
    prompt_tokens_before: int | None = None
    prompt_tokens_after: int | None = None


class LiteLLMClient:
    """Low-level wrapper over LiteLLM completion calls.

    All provider configuration is baked in at construction time via
    :func:`create_llm_client`.  Callers only need to call :meth:`send`.
    """

    def __init__(
        self,
        provider: Provider,
        model: str,
        api_base: str,
        api_key: str | None = None,
        proxy_host: str | None = None,
        proxy_port: int | None = None,
        http_proxy_port: int | None = None,
        https_proxy_port: int | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.proxy_host = proxy_host or "127.0.0.1"
        self.proxy_port = proxy_port
        self.http_proxy_port = http_proxy_port
        self.https_proxy_port = https_proxy_port

    @contextmanager
    def _proxy_env(self, payload: RequestPayload):
        """Temporarily set proxy env vars for LiteLLM HTTP calls.

        Supports two modes:
        1. Single-port mode (``proxy_port``) — sets both HTTP_PROXY and HTTPS_PROXY.
        2. Split-port mode (``http_proxy_port`` + ``https_proxy_port``) — separate
           ports for HTTP and HTTPS proxies (used by ``proxied_github`` etc.).
        """
        http_port = (
            payload.http_proxy_port
            if payload.http_proxy_port is not None
            else self.http_proxy_port
        )
        https_port = (
            payload.https_proxy_port
            if payload.https_proxy_port is not None
            else self.https_proxy_port
        )
        single_port = (
            payload.proxy_port
            if payload.proxy_port is not None
            else self.proxy_port
        )

        if single_port is not None:
            http_port = https_port = single_port

        if http_port is None and https_port is None:
            yield
            return

        http_url = f"http://{self.proxy_host}:{http_port}" if http_port is not None else None
        https_url = f"https://{self.proxy_host}:{https_port}" if https_port is not None else None

        keys = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]
        previous = {key: os.environ.get(key) for key in keys}

        if http_url:
            os.environ["HTTP_PROXY"] = http_url
            os.environ["http_proxy"] = http_url
        if https_url:
            os.environ["HTTPS_PROXY"] = https_url
            os.environ["https_proxy"] = https_url

        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def send(
        self,
        request: RequestPayload,
        response_class: Type[ModelResponse] = ConsoleModelResponse,
    ) -> ModelResponse:
        if request.messages is not None:
            messages = list(request.messages)
        else:
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})

        if request.auto_truncate_prompt:
            guard = _apply_model_input_guard(messages, self.model)
        else:
            current_tokens = count_message_tokens(messages)
            guard = PromptGuardResult(
                messages=messages,
                truncated=False,
                prompt_tokens_before=current_tokens,
                prompt_tokens_after=current_tokens,
            )
        messages = guard.messages

        call_args: dict[str, Any] = {
            "model": self.model,
            "api_base": self.api_base,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.tools is not None:
            call_args["tools"] = request.tools
        if self.api_key is not None:
            call_args["api_key"] = self.api_key
        if request.max_tokens is not None:
            call_args["max_tokens"] = request.max_tokens

        with self._proxy_env(request):
            raw = completion(**call_args)
        usage = getattr(raw, "usage", None)
        choice = raw.choices[0]

        metadata = ResponseMetadata(
            provider=self.provider,
            model=self.model,
            api_base=self.api_base,
            finish_reason=getattr(choice, "finish_reason", None),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            prompt_truncated=guard.truncated,
            prompt_tokens_before_guard=guard.prompt_tokens_before,
            prompt_tokens_after_guard=guard.prompt_tokens_after,
            raw_response=raw,
        )
        return response_class(content=choice.message.content or "", metadata=metadata)


def _apply_model_input_guard(messages: list[dict[str, str]], model: str) -> PromptGuardResult:
    """Trim oversized prompts for strict-input models before API call."""
    current_tokens = count_message_tokens(messages)
    safe_budget = get_model_safe_input_budget(model)
    if safe_budget is None:
        return PromptGuardResult(messages=messages, prompt_tokens_before=current_tokens, prompt_tokens_after=current_tokens)

    if current_tokens <= safe_budget:
        return PromptGuardResult(messages=messages, prompt_tokens_before=current_tokens, prompt_tokens_after=current_tokens)

    adjusted = deepcopy(messages)
    user_indexes = [idx for idx, msg in enumerate(adjusted) if msg.get("role") == "user"]
    if not user_indexes:
        return PromptGuardResult(messages=adjusted, prompt_tokens_before=current_tokens, prompt_tokens_after=current_tokens)

    target_idx = user_indexes[-1]
    other_tokens = count_message_tokens(
        [msg for idx, msg in enumerate(adjusted) if idx != target_idx]
    )
    max_user_tokens = max(128, safe_budget - other_tokens)

    original_user_text = str(adjusted[target_idx].get("content", ""))
    trimmed_user_text = trim_text_to_tokens(original_user_text, max_user_tokens)
    truncated = trimmed_user_text != original_user_text
    if truncated:
        limit = get_model_input_token_limit(model)
        limit_note = f"{limit} token" if limit is not None else "model"
        trimmed_user_text += f"\n\n[Prompt truncated to fit {limit_note} input limit.]"
    adjusted[target_idx]["content"] = trimmed_user_text
    after_tokens = count_message_tokens(adjusted)
    return PromptGuardResult(
        messages=adjusted,
        truncated=truncated,
        prompt_tokens_before=current_tokens,
        prompt_tokens_after=after_tokens,
    )


class ModelRequest:
    """Consolidated wrapper for model requests across providers.

    Builds a :class:`LiteLLMClient` via :func:`create_llm_client` from
    a provider name and optional overrides.  Supports the legacy env-var
    pattern for backward compatibility.
    """

    def __init__(
        self,
        provider: Provider,
        format: str = "console",
        *,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        proxy_port: int | None = None,
        http_proxy_port: int | None = None,
        https_proxy_port: int | None = None,
    ) -> None:
        self.provider = provider
        self.response_class = ResponseFactory.get(format)
        cfg = get_provider_config(provider)
        resolved_model = model or resolve_model_name(provider)
        resolved_api_base = cfg.build_api_base(api_base)

        # Resolve API key from env vars when not explicitly provided
        if api_key is None:
            if provider == "github" or provider == "proxied_github":
                api_key = os.getenv("GITHUB_TOKEN")
            elif provider == "deepseek" or provider == "proxied_deepseek":
                api_key = os.getenv("DEEPSEEK_TOKEN")

        self.client = LiteLLMClient(
            provider=provider,
            model=resolved_model,
            api_base=resolved_api_base,
            api_key=api_key,
            proxy_port=proxy_port,
            http_proxy_port=http_proxy_port,
            https_proxy_port=https_proxy_port,
        )

    def request(self, payload: RequestPayload) -> ModelResponse:
        return self.client.send(payload, response_class=self.response_class)


# ---------------------------------------------------------------------------
# create_llm_client — the primary factory function
# ---------------------------------------------------------------------------


def create_llm_client(
    provider: str,
    *,
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    proxy_port: int | None = None,
    http_proxy_port: int | None = None,
    https_proxy_port: int | None = None,
) -> LiteLLMClient:
    """Create a fully-configured :class:`LiteLLMClient` for *provider*.

    All parameters are explicit — no env-var reading happens inside this
    function except for sensible defaults (e.g. Ollama's ``localhost:11434``
    as the default ``api_base``).

    Parameters
    ----------
    provider : str
        One of ``"ollama"``, ``"github"``, ``"deepseek"``,
        ``"proxied_github"``, ``"proxied_deepseek"``.
    model : str, optional
        Model identifier.  Uses ``ProviderConfig.build_model_string()`` to
        format it for LiteLLM (e.g. ``"ollama/llama3.1"`` for Ollama).
    api_base : str, optional
        API base URL.  Falls back to ``ProviderConfig.default_base_url``.
    api_key : str, optional
        API key / token.  Required for ``github`` and ``deepseek``.
    proxy_port : int, optional
        Single proxy port (applied to both HTTP and HTTPS).
    http_proxy_port : int, optional
        Separate HTTP proxy port (``proxied_*`` providers).
    https_proxy_port : int, optional
        Separate HTTPS proxy port (``proxied_*`` providers).

    Returns
    -------
    LiteLLMClient
        Ready-to-use client.

    Raises
    ------
    ValueError
        If *provider* is unknown.
    """
    cfg = get_provider_config(provider)
    resolved_model = model or resolve_model_name(provider)
    resolved_api_base = cfg.build_api_base(api_base)

    # Validate required params
    for param in cfg.required_params:
        value = locals().get(param)
        if value is None:
            raise ValueError(
                f"Provider {provider!r} requires '{param}' but none was provided."
            )

    return LiteLLMClient(
        provider=provider,
        model=resolved_model,
        api_base=resolved_api_base,
        api_key=api_key,
        proxy_port=proxy_port,
        http_proxy_port=http_proxy_port,
        https_proxy_port=https_proxy_port,
    )


# ---------------------------------------------------------------------------
# Legacy ModelRequestFactory (deprecated, kept for backward compat)
# ---------------------------------------------------------------------------


class ModelRequestFactory:
    """Legacy factory — kept for backward compatibility.

    Prefer :func:`create_llm_client` in new code.
    """

    @classmethod
    def create(cls, provider: str) -> LiteLLMClient:
        kwargs: dict[str, Any] = {}
        cfg = get_provider_config(provider)
        resolved_model = resolve_model_name(provider)
        resolved_api_base = cfg.build_api_base(None)

        if provider == "ollama":
            kwargs["api_base"] = _normalize_ollama_endpoint(
                os.getenv("OLLAMA_ENDPOINT", resolved_api_base)
            )
        elif provider == "github":
            token = os.getenv("GITHUB_TOKEN")
            if not token:
                raise ValueError("Missing GITHUB_TOKEN in environment.")
            kwargs["api_key"] = token
            kwargs["api_base"] = os.getenv("GITHUB_ENDPOINT", resolved_api_base)
        elif provider == "deepseek":
            token = os.getenv("DEEPSEEK_TOKEN")
            if not token:
                raise ValueError("Missing DEEPSEEK_TOKEN in environment.")
            kwargs["api_key"] = token
            kwargs["api_base"] = os.getenv("DEEPSEEK_BASE_URL", resolved_api_base)
        elif provider == "proxied_github":
            kwargs["api_base"] = os.getenv("GITHUB_ENDPOINT", resolved_api_base)
        elif provider == "proxied_deepseek":
            kwargs["api_base"] = os.getenv("DEEPSEEK_BASE_URL", resolved_api_base)
        else:
            raise ValueError(f"Unknown provider {provider!r}.")

        proxy_port = _read_proxy_port_from_env()
        return LiteLLMClient(
            provider=provider,
            model=resolved_model,
            **kwargs,
            proxy_port=proxy_port,
        )

    @classmethod
    def available(cls) -> list[str]:
        return sorted(_PROVIDER_CONFIGS)

    @classmethod
    def register(cls, name: str, builder: Any) -> None:
        raise NotImplementedError(
            "ModelRequestFactory.register is deprecated. "
            "Use register_provider_config() instead."
        )

    @classmethod
    def get(cls, provider: str) -> Any:
        cfg = get_provider_config(provider)
        return lambda: cls.create(provider)


def _normalize_ollama_endpoint(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1"):
        return endpoint[: -len("/v1")]
    return endpoint


def _read_proxy_port_from_env() -> int | None:
    """Read px proxy port from env, if configured."""
    raw = (os.getenv("PX_PROXY_PORT") or os.getenv("PROXY_PORT") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None
