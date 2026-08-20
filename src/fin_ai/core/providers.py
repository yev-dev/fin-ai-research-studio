"""Provider model listing utilities.

Provides a unified interface to list available models from different providers
(Ollama, GitHub Models) via their respective APIs. Each provider's listing
function is registered in a central registry so that callers can simply pass a
provider name and receive a list of available models.

Usage:
    from fin_ai.core.providers import list_models, available_providers

    models = list_models("ollama")
    models = list_models("github", api_key="ghp_...")

    for m in models:
        print(m.id, m.name)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Describes a single model available from a provider."""

    id: str
    name: str
    provider: str


# ---------------------------------------------------------------------------
# Provider-specific listing implementations
# ---------------------------------------------------------------------------


def list_ollama_models(
    base_url: str | None = None,
    **kwargs: Any,
) -> list[ModelInfo]:
    """List available models from a local Ollama instance.

    Uses Ollama's ``/api/tags`` endpoint to retrieve the list of pulled models.
    """
    endpoint = (base_url or os.getenv("OLLAMA_ENDPOINT", "http://127.0.0.1:11434")).rstrip("/")
    url = f"{endpoint}/api/tags"

    try:
        req = Request(url, headers={"User-Agent": "fin-ai/1.0"})
        with urlopen(req, timeout=15) as response:
            payload = json.load(response)
    except URLError as exc:
        logger.warning("Failed to list Ollama models from %s: %s", url, exc.reason)
        return []
    except Exception as exc:
        logger.warning("Failed to list Ollama models from %s: %s", url, exc)
        return []

    models = payload.get("models", [])
    if not models:
        logger.info("Ollama returned an empty model list at %s", url)
        return []

    return [
        ModelInfo(
            id=model.get("name", ""),
            name=model.get("name", ""),
            provider="ollama",
        )
        for model in models
        if model.get("name")
    ]


def list_github_models(
    api_key: str | None = None,
    **kwargs: Any,
) -> list[ModelInfo]:
    """List available models from the GitHub Models catalog.

    Fetches the official model catalog at
    ``https://models.github.ai/catalog/models`` (or a user-provided base URL)
    and parses model IDs.
    """
    token = api_key or os.getenv("GITHUB_TOKEN", "")
    # Allow callers to override the base URL (e.g. enterprise/proxy endpoints).
    # Accept either `base_url` or `api_base` as the kwarg for compatibility.
    base = kwargs.get("base_url") or kwargs.get("api_base") or os.getenv("GITHUB_ENDPOINT", "https://models.github.ai/inference")
    base = str(base).rstrip("/")
    # Common GitHub-hosted catalog lives under /catalog/models. If the provided
    # base looks like an inference endpoint, adapt it to the catalog path.
    if base.endswith("/inference"):
        url = base[:-len("/inference")] + "/catalog/models"
    elif base.endswith("/api") or base.endswith("/v1"):
        url = base + "/catalog/models"
    else:
        # Default fallback: append the well-known catalog path
        url = base + "/catalog/models"
    headers: dict[str, str] = {"User-Agent": "fin-ai/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Try multiple candidate endpoints to be tolerant of proxied or enterprise
    # deployments. Accept whichever returns a JSON list or a dict containing
    # a list under common keys ('models', 'data').
    candidate_paths = [
        url,
        base + "/catalog/models",
        base + "/models",
        base + "/v1/models",
        base + "/api/models",
    ]

    payload = None
    last_exc: Exception | None = None
    for candidate in candidate_paths:
        try:
            req = Request(candidate, headers=headers)
            with urlopen(req, timeout=15) as response:
                payload = json.load(response)
            if payload is not None:
                logger.debug("GitHub model list fetched from %s", candidate)
                break
        except URLError as exc:
            last_exc = exc
            logger.debug("Failed to fetch GitHub models from %s: %s", candidate, exc)
            continue
        except Exception as exc:
            last_exc = exc
            logger.debug("Failed to fetch GitHub models from %s: %s", candidate, exc)
            continue

    if payload is None:
        logger.warning("Could not fetch GitHub models from any candidate endpoint. Last error: %s", last_exc)
        # As a pragmatic fallback return the environment default so the UI
        # remains usable and the user can still proceed.
        default = os.getenv("GITHUB_MODEL", "openai/gpt-4o")
        return [ModelInfo(id=default, name=default, provider="github")]

    # Normalize payload to a list of model dicts or ids
    models_list = []
    if isinstance(payload, list):
        models_list = payload
    elif isinstance(payload, dict):
        # Try common container keys
        if "models" in payload and isinstance(payload["models"], list):
            models_list = payload["models"]
        elif "data" in payload and isinstance(payload["data"], list):
            models_list = payload["data"]
        else:
            # Unexpected dict shape — attempt to extract ids if present
            models_list = [payload]

    # Extract ids from items that may be dicts or strings
    model_ids = sorted(
        {
            (item.get("id") if isinstance(item, dict) else str(item)).strip()
            for item in models_list
            if (isinstance(item, dict) and item.get("id")) or (isinstance(item, str) and item)
        }
    )

    if not model_ids:
        logger.warning("GitHub model catalog returned no model ids; using environment default.")
        default = os.getenv("GITHUB_MODEL", "openai/gpt-4o")
        return [ModelInfo(id=default, name=default, provider="github")]

    return [ModelInfo(id=mid, name=mid, provider="github") for mid in model_ids]


DEEPSEEK_KNOWN_MODELS = [
    "deepseek-chat",       # DeepSeek-V3
    "deepseek-reasoner",   # DeepSeek-R1
]


def list_deepseek_models(
    api_key: str | None = None,
    **kwargs: Any,
) -> list[ModelInfo]:
    """List available DeepSeek models.

    Attempts to fetch models from DeepSeek's OpenAI-compatible ``/v1/models``
    endpoint.  Falls back to a hardcoded list of known models on error.
    """
    token = api_key or os.getenv("DEEPSEEK_TOKEN") or os.getenv("DEEPSEAK_TOKEN", "")
    base_url = kwargs.get("base_url") or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    url = f"{base_url.rstrip('/')}/models"
    headers: dict[str, str] = {"User-Agent": "fin-ai/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=15) as response:
            payload = json.load(response)
        data = payload.get("data") if isinstance(payload, dict) else payload
        if isinstance(data, list):
            model_ids = sorted(
                {
                    str(item.get("id", "")).strip()
                    for item in data
                    if isinstance(item, dict) and item.get("id")
                }
            )
            if model_ids:
                return [
                    ModelInfo(id=mid, name=mid, provider="deepseek")
                    for mid in model_ids
                ]
    except Exception as exc:
        logger.debug("Could not fetch DeepSeek model list from %s: %s", url, exc)

    # Fallback to known models
    return [
        ModelInfo(id=mid, name=mid, provider="deepseek")
        for mid in DEEPSEEK_KNOWN_MODELS
    ]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PROVIDER_MODEL_REGISTRY: dict[str, Callable[..., list[ModelInfo]]] = {
    "ollama": list_ollama_models,
    "github": list_github_models,
    "deepseek": list_deepseek_models,
}


def register_provider(provider: str, listing_fn: Callable[..., list[ModelInfo]]) -> None:
    """Register a custom provider model-listing function.

    Args:
        provider: Provider name (e.g. ``"azure"``, ``"anthropic"``).
        listing_fn: A callable that accepts keyword arguments and returns a
            list of ``ModelInfo``.
    """
    if not callable(listing_fn):
        raise TypeError(f"listing_fn must be callable, got {type(listing_fn)}")
    _PROVIDER_MODEL_REGISTRY[provider] = listing_fn


def list_models(provider: str, **kwargs: Any) -> list[ModelInfo]:
    """List available models for a given provider.

    The function dispatches to the registered provider-specific listing
    implementation.  Extra keyword arguments are forwarded to that
    implementation (e.g. ``api_key`` for GitHub, ``base_url`` for Ollama).

    Args:
        provider: Provider name (``"ollama"``, ``"github"``, or a custom name
            added via :func:`register_provider`).
        **kwargs: Forwarded to the provider's listing function.

    Returns:
        List of :class:`ModelInfo` instances.

    Raises:
        ValueError: If the provider is not registered.
    """
    try:
        listing_fn = _PROVIDER_MODEL_REGISTRY[provider]
    except KeyError:
        available = ", ".join(sorted(_PROVIDER_MODEL_REGISTRY))
        raise ValueError(
            f"Unknown provider {provider!r}. Available: {available}."
        )

    try:
        return listing_fn(**kwargs)
    except Exception:
        logger.exception("Provider model listing crashed for %r", provider)
        return []


def available_providers() -> list[str]:
    """Return sorted list of registered provider names."""
    return sorted(_PROVIDER_MODEL_REGISTRY)
