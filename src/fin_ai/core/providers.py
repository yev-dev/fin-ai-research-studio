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
    endpoint = (base_url or os.getenv("OLLAMA_ENDPOINT", "http://192.168.1.141:11434")).rstrip("/")
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

    # Ollama responses may vary across versions. Try to extract model names
    # from common shapes: {'models': [...]}, {'tags': [...]}, or a plain list.
    candidates = []
    if isinstance(payload, dict):
        if "models" in payload and isinstance(payload["models"], list):
            candidates = payload["models"]
        elif "tags" in payload and isinstance(payload["tags"], list):
            candidates = payload["tags"]
        else:
            # Some Ollama variants return a dict of model-name -> metadata
            # or a list under other keys; fall back to any list values.
            for v in payload.values():
                if isinstance(v, list):
                    candidates = v
                    break
    elif isinstance(payload, list):
        candidates = payload

    models: list[ModelInfo] = []
    for item in candidates:
        if isinstance(item, dict):
            name = item.get("name") or item.get("model") or item.get("id")
            if name:
                models.append(ModelInfo(id=str(name), name=str(name), provider="ollama"))
        elif isinstance(item, str):
            models.append(ModelInfo(id=item, name=item, provider="ollama"))

    if not models:
        logger.info("Ollama returned no models at %s (raw payload shape: %s)", url, type(payload).__name__)

    return models


def list_github_models(
    api_key: str | None = None,
    **kwargs: Any,
) -> list[ModelInfo]:
    """List available models from the GitHub Models catalog.

    Fetches the official model catalog at
    ``https://models.github.ai/catalog/models`` and parses model IDs.
    """
    token = api_key or os.getenv("GITHUB_TOKEN", "")
    url = "https://models.github.ai/catalog/models"
    headers: dict[str, str] = {"User-Agent": "fin-ai/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=15) as response:
            payload = json.load(response)
    except URLError as exc:
        logger.warning("Failed to list GitHub models: %s", exc.reason)
        return []
    except Exception as exc:
        logger.warning("Failed to list GitHub models: %s", exc)
        return []

    # GitHub catalog may return a list or a dict with nested lists. Try to
    # extract all candidate model ids from common shapes.
    candidates: list[Any] = []
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict):
        # Common patterns: {'data': [...]} or {'models': [...]} or direct mapping
        for key in ("data", "models", "models_catalog", "items"):
            if key in payload and isinstance(payload[key], list):
                candidates = payload[key]
                break
        if not candidates:
            # Fallback: collect first list value
            for v in payload.values():
                if isinstance(v, list):
                    candidates = v
                    break

    model_ids = sorted(
        {
            str(item.get("id", item.get("model", item.get("name", "")).strip()))
            for item in candidates
            if isinstance(item, dict) and (item.get("id") or item.get("model") or item.get("name"))
        }
    )

    # Also accept plain-string lists
    if not model_ids and isinstance(candidates, list) and all(isinstance(i, str) for i in candidates):
        model_ids = sorted({str(i).strip() for i in candidates if i})

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
