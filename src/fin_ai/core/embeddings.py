"""
Embeddings factory — creates embedding instances via the same provider config
system used for chat LLM clients.

Usage::

    from fin_ai.core.embeddings import create_embeddings

    emb = create_embeddings(provider="ollama", model="nomic-embed-text:latest")
    vector = emb.embed_query("some text")
"""

from __future__ import annotations

import os
import time
import warnings

from typing import Any

import requests
from requests.exceptions import RequestException
import logging

from fin_ai.core.request import get_provider_config


# ---------------------------------------------------------------------------
# Ollama embeddings (langchain wrapper)
# ---------------------------------------------------------------------------


def _create_ollama_embeddings(model_name: str, base_url: str) -> Any:
    """Return an OllamaEmbeddings instance connected to the given base_url."""
    try:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model_name, base_url=base_url)
    except Exception as exc:
        error_text = str(exc)
        if "not found" in error_text.lower() and model_name in error_text:
            raise RuntimeError(
                f"Embedding model '{model_name}' is not available in Ollama. "
                f"Run: ollama pull {model_name}"
            ) from exc
        raise RuntimeError(
            f"Failed to initialize Ollama embeddings: {error_text}.\n"
            f"Quick check: curl {base_url}/api/tags"
        ) from exc


# ---------------------------------------------------------------------------
# GitHub / OpenAI-compatible embeddings (lightweight adapter)
# ---------------------------------------------------------------------------


class _GitHubEmbeddings:
    """Minimal embeddings adapter for OpenAI-compatible endpoints.

    POSTs JSON ``{"model": "...", "input": [...]}`` to ``<endpoint>/embeddings``.
    """

    _MODEL_TOKEN_LIMIT = 8192

    def __init__(self, model: str, endpoint: str, token: str):
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.token = token
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json",
        }
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"

    @staticmethod
    def _truncate_text(text: str, max_tokens: int) -> str:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return enc.decode(tokens[:max_tokens])
        except Exception:
            max_chars = max_tokens * 4
            if len(text) <= max_chars:
                return text
            return text[:max_chars]

    def _request(self, inputs: list[str]) -> list[list[float]]:
        url = f"{self.endpoint}/embeddings"
        truncated = [self._truncate_text(t, self._MODEL_TOKEN_LIMIT) for t in inputs]
        payload = {"model": self.model, "input": truncated}

        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, json=payload, headers=self.headers, timeout=30)
                if resp.status_code == 429 and attempt < max_retries - 1:
                    sleep_time = 2**attempt
                    warnings.warn(
                        f"Embedding rate limited (429), retrying in {sleep_time}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(sleep_time)
                    continue
                resp.raise_for_status()
                break
            except RequestException as re:
                if attempt < max_retries - 1:
                    status = getattr(getattr(re, "response", None), "status_code", None)
                    if status == 429:
                        sleep_time = 2**attempt
                        warnings.warn(
                            f"Embedding rate limited (429), retrying in {sleep_time}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(sleep_time)
                        continue
                response = getattr(re, "response", None)
                detail = ""
                if response is not None and response.text:
                    detail = f" Response body: {response.text[:600]}"
                status = getattr(getattr(re, "response", None), "status_code", None)
                if status == 429:
                    raise RuntimeError(
                        f"Embedding API rate limit exceeded after {max_retries} retries. "
                        f"Set a token for higher rate limits.{detail}"
                    ) from re
                if status == 401:
                    raise RuntimeError(
                        f"Embedding endpoint returned 401 Unauthorized. "
                        f"The GitHub token may be invalid or expired. "
                        f"Generate a new token at https://github.com/settings/tokens "
                        f"with access to GitHub Models.{detail}"
                    ) from re
                raise RuntimeError(
                    f"Failed to call embedding endpoint {url}: {re}.{detail}"
                ) from re

        try:
            data = resp.json()
        except Exception as exc:
            text = getattr(resp, "text", None)
            snippet = text[:1000] if isinstance(text, str) else str(text)
            raise RuntimeError(
                f"Failed to decode JSON from embedding endpoint {url} (status={getattr(resp, 'status_code', None)}). Response: {snippet}"
            ) from exc

        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                embeddings = []
                for item in data["data"]:
                    if isinstance(item, dict) and "embedding" in item:
                        embeddings.append(item["embedding"])
                    elif isinstance(item, dict) and "embeddings" in item:
                        embeddings.append(item["embeddings"])
                    else:
                        raise RuntimeError(f"Unexpected embedding item shape: {item}")
                return embeddings
            if "embeddings" in data and isinstance(data["embeddings"], list):
                return data["embeddings"]
        if isinstance(data, list) and all(isinstance(i, list) for i in data):
            return data
        # Include status code and (truncated) raw response for easier debugging
        raw = getattr(resp, "text", None)
        raw_snip = raw[:1000] if isinstance(raw, str) else str(raw)
        raise RuntimeError(
            f"Unable to parse embeddings response (status={getattr(resp, 'status_code', None)}): {data}. Raw response: {raw_snip}"
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._request(texts)

    def embed_query(self, text: str) -> list[float]:
        vectors = self._request([text])
        if not vectors:
            raise RuntimeError("Embedding endpoint returned no vectors for query input.")
        return vectors[0]

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def create_embeddings(
    provider: str,
    model: str,
    *,
    api_base: str | None = None,
    api_key: str | None = None,
) -> Any:
    """Create an embeddings instance for *provider*.

    Parameters
    ----------
    provider : str
        ``"ollama"`` or ``"github"`` (also accepts ``"deepseek"``, routes to
        the same OpenAI-compatible path).
    model : str
        Embedding model identifier.
    api_base : str, optional
        API base URL.  Falls back to ``ProviderConfig.default_base_url``.
    api_key : str, optional
        API key / token (needed for GitHub/OpenAI-compatible endpoints).

    Returns
    -------
    An object with ``embed_query(text)``, ``embed_documents(texts)``, and
    ``__call__(text)`` methods.
    """
    logger = logging.getLogger(__name__)
    cfg = get_provider_config(provider)
    resolved_base = cfg.build_api_base(api_base)
    resolved_key = api_key or os.getenv("GITHUB_TOKEN", "")

    # Log whether an API key was provided (do not log the key itself).
    if resolved_key:
        logger.debug("Embeddings: using provided GitHub API key (redacted)")
    else:
        logger.debug("Embeddings: no GitHub API key provided; proceeding without Authorization header")

    if provider == "ollama":
        return _create_ollama_embeddings(model, resolved_base)

    if provider in ("github", "deepseek"):
        if provider == "github" and not resolved_key:
            raise ValueError(
                "GitHub provider requires an API token. "
                "Set the GITHUB_TOKEN environment variable or pass api_key."
            )
        return _GitHubEmbeddings(model=model, endpoint=resolved_base, token=resolved_key)

    raise ValueError(f"Unknown embeddings provider: {provider!r}")
