"""Shared fixtures and helpers for integration tests.

All integration tests require a running local Ollama instance with the
``llama3.1`` model pulled.  Tests that need a model will use this fixture
and skip gracefully if Ollama is unreachable.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from fin_ai.config import OLLAMA_BASE_URL

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

INTEGRATION_MARKER = pytest.mark.integration
OLLAMA_MARKER = pytest.mark.ollama

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ollama_is_reachable(url: str = OLLAMA_BASE_URL) -> bool:
    """Return ``True`` if an Ollama instance responds at *url*."""
    try:
        resp = requests.get(f"{url}/api/tags", timeout=5)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def model_is_available(model: str, url: str = OLLAMA_BASE_URL) -> bool:
    """Return ``True`` if *model* is listed in the local Ollama instance."""
    try:
        resp = requests.get(f"{url}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(model in m for m in models)
    except (requests.ConnectionError, requests.Timeout, KeyError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ollama_available() -> bool:
    """Session-scoped check: is Ollama reachable?"""
    return ollama_is_reachable()


@pytest.fixture(scope="session")
def llama3_available(ollama_available: bool) -> bool:
    """Session-scoped check: is llama3.1 model pulled?"""
    if not ollama_available:
        return False
    return model_is_available("llama3.1")


@pytest.fixture
def skip_if_no_ollama(ollama_available: bool) -> None:
    """Skip the test if Ollama is not running."""
    if not ollama_available:
        pytest.skip("Ollama service is not available — skipping integration test.")


@pytest.fixture
def skip_if_no_llama(skip_if_no_ollama, llama3_available: bool) -> None:
    """Skip the test if the llama3.1 model is not pulled."""
    if not llama3_available:
        pytest.skip("llama3.1 model not found in Ollama — skipping test.")


@pytest.fixture(scope="session")
def ollama_llm_config() -> dict[str, Any]:
    """Return a default ``llm_config`` dict targeting the local Ollama instance."""
    return {
        "config_list": [
            {
                "model": "llama3.1",
                "base_url": f"{OLLAMA_BASE_URL}/v1",
                "api_key": "ollama",
            }
        ],
        "temperature": 0.0,
        "timeout": 300,
    }


@pytest.fixture(scope="session")
def ollama_llama3_llm_config() -> dict[str, Any]:
    """Return an ``llm_config`` using llama3.1 (the default fin_ai model)."""
    return {
        "config_list": [
            {
                "model": "llama3.1",
                "base_url": f"{OLLAMA_BASE_URL}/v1",
                "api_key": "ollama",
            }
        ],
        "temperature": 0.0,
        "timeout": 300,
    }


@pytest.fixture
def mock_llm_config() -> dict[str, Any]:
    """Return an ``llm_config`` with a MagicMock instead of a real model.

    Use this for tests that verify wiring/dispatch but do not actually
    invoke the LLM — the agent will call tool functions via the mock.
    """
    return {
        "config_list": [
            {
                "model": MagicMock(),
                "api_key": "mock",
            }
        ],
        "temperature": 0.0,
        "timeout": 5,
    }


# ---------------------------------------------------------------------------
# Shared mini retrieve_config for RAG tests
# ---------------------------------------------------------------------------

SAMPLE_RETRIEVE_CONFIG = {
    "task": "default",
    "docs_path": "/tmp/fin_ai_integration_test_docs",
    "chunk_token_size": 500,
    "model": "llama3.1",
    "client": "chromadb",
    "embedding_model": "nomic-embed-text:latest",
    "get_or_create": True,
    "overwrite": True,
}
