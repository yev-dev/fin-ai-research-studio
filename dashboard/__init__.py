"""
Dashboard package — re-exports configuration from ``fin_ai.config.fin_ai``
for backward compatibility.  Prefer importing directly from ``fin_ai.config``
in new code.
"""

import sys
from pathlib import Path

# .env loading is handled by fin_ai.config.fin_ai on first import

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
SRC_DIR = PARENT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Re-export all configuration from the central config module
from fin_ai.config import (  # noqa: E402, F401
    # Paths
    VECTOR_DB_DIR,
    QUESTION_HISTORY_DIR,
    # Provider URLs
    OLLAMA_BASE_URL,
    GITHUB_BASE_URL,
    GITHUB_EMBEDDING_BASE_URL,
    DEEPSEEK_BASE_URL,
    # Default models
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDINGS_PROVIDER,
    DEFAULT_GITHUB_MODEL,
    DEFAULT_GITHUB_EMBEDDING_MODEL,
    DEFAULT_DEEPSEEK_MODEL,
    LOG_DIR,
)

__all__ = [
    "BASE_DIR",
    "PARENT_DIR",
    "SRC_DIR",
    "VECTOR_DB_DIR",
    "QUESTION_HISTORY_DIR",
    "GITHUB_BASE_URL",
    "GITHUB_EMBEDDING_BASE_URL",
    "DEEPSEEK_BASE_URL",
    "OLLAMA_BASE_URL",
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDINGS_PROVIDER",
    "DEFAULT_GITHUB_MODEL",
    "DEFAULT_GITHUB_EMBEDDING_MODEL",
    "DEFAULT_DEEPSEEK_MODEL",
    "LOG_DIR",
]