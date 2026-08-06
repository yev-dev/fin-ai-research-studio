"""
Application configuration — paths, model defaults, provider endpoints.

All configuration consumed by ``fin_ai.core.processor``, the dashboard,
and notebooks lives here.  Environment variables override defaults.

Imports should use::

    from fin_ai.config.fin_ai import VECTOR_DB_DIR, OLLAMA_BASE_URL, ...
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

# __file__ = src/fin_ai/config/fin_ai.py
#   .parent            = src/fin_ai/config/
#   .parent.parent     = src/fin_ai/
#   .parent.parent.parent = src/
#   .parent.parent.parent.parent = project root (ai-financial-analysis/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# Load .env from the project root (ai-financial-analysis/.env)
# Keep only secrets here: tokens, passwords.  Everything else goes below.
_env_file = PROJECT_ROOT / ".env"

if _env_file.exists():
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_env_file, override=False)

_vector_db_env = os.environ.get("VECTOR_DB_DIR", "").strip()
VECTOR_DB_DIR = _vector_db_env or str(PROJECT_ROOT / "vector_db")
PUBLISHED_RESEARCH_DIR = PROJECT_ROOT / "published_research"
LOG_DIR = PROJECT_ROOT / "logs"
QUESTION_HISTORY_DIR = Path(VECTOR_DB_DIR) / "question_history"

os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(QUESTION_HISTORY_DIR, exist_ok=True)
os.makedirs(PUBLISHED_RESEARCH_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Yahoo Finance service mode
# ---------------------------------------------------------------------------

YAHOO_SERVICE_OFFLINE = (
    os.environ.get("YAHOO_SERVICE_OFFLINE", "true").strip().lower() == "true"
)
YAHOO_DATA_DIR = os.environ.get(
    "YAHOO_DATA_DIR",
    str(PROJECT_ROOT / "data"),
)

# ---------------------------------------------------------------------------
# Provider endpoints
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.environ.get("OLLAMA_ENDPOINT", "http://127.0.0.1:11434").rstrip("/")
GITHUB_BASE_URL = "https://models.github.ai/inference"
GITHUB_EMBEDDING_BASE_URL = "https://models.github.ai/inference"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------

DEFAULT_PROVIDER = os.environ.get("DEFAULT_PROVIDER", "ollama").strip().lower()
DEFAULT_CHAT_MODEL = "llama3.2"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"
DEFAULT_EMBEDDINGS_PROVIDER = os.getenv("DEFAULT_EMBEDDINGS_PROVIDER", "ollama").strip().lower()

DEFAULT_GITHUB_MODEL = "openai/gpt-4o"
DEFAULT_GITHUB_EMBEDDING_MODEL = "openai/text-embedding-3-small"

DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"

# ---------------------------------------------------------------------------
# Supported document types
# ---------------------------------------------------------------------------

SUPPORTED_UPLOAD_TYPES = ["pdf", "csv", "json", "html", "docx"]
SUPPORTED_SOURCE_SUFFIXES = [".pdf", ".csv", ".json", ".html", ".docx"]

# ---------------------------------------------------------------------------
# SMTP (email) — non-secret defaults
# ---------------------------------------------------------------------------

# Override in .env if needed (AI_RESEARCH_SMTP_HOST, etc.)
AI_RESEARCH_SMTP_HOST = os.environ.get("AI_RESEARCH_SMTP_HOST", "smtp.gmail.com")
AI_RESEARCH_SMTP_PORT = int(os.environ.get("AI_RESEARCH_SMTP_PORT", "587"))
AI_RESEARCH_SMTP_USER = os.environ.get("AI_RESEARCH_SMTP_USER", "")

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

__all__ = [
    "PROJECT_ROOT",
    "SRC_DIR",
    "VECTOR_DB_DIR",
    "PUBLISHED_RESEARCH_DIR",
    "QUESTION_HISTORY_DIR",
    "OLLAMA_BASE_URL",
    "GITHUB_BASE_URL",
    "GITHUB_EMBEDDING_BASE_URL",
    "DEEPSEEK_BASE_URL",
    "DEFAULT_PROVIDER",
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDINGS_PROVIDER",
    "YAHOO_SERVICE_OFFLINE",
    "YAHOO_DATA_DIR",
    "DEFAULT_GITHUB_MODEL",
    "DEFAULT_GITHUB_EMBEDDING_MODEL",
    "DEFAULT_DEEPSEEK_MODEL",
    "AI_RESEARCH_SMTP_HOST",
    "AI_RESEARCH_SMTP_PORT",
    "AI_RESEARCH_SMTP_USER",
    "SUPPORTED_UPLOAD_TYPES",
    "SUPPORTED_SOURCE_SUFFIXES",
]
