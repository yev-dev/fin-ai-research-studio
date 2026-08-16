#!/usr/bin/env sh

set -eu

APP_DIR="${APP_DIR:-$HOME/research_studio}"
VECTOR_DB_DIR="${VECTOR_DB_DIR:-$APP_DIR/vector_db}"
PUBLISHED_RESEARCH_DIR="${PUBLISHED_RESEARCH_DIR:-$APP_DIR/published_research}"
QUESTION_HISTORY_DIR="${QUESTION_HISTORY_DIR:-$APP_DIR/question_history}"
PORT_NUMBER="${PORT_NUMBER:-8601}"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
RAG_SCRIPT="$PROJECT_DIR/scripts/rag_bulk_upload.py"

if [ -n "${OLLAMA_CHATBOT_PYTHON:-}" ]; then
    PYTHON_CMD="$OLLAMA_CHATBOT_PYTHON"
elif [ -x "/Users/yevgeniy/Development/Ext/anaconda3/envs/ollama_chatbot/bin/python" ]; then
    PYTHON_CMD="/Users/yevgeniy/Development/Ext/anaconda3/envs/ollama_chatbot/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_CMD="$CONDA_PREFIX/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "Python was not found. Set OLLAMA_CHATBOT_PYTHON or activate an environment first." >&2
    exit 1
fi

cd "$PROJECT_DIR"

if ! "$PYTHON_CMD" -c "import importlib; importlib.import_module('fin_ai')" >/dev/null 2>&1; then
    echo "fin_ai package not importable in $PYTHON_CMD environment." >&2
    exit 1
fi

echo "Running RAG bulk uploader"

"$PYTHON_CMD" "$RAG_SCRIPT" "$@"
