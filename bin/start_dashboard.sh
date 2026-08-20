#!/usr/bin/env sh

set -eu

APP_DIR="$HOME/research_studio"
VECTOR_DB_DIR="$APP_DIR/vector_db"
PUBLISHED_RESEARCH_DIR="$APP_DIR/published_research"
QUESTION_HISTORY_DIR="$APP_DIR/question_history"
PORT_NUMBER="${PORT_NUMBER:-8601}"

export APP_DIR
export VECTOR_DB_DIR
export PUBLISHED_RESEARCH_DIR
export QUESTION_HISTORY_DIR

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
FIN_APP_PATH="$PROJECT_DIR/dashboard/financial_analyst_dashboard.py"
VECTOR_DB_PATH_DEFAULT="$PROJECT_DIR/vector_db"

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

export VECTOR_DB_DIR="${VECTOR_DB_DIR:-$VECTOR_DB_PATH_DEFAULT}"
cd "$PROJECT_DIR"

# Ensure `streamlit` is importable in the chosen Python environment
if ! "$PYTHON_CMD" -c "import importlib; importlib.import_module('streamlit')" >/dev/null 2>&1; then
    echo "Streamlit is not installed in $PYTHON_CMD environment. Install with: $PYTHON_CMD -m pip install streamlit" >&2
    exit 1
fi

echo "Starting FinAI dashboard on port $PORT_NUMBER using $PYTHON_CMD"

"$PYTHON_CMD" -m streamlit run "$FIN_APP_PATH" --server.port "$PORT_NUMBER" "$@" &
FIN_PID=$!


cleanup() {
    kill "$FIN_PID" 2>/dev/null || true
}

trap cleanup INT TERM EXIT
wait "$FIN_PID"