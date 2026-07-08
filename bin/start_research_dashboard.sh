#!/usr/bin/env sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
FIN_APP_PATH="$PROJECT_DIR/dashboard/financial_analyst_app.py"
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

FIN_DASHBOARD_PORT="${FIN_DASHBOARD_PORT:-8501}"

"$PYTHON_CMD" -m streamlit run "$FIN_APP_PATH" --server.port "$FIN_DASHBOARD_PORT" "$@" &
FIN_PID=$!

cleanup() {
    kill "$FIN_PID" 2>/dev/null || true
}

trap cleanup INT TERM EXIT
wait "$FIN_PID"