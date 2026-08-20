@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_DIR=%%~fI"
set "FIN_APP_PATH=%PROJECT_DIR%\dashboard\financial_analyst_dashboard.py"
set "VECTOR_DB_PATH_DEFAULT=%PROJECT_DIR%\vector_db"
set "PYTHON_CMD="
set "FIN_DASHBOARD_PORT=8502"

if defined FIN_DASHBOARD_PORT_OVERRIDE set "FIN_DASHBOARD_PORT=%FIN_DASHBOARD_PORT_OVERRIDE%"

if defined OLLAMA_CHATBOT_PYTHON (
    set "PYTHON_CMD=%OLLAMA_CHATBOT_PYTHON%"
) else if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" (
    set "PYTHON_CMD=%CONDA_PREFIX%\python.exe"
) else if exist "%USERPROFILE%\anaconda3\envs\ollama_chatbot\python.exe" (
    set "PYTHON_CMD=%USERPROFILE%\anaconda3\envs\ollama_chatbot\python.exe"
) else if exist "%LOCALAPPDATA%\anaconda3\envs\ollama_chatbot\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\anaconda3\envs\ollama_chatbot\python.exe"
) else if exist "%ProgramData%\anaconda3\envs\ollama_chatbot\python.exe" (
    set "PYTHON_CMD=%ProgramData%\anaconda3\envs\ollama_chatbot\python.exe"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python was not found. Set OLLAMA_CHATBOT_PYTHON or activate an environment first.
        exit /b 1
    )
    set "PYTHON_CMD=python"
)

if not defined VECTOR_DB_DIR (
    set "VECTOR_DB_DIR=%VECTOR_DB_PATH_DEFAULT%"
)

cd /d "%PROJECT_DIR%"
start "Financial Analyst Dashboard" "%PYTHON_CMD%" -m streamlit run "%FIN_APP_PATH%" --server.port %FIN_DASHBOARD_PORT% %*