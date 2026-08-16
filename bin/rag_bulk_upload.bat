@echo off
setlocal

set APP_DIR=%APP_DIR%
if "%APP_DIR%"=="" set APP_DIR=%USERPROFILE%\research_studio
set VECTOR_DB_DIR=%VECTOR_DB_DIR%
if "%VECTOR_DB_DIR%"=="" set VECTOR_DB_DIR=%APP_DIR%\vector_db

set PROJECT_DIR=%~dp0\..
set RAG_SCRIPT=%PROJECT_DIR%\scripts\rag_bulk_upload.py

python "%RAG_SCRIPT%" %*
