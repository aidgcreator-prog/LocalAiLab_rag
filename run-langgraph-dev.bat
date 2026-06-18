@echo off
REM LangGraph Development Server Launcher
REM Runs the LangGraph development server for the combined agents project
REM This uses uv to manage the Python environment automatically

setlocal enabledelayedexpansion

REM Set UTF-8 encoding for Python to handle Unicode characters properly
set PYTHONIOENCODING=utf-8

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo.
echo ============================================
echo  LangGraph Development Server
echo ============================================
echo.
echo Starting LangGraph development server...
echo Project: %SCRIPT_DIR%
echo.
echo The server will auto-discover an available port
echo You can access the server at:
echo   http://127.0.0.1:2024 (or next available port)
echo.
echo Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:PORT
echo API Docs: http://127.0.0.1:PORT/docs
echo.
echo Press Ctrl+C to stop the server
echo ============================================
echo.

REM Run langgraph dev with auto-port discovery
REM By omitting the port argument, it will find an available port automatically
uv run langgraph dev --no-reload

REM If the command fails, show error message
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to start LangGraph development server
    echo.
    echo Possible solutions:
    echo   1. Port 2024 might be in use from a previous server
    echo      - Try: taskkill /F /IM python.exe (or find and kill the process)
    echo   2. Make sure you have:
    echo      - uv installed globally
    echo      - langgraph-cli installed (via uv)
    echo      - Agent files properly configured
    echo.
    pause
)

endlocal
