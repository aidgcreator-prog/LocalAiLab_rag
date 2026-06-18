@echo off
REM Setup and Dependency Installer
REM Installs all project dependencies using uv

setlocal enabledelayedexpansion

REM Set UTF-8 encoding for Python to handle Unicode characters properly
set PYTHONIOENCODING=utf-8

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo.
echo ============================================
echo  Project Setup - Install Dependencies
echo ============================================
echo.
echo Installing project dependencies using uv...
echo Project: %SCRIPT_DIR%
echo.

REM Check if uv is installed
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: uv is not installed or not in PATH
    echo.
    echo Install uv from: https://docs.astral.sh/uv/
    echo.
    pause
    exit /b 1
)

echo uv found. Installing dependencies...
echo.

REM Install dependencies using uv pip (using pyproject.toml or requirements)
uv pip install -e .

if %errorlevel% equ 0 (
    echo.
    echo ============================================
    echo  SUCCESS: Dependencies installed
    echo ============================================
    echo.
    echo You can now run:
    echo   - run-streamlit-app.bat     (Web UI)
    echo   - run-langgraph-dev.bat     (LangGraph Studio)
    echo   - uv run python agent.py    (CLI)
    echo.
    pause
) else (
    echo.
    echo ERROR: Failed to install dependencies
    echo.
    pause
    exit /b 1
)

endlocal
