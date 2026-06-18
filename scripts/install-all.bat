@echo off
REM ============================================
REM  1-Click Installation Script
REM  Installs all project dependencies and setup
REM ============================================

setlocal enabledelayedexpansion

REM Set UTF-8 encoding
set PYTHONIOENCODING=utf-8

REM Navigate to project root (parent of scripts folder)
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

echo.
echo ============================================
echo  LocalAiLab - 1-Click Install
echo ============================================
echo.
echo Project Root: %CD%
echo.

REM ============================================
REM  Step 1: Check Python Installation
REM ============================================
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.11+ from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%A in ('python --version') do set PYTHON_VERSION=%%A
echo [OK] %PYTHON_VERSION% found
echo.

REM ============================================
REM  Step 2: Install/Check uv (Fast Python package manager)
REM ============================================
echo [2/4] Checking/Installing uv package manager...
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   Installing uv...
    pip install uv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install uv
        pause
        exit /b 1
    )
)

for /f "tokens=*" %%A in ('uv --version') do set UV_VERSION=%%A
echo [OK] %UV_VERSION% found
echo.

REM ============================================
REM  Step 3: Create Virtual Environment (if needed)
REM ============================================
echo [3/4] Setting up virtual environment...
if not exist ".venv" (
    echo   Creating .venv...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)
echo.

REM ============================================
REM  Step 4: Install Project Dependencies
REM ============================================
echo [4/4] Installing project dependencies...
echo   This may take 2-5 minutes on first install...
echo.

uv pip install -e .

if %errorlevel% equ 0 (
    echo.
    echo ============================================
    echo  SUCCESS: Installation Complete!
    echo ============================================
    echo.
    echo Project is ready to use. You can now run:
    echo.
    echo   Web UI:
    echo   - scripts\run-streamlit-app.bat
    echo.
    echo   Development Tools:
    echo   - scripts\run-langgraph-dev.bat    (LangGraph Studio)
    echo   - scripts\launcher.bat             (Quick Launcher)
    echo.
    echo   Python CLI:
    echo   - uv run python agent.py
    echo.
    echo   Utilities:
    echo   - scripts\setup-dependencies.bat   (Update dependencies)
    echo   - scripts\kill-port-2024.bat       (Free port 2024)
    echo.
    echo For more information, see README.md
    echo.
    pause
    exit /b 0
) else (
    echo.
    echo ============================================
    echo  ERROR: Installation Failed
    echo ============================================
    echo.
    echo Please check the error messages above.
    echo Make sure you have internet connection for package downloads.
    echo.
    pause
    exit /b 1
)
