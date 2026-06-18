@echo off
REM Agent Orchestrator - Quick Launcher
REM Choose which component to run

setlocal enabledelayedexpansion

REM Set UTF-8 encoding for Python to handle Unicode characters properly
set PYTHONIOENCODING=utf-8

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

:menu
cls
echo.
echo ============================================
echo  Agent Orchestrator - Quick Launcher
echo ============================================
echo.
echo Choose what to run:
echo.
echo   1) Streamlit Web App + llama-server  (recommended)
echo   2) llama-server only                (http://localhost:8080/v1)
echo   3) Streamlit Web App only           (http://localhost:8501)
echo   4) LangGraph Dev Server             (http://localhost:8100)
echo   5) Run Agent from CLI               (command line)
echo   6) Install/Update Dependencies
echo   7) Exit
echo.
echo ============================================
echo.

set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" (
    echo.
    start "llama-server" cmd /k call "%SCRIPT_DIR%run-llama-server.bat"
    echo  Waiting 3 s for llama-server to start...
    timeout /t 3 /nobreak >nul
    call run-streamlit-app.bat
    goto menu
) else if "%choice%"=="2" (
    echo.
    call run-llama-server.bat
    goto menu
) else if "%choice%"=="3" (
    echo.
    call run-streamlit-app.bat
    goto menu
) else if "%choice%"=="4" (
    echo.
    call run-langgraph-dev.bat
    goto menu
) else if "%choice%"=="5" (
    echo.
    echo Running agent from CLI...
    echo.
    uv run python agent.py
    echo.
    pause
    goto menu
) else if "%choice%"=="6" (
    echo.
    call setup-dependencies.bat
    goto menu
) else if "%choice%"=="7" (
    echo.
    echo Goodbye!
    echo.
    exit /b 0
) else (
    echo.
    echo Invalid choice. Please try again.
    echo.
    pause
    goto menu
)

endlocal
