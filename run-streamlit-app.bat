@echo off
REM Streamlit App Launcher
REM Runs the Streamlit web interface for the combined agents

setlocal enabledelayedexpansion

REM Set UTF-8 encoding for Python to handle Unicode characters properly
set PYTHONIOENCODING=utf-8

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo.
echo ============================================
echo  Agent Orchestrator - Streamlit App
echo ============================================
echo.
echo Starting Streamlit application...
echo Project: %SCRIPT_DIR%
echo.
echo The app will open in your browser at:
echo   http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ============================================
echo.

REM Force local .venv for consistent runtime (avoid uv picking a different env)
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Missing .venv\Scripts\python.exe
    echo Run setup-dependencies.bat first.
    pause
    exit /b 1
)

set "LLAMA_CPP_FLASH_ATTN=1"

REM ── Auto-start llama-server if not already running ──────────────────────────
echo Checking if llama-server is running on port 8080...
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8080/health' -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop; Write-Host '  llama-server already running.' } catch { Write-Host '  Starting llama-server in a new window...'; Start-Process cmd -ArgumentList '/k', '"%~dp0run-llama-server.bat"'; Start-Sleep -Seconds 3 }"
echo.

echo Using interpreter:
".venv\Scripts\python.exe" -c "import sys; print(sys.executable)"
echo.

REM Run Streamlit app with watcher disabled to avoid noisy transformers lazy-import traces
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.fileWatcherType none

REM If the command fails, show error message
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to start Streamlit app
    echo Make sure you have:
    echo   - .venv created in this folder
    echo   - All dependencies installed (streamlit, deepagents, etc.)
    echo   - Python 3.10+ available
    echo.
    pause
)

endlocal
