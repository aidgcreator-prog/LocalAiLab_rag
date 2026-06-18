@echo off
REM Streamlit App Launcher
REM This script activates the virtual environment and runs the Streamlit app

cd /d "%~dp0"

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run Streamlit with watcher disabled to avoid noisy transformers lazy-import traces
streamlit run streamlit_app.py --server.fileWatcherType none

REM Pause to see any errors
pause
