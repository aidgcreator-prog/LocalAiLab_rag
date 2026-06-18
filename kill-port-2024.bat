@echo off
REM Kill Process on Port 2024 - Helper Script
REM Stops any LangGraph dev server running on port 2024

setlocal enabledelayedexpansion

echo.
echo ============================================
echo  Kill Process on Port 2024
echo ============================================
echo.
echo Checking for processes using port 2024...
echo.

REM Find process using port 2024
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":2024"') do (
    echo Found process ID: %%a
    echo Killing process...
    taskkill /PID %%a /F
    if %errorlevel% equ 0 (
        echo ✓ Process killed successfully
    ) else (
        echo ⚠ Could not kill process (may require admin privileges)
    )
)

echo.
echo Checking again...
netstat -ano | findstr ":2024" >nul 2>&1
if %errorlevel% equ 0 (
    echo ⚠ Port 2024 is still in use
    echo Try running Command Prompt as Administrator
) else (
    echo ✓ Port 2024 is now free
)

echo.
echo You can now run: run-langgraph-dev.bat
echo.
pause

endlocal
