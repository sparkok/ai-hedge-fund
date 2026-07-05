@echo off
setlocal enabledelayedexpansion

:: AI Hedge Fund -- Backend server (uv)

:: Kill leftover process on port 8000 if any
for /f "tokens=5" %%a in (\'netstat -ano^^ ^| findstr :8000^^\') do (
    echo [INFO] Killing process %%a on port 8000
    taskkill /f /pid %%a >nul 2>&1
    timeout /t 1 /nobreak >nul
)

if not exist ".env" (
    echo [ERROR] .env file not found in project root
    pause
    exit /b 1
)

echo [INFO] Starting backend on http://127.0.0.1:8000
echo [INFO] Press Ctrl+C to stop
echo.
uv run uvicorn app.backend.main:app --reload --host 127.0.0.1 --port 8000
echo.
pause
