@echo off
setlocal enabledelayedexpansion

:: AI Hedge Fund -- Backend server (uv)

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
