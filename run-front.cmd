@echo off
setlocal enabledelayedexpansion

:: AI Hedge Fund -- Frontend dev server

set FRONTEND_DIR=app\frontend

if not exist "%FRONTEND_DIR%\node_modules" (
    echo [ERROR] node_modules not found.
    echo [INFO] Run: cd %FRONTEND_DIR% ^&^& npm install
    pause
    exit /b 1
)

echo [INFO] Starting frontend on http://localhost:5173
echo [INFO] Press Ctrl+C to stop
echo.
pushd %FRONTEND_DIR%
npx vite --host 127.0.0.1 --port 5173
popd
echo.
pause
