@echo off
setlocal enabledelayedexpansion

:: AI Hedge Fund -- .venv launcher

set VENV_DIR=.venv
set VENV_ACTIVATE=%VENV_DIR%\Scripts\activate.bat

if not exist "%VENV_ACTIVATE%" (
    echo [ERROR] .venv not found at %VENV_DIR%\Scripts\activate.bat
    echo.
    echo Create one first:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -e .
    echo.
    pause
    exit /b 1
)

:: ---------------------------------------------------------------------------
:: Detect subcommand
:: ---------------------------------------------------------------------------

set "CMD="
set "TAIL="
set "FIRST="

for %%a in (%*) do if not defined FIRST set "FIRST=%%~a"

if not "%FIRST%"=="" (
    for %%a in (main backtest web shell) do (
        if /i "%FIRST%"=="%%a" (
            set "CMD=%%a"
            goto :shift_cmd
        )
    )
)

:: Default: web when no subcommand given
set "CMD=web"
goto :exec

:shift_cmd
set "TAIL="
:collect
shift
if "%~1"=="" goto :exec
if defined TAIL (set "TAIL=!TAIL! %~1") else (set "TAIL=%~1")
goto :collect

:exec

:: ---------------------------------------------------------------------------
:: Subcommand: shell -- just activate venv
:: ---------------------------------------------------------------------------
if /i "%CMD%"=="shell" (
    call "%VENV_ACTIVATE%"
    echo .venv activated. Run  deactivate  to exit.
    cmd /k
    exit /b 0
)

:: ---------------------------------------------------------------------------
:: Subcommand: main -- run hedge fund CLI
:: ---------------------------------------------------------------------------
if /i "%CMD%"=="main" (
    call "%VENV_ACTIVATE%" || exit /b 1
    echo [RUN] python src/main.py %TAIL%
    echo.
    python src/main.py %TAIL%
    echo.
    pause
    exit /b !ERRORLEVEL!
)

:: ---------------------------------------------------------------------------
:: Subcommand: backtest -- run backtester
:: ---------------------------------------------------------------------------
if /i "%CMD%"=="backtest" (
    call "%VENV_ACTIVATE%" || exit /b 1
    echo [RUN] python src/backtester.py %TAIL%
    echo.
    python src/backtester.py %TAIL%
    echo.
    pause
    exit /b !ERRORLEVEL!
)

:: ---------------------------------------------------------------------------
:: Subcommand: web -- start backend + frontend
:: ---------------------------------------------------------------------------
if /i "%CMD%"=="web" (
    call "%VENV_ACTIVATE%" || exit /b 1

    echo ===== AI Hedge Fund -- Web Application =====
    echo.

    :: Check for .env
    if not exist ".env" (
        if exist ".env.example" (
            echo [WARN] No .env found. Copying from .env.example...
            copy .env.example .env >nul
            echo [WARN] Edit .env to add your API keys before running.
            echo.
        ) else (
            echo [ERROR] No .env or .env.example found.
            pause
            exit /b 1
        )
    )

    :: Start backend
    echo [INFO] Starting backend on http://127.0.0.1:8000
    start "hedge-fund-backend" /b python -m uvicorn app.backend.main:app --reload --host 127.0.0.1 --port 8000

    :: Wait for backend to initialize
    timeout /t 3 /nobreak >nul

    :: Start frontend
    if exist "app\frontend\node_modules" (
        echo [INFO] Starting frontend on http://localhost:5173
        pushd app\frontend
        start "hedge-fund-frontend" /b npx vite --host 127.0.0.1 --port 5173
        popd
    ) else (
        echo [WARN] Frontend dependencies not installed.
        echo [WARN] Run: cd app\frontend ^&^& npm install
        echo [WARN] Then re-run this script.
    )

    echo.
    echo ===== Services =====
    echo   Frontend:  http://localhost:5173
    echo   Backend:   http://127.0.0.1:8000
    echo   API Docs:  http://127.0.0.1:8000/docs
    echo ====================
    echo.
    echo Press any key to stop all services...
    pause >nul

    taskkill /f /fi "WINDOWTITLE eq hedge-fund-backend" >nul 2>&1
    taskkill /f /fi "WINDOWTITLE eq hedge-fund-frontend" >nul 2>&1
    echo [OK] Services stopped.
    exit /b 0
)

echo [ERROR] Unknown command: %CMD%
exit /b 1
