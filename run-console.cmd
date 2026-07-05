@echo off
setlocal enabledelayedexpansion

:: Run hedge fund CLI via uv

if not exist "pyproject.toml" (
    echo [ERROR] pyproject.toml not found. Run this from the project root.
    pause
    exit /b 1
)

echo [RUN] uv run python src/main.py --ticker AAPL,MSFT,NVDA
echo.
uv run python src/main.py --ticker AAPL,MSFT,NVDA
echo.
pause
