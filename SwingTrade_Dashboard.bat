@echo off
REM ===================================================
REM  Swing Trade Dashboard Launcher
REM  Double-click to start backend + modern web dashboard
REM ===================================================

echo.
echo =============================================
echo   SWING TRADE DASHBOARD (Web)
echo   Starting backend + frontend...
echo =============================================
echo.

REM --- Start FastAPI backend in a separate window ---
echo Starting FastAPI backend on http://localhost:8000 ...
start "SwingTrade Backend" cmd /k "cd /d C:\swing_trade && call venv\Scripts\activate.bat && uvicorn api.main:app --reload --port 8000"

echo.
echo Waiting 3 seconds for backend to initialize...
timeout /t 3 /nobreak >nul

REM --- Start Next.js frontend in this window ---
cd /d "C:\swing_trade\frontend"

echo.
echo Launching Next.js dev server on http://localhost:5000 ...
echo (Press CTRL+C in this window to stop the frontend.)
echo.

npm run dev

REM Keep window open if there's an error after the dev server stops
pause
