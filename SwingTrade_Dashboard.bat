@echo off
REM ===================================================
REM  Swing Trade Dashboard Launcher
REM  Double-click to start the dashboard
REM ===================================================

echo.
echo =============================================
echo   SWING TRADE DASHBOARD
echo   Starting...
echo =============================================
echo.

cd /d "C:\swing_trade"

REM Activate virtual environment and run dashboard
call .\venv\Scripts\activate.bat
streamlit run swing_trader/dashboard/app.py

REM Keep window open if there's an error
pause
