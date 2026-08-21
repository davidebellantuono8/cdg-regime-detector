@echo off
setlocal
cd /d "%~dp0"
title CDG Regime Detector - Diagnostica

echo ============================================================
echo   DIAGNOSTICA CDG MACRO REGIME DETECTOR
echo ============================================================
echo.

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    where py >nul 2>nul && set "PY=py"
    if not defined PY set "PY=python"
)

%PY% diagnose.py

echo.
pause
endlocal
