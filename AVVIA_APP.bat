@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title CDG Macro Regime Detector

echo ============================================================
echo   CDG MACRO REGIME DETECTOR - AVVIO AUTOMATICO
echo ============================================================
echo.

set "BASEPY="
where py >nul 2>nul
if %errorlevel%==0 set "BASEPY=py"
if not defined BASEPY (
    where python >nul 2>nul
    if %errorlevel%==0 set "BASEPY=python"
)

if not defined BASEPY (
    echo ERRORE: Python non trovato sul PC.
    echo Installa Python 3.10-3.14 e seleziona "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

rem 1) Se il Python gia' installato ha Streamlit e le librerie core, usalo direttamente.
%BASEPY% -c "import streamlit,pandas,numpy,plotly,yaml,openpyxl,requests" >nul 2>nul
if %errorlevel%==0 (
    echo Uso l'ambiente Python gia' configurato sul PC.
    %BASEPY% launcher.py
    set "RC=%errorlevel%"
    goto :END
)

rem 2) Altrimenti crea un ambiente dedicato nella cartella del progetto.
if not exist ".venv\Scripts\python.exe" (
    echo Prima esecuzione: creo l'ambiente dedicato .venv...
    %BASEPY% -m venv .venv
    if errorlevel 1 (
        echo.
        echo ERRORE nella creazione dell'ambiente Python.
        pause
        exit /b 2
    )
)

echo Avvio con l'ambiente dedicato...
".venv\Scripts\python.exe" launcher.py
set "RC=%errorlevel%"

:END
if not "%RC%"=="0" (
    echo.
    echo ============================================================
    echo L'app non e' partita. Codice errore: %RC%
    echo Dettagli salvati in: avvio_log.txt
    echo ============================================================
    echo.
    if exist avvio_log.txt type avvio_log.txt
    echo.
    pause
)
endlocal
