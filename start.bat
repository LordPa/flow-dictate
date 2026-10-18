@echo off
REM Flow Dictate launcher for Windows. Double-click or run from a terminal.
REM First run creates a virtual environment and installs dependencies.
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
    echo Creating virtual environment...
    python -m venv venv || (echo Python not found. Install it from python.org and tick "Add to PATH". & pause & exit /b 1)
    echo Installing dependencies, this takes a few minutes the first time...
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\python.exe -m pip install -r requirements.txt || (echo Install failed. & pause & exit /b 1)
)
venv\Scripts\python.exe -m flow_dictate %*
pause
