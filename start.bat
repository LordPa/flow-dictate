@echo off
REM Flow Dictate launcher for Windows. Double-click or run from a terminal.
REM First run creates a virtual environment and installs dependencies.
cd /d "%~dp0"
REM Pick up the latest version if this folder came from git.
if exist .git (
    where git >nul 2>nul && git pull --ff-only -q
)
if not exist venv\Scripts\python.exe (
    echo Creating virtual environment...
    python -m venv venv || (echo Python not found. Install it from python.org and tick "Add to PATH". & pause & exit /b 1)
    echo Installing dependencies, this takes a few minutes the first time...
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\python.exe -m pip install -r requirements.txt || (echo Install failed. & pause & exit /b 1)
)
if /I "%1"=="gpu" (
    echo Installing NVIDIA GPU libraries, about 1 GB...
    venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
    echo Done. Run start.bat again.
    pause
    exit /b
)
venv\Scripts\python.exe -m flow_dictate %*
pause
