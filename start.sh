#!/usr/bin/env bash
# Flow Dictate launcher for macOS and Linux. First run installs dependencies.
cd "$(dirname "$0")"
if [ ! -x venv/bin/python ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv || { echo "python3 not found. Install Python 3.10+ first."; exit 1; }
    echo "Installing dependencies, this takes a few minutes the first time..."
    venv/bin/python -m pip install --upgrade pip
    venv/bin/python -m pip install -r requirements.txt || { echo "Install failed."; exit 1; }
fi
exec venv/bin/python -m flow_dictate "$@"
