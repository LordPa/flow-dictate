#!/usr/bin/env bash
# Flow Dictate launcher for macOS and Linux. First run installs dependencies.
cd "$(dirname "$0")"
# Pick up the latest version if this folder came from git.
if [ -d .git ] && command -v git >/dev/null; then git pull --ff-only -q; fi
if [ ! -x venv/bin/python ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv || { echo "python3 not found. Install Python 3.10+ first."; exit 1; }
    venv/bin/python -m pip install --upgrade pip
fi
# Install or update dependencies whenever requirements.txt changed.
if ! cmp -s requirements.txt venv/requirements.installed; then
    echo "Installing dependencies, this takes a few minutes the first time..."
    venv/bin/python -m pip install -r requirements.txt || { echo "Install failed."; exit 1; }
    cp requirements.txt venv/requirements.installed
fi
if [ "$1" = "gpu" ]; then
    exec venv/bin/python -m pip install -r requirements-gpu.txt
fi
exec venv/bin/python -m flow_dictate "$@"
