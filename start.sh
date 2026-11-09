#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Installing dependencies..."
.venv/bin/pip install -r requirements.txt -q --upgrade

echo ""
echo "Starting Running Dashboard..."
echo "Open http://localhost:8050 in your browser"
echo ""
.venv/bin/python app.py
