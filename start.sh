#!/usr/bin/env bash
# AI PDF Translator - Mac / Linux launcher
cd "$(dirname "$0")"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "==================================================="
echo "         Starting AI PDF Translator..."
echo "==================================================="

if command -v python3 &>/dev/null; then
    python3 app.py
elif command -v python &>/dev/null; then
    python app.py
else
    echo "Error: Python is not installed. Please install Python 3.10+ first."
    exit 1
fi
