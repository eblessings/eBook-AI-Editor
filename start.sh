#!/bin/bash
# Quick start script for eBook Editor Pro

echo "🚀 Starting eBook Editor Pro..."

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️ Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Start the server
python main.py
