#!/bin/bash
# Run type checking with mypy

set -e

echo "Running mypy..."
uv run mypy backend/ main.py

echo "Type checking complete!"
