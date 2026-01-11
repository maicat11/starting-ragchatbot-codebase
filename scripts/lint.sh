#!/bin/bash
# Run code quality checks (linting)

set -e

echo "Running flake8..."
uv run flake8 backend/ main.py

echo "Linting complete!"
