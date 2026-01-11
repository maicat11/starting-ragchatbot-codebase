#!/bin/bash
# Run all code quality checks

set -e

echo "================================"
echo "Running Code Quality Checks"
echo "================================"
echo ""

echo "1. Checking code formatting..."
uv run black --check backend/ main.py
uv run isort --check-only backend/ main.py
echo "✓ Formatting check passed"
echo ""

echo "2. Running linter..."
uv run flake8 backend/ main.py
echo "✓ Linting passed"
echo ""

echo "3. Running type checker (optional - may show warnings)..."
uv run mypy backend/ main.py || echo "⚠ Type checking found some warnings (non-blocking)"
echo ""

echo "================================"
echo "Core quality checks passed!"
echo "================================"
