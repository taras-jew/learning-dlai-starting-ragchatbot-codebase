#!/bin/bash

echo "Running code quality checks..."

echo "1. Checking code style with flake8..."
uv run flake8 backend/ main.py

echo "2. Checking import sorting with isort..."
uv run isort --check-only backend/ main.py

echo "3. Checking code formatting with black..."
uv run black --check backend/ main.py

echo "Code quality checks complete!"