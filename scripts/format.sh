#!/bin/bash

echo "Running code formatting..."

echo "1. Formatting Python code with black..."
uv run black backend/ main.py

echo "2. Sorting imports with isort..."
uv run isort backend/ main.py

echo "Code formatting complete!"