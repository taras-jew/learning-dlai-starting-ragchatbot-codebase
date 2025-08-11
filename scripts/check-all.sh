#!/bin/bash

echo "Running complete code quality pipeline..."

echo "========================================="
echo "STEP 1: Code Formatting"
echo "========================================="
./scripts/format.sh

echo ""
echo "========================================="
echo "STEP 2: Code Quality Checks"
echo "========================================="
./scripts/lint.sh

echo ""
echo "========================================="
echo "STEP 3: Running Tests"
echo "========================================="
uv run pytest backend/tests/ -v

echo ""
echo "========================================="
echo "Quality pipeline complete!"
echo "========================================="