# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Start the application:**
```bash
# Quick start using provided script
./run.sh

# Manual start (from root directory)
cd backend && uv run uvicorn app:app --reload --port 8000
```

**Install dependencies:**
```bash
uv sync
```

**Code quality tools:**
```bash
# Format code (black + isort)
./scripts/format.sh

# Run quality checks (flake8, isort, black)
./scripts/lint.sh

# Run complete quality pipeline (format + lint + test)
./scripts/check-all.sh

# Individual commands
uv run black backend/ main.py          # Format code
uv run isort backend/ main.py          # Sort imports
uv run flake8 backend/ main.py         # Lint code
uv run pytest backend/tests/ -v       # Run tests
```

**Environment setup:**
Create `.env` file in root with:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**Access points:**
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Development Notes

- Uses `uv` for Python dependency management
- ChromaDB data persists in `backend/chroma_db/`
- Course documents are automatically loaded on server startup
- Vector embeddings use sentence-transformers
- Frontend uses relative API paths for proxy compatibility
- Test framework: pytest configured with tests in backend/tests/
- Use `uv` to run python files
- Dont run the server using ./run.sh I will start it myself

## Architecture Overview

The rest of the file remains unchanged...