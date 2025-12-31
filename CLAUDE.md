# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG (Retrieval-Augmented Generation) chatbot for answering questions about course materials. Uses semantic search with ChromaDB and Claude AI for response generation.

## Commands

```bash
# Install dependencies
uv sync

# Run the application (starts on http://localhost:8000)
./run.sh
# Or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

## Architecture

```
User Query → FastAPI (/api/query) → RAG System
                                        ├── Session Manager (conversation history)
                                        ├── AI Generator (Claude with tool calling)
                                        │       └── decides: answer directly OR search first
                                        ├── Tool Manager → CourseSearchTool
                                        │                      └── Vector Store (ChromaDB)
                                        └── Document Processor (on startup)
```

### Key Components

- **`backend/rag_system.py`**: Main orchestrator. Coordinates document loading, querying, and prevents duplicate course processing.

- **`backend/ai_generator.py`**: Claude integration with tool calling. Claude decides whether to search course content or answer from general knowledge. Temperature=0, max_tokens=800.

- **`backend/vector_store.py`**: ChromaDB wrapper with two collections:
  - `course_catalog`: Course metadata (titles, instructors)
  - `course_content`: Chunked course material for semantic search

- **`backend/document_processor.py`**: Parses course files, extracts metadata (title, instructor, lessons), chunks text into 800-char pieces with 100-char overlap.

- **`backend/search_tools.py`**: Tool definitions for Claude. `CourseSearchTool` performs vector search with optional course/lesson filtering.

- **`backend/session_manager.py`**: Maintains conversation history per session (limited to last 2 exchanges).

### Document Format

Course files in `docs/` follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [title]
Lesson Link: [url]
[content...]

Lesson 1: [title]
[content...]
```

### Configuration

Settings in `backend/config.py`:
- `CHUNK_SIZE`: 800 chars
- `CHUNK_OVERLAP`: 100 chars
- `MAX_RESULTS`: 5 search results
- `MAX_HISTORY`: 2 conversation exchanges
- `EMBEDDING_MODEL`: all-MiniLM-L6-v2
- `ANTHROPIC_MODEL`: claude-sonnet-4-20250514
