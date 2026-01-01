"""Shared pytest fixtures for RAG chatbot tests.

This module provides reusable fixtures for testing the RAG system components
including mock VectorStore responses, tool definitions, and Anthropic client mocks.
Also provides API testing infrastructure with TestClient and mock dependencies.
"""

import sys
from pathlib import Path
from typing import Any, Generator
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add backend to path for imports
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from vector_store import SearchResults  # noqa: E402


# =============================================================================
# Mock Configuration Fixtures
# =============================================================================


@pytest.fixture
def mock_config() -> Mock:
    """Create a mock configuration object with all required attributes.

    Returns:
        Mock config matching the structure expected by RAGSystem and components.
    """
    config = Mock()
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.CHROMA_PATH = "./test_chroma_db"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.MAX_RESULTS = 5
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.MAX_HISTORY = 2
    return config


@pytest.fixture
def mock_rag_system(mock_config: Mock) -> Mock:
    """Create a fully mocked RAGSystem for API testing.

    Args:
        mock_config: Mock configuration fixture.

    Returns:
        Mock RAGSystem with all methods configured for testing.
    """
    rag = Mock()

    # Configure session_manager
    rag.session_manager = Mock()
    rag.session_manager.create_session.return_value = "test_session_123"
    rag.session_manager.get_conversation_history.return_value = None
    rag.session_manager.clear_session.return_value = None

    # Configure query method
    rag.query.return_value = (
        "This is a test response about course materials.",
        [{"title": "Test Course - Lesson 1", "url": "https://example.com/lesson1"}],
    )

    # Configure get_course_analytics
    rag.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Course A", "Course B", "Course C"],
    }

    return rag


# =============================================================================
# API Testing Fixtures
# =============================================================================


def create_test_app(mock_rag: Mock) -> FastAPI:
    """Create a FastAPI app for testing without static file mounting.

    This function creates a minimal FastAPI app with the same endpoints
    as the production app but without the static file mount that would
    fail in test environments.

    Args:
        mock_rag: Mock RAGSystem to use for handling requests.

    Returns:
        FastAPI app configured for testing.
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional

    app = FastAPI(title="Course Materials RAG System - Test")

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceItem(BaseModel):
        title: str
        url: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceItem]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest) -> QueryResponse:
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag.session_manager.create_session()

            answer, sources = mock_rag.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id,
            )
        except Exception as e:
            if hasattr(e, "__class__") and "AIGeneratorError" in e.__class__.__name__:
                session_id = request.session_id or mock_rag.session_manager.create_session()
                return QueryResponse(
                    answer=str(e),
                    sources=[],
                    session_id=session_id,
                )
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats() -> CourseStats:
        try:
            analytics = mock_rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def clear_session(session_id: str) -> dict[str, str]:
        mock_rag.session_manager.clear_session(session_id)
        return {"status": "cleared", "session_id": session_id}

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"status": "ok", "message": "RAG System API"}

    return app


@pytest.fixture
def test_app(mock_rag_system: Mock) -> FastAPI:
    """Create a test FastAPI application.

    Args:
        mock_rag_system: Mock RAGSystem fixture.

    Returns:
        FastAPI app configured for testing.
    """
    return create_test_app(mock_rag_system)


@pytest.fixture
def test_client(test_app: FastAPI) -> Generator[TestClient, None, None]:
    """Create a TestClient for API testing.

    Args:
        test_app: Test FastAPI application fixture.

    Yields:
        TestClient instance for making HTTP requests.
    """
    with TestClient(test_app) as client:
        yield client


@pytest.fixture
def test_client_with_rag(mock_rag_system: Mock) -> Generator[tuple[TestClient, Mock], None, None]:
    """Create a TestClient along with access to the mock RAG system.

    This fixture is useful when you need to modify mock behavior
    or verify calls made to the RAG system during tests.

    Args:
        mock_rag_system: Mock RAGSystem fixture.

    Yields:
        Tuple of (TestClient, Mock RAGSystem).
    """
    app = create_test_app(mock_rag_system)
    with TestClient(app) as client:
        yield client, mock_rag_system


@pytest.fixture
def sample_search_results() -> SearchResults:
    """Create sample SearchResults for testing.

    Returns:
        SearchResults with realistic test data including documents,
        metadata, and distance scores.
    """
    return SearchResults(
        documents=[
            "Claude is an AI assistant made by Anthropic.",
            "Computer use allows Claude to interact with computers.",
        ],
        metadata=[
            {"course_title": "Building Towards Computer Use", "lesson_number": 0},
            {"course_title": "Building Towards Computer Use", "lesson_number": 1},
        ],
        distances=[0.15, 0.25],
        error=None,
    )


@pytest.fixture
def empty_search_results() -> SearchResults:
    """Create empty SearchResults for testing no-match scenarios.

    Returns:
        SearchResults with empty lists and no error.
    """
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None,
    )


@pytest.fixture
def error_search_results() -> SearchResults:
    """Create SearchResults with error for testing error propagation.

    Returns:
        SearchResults with error message set.
    """
    return SearchResults.empty("Search error: ChromaDB connection failed")


@pytest.fixture
def sample_tool_definitions() -> list[dict[str, Any]]:
    """Create sample Anthropic tool definitions.

    Returns:
        List of tool definitions matching the actual CourseSearchTool
        and CourseOutlineTool schemas.
    """
    return [
        {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the course content",
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work)",
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_course_outline",
            "description": "Get complete course outline including lesson list",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_title": {
                        "type": "string",
                        "description": "Course title or partial name",
                    },
                },
                "required": ["course_title"],
            },
        },
    ]


@pytest.fixture
def mock_vector_store() -> Mock:
    """Create a mock VectorStore with default successful responses.

    Returns:
        Mock VectorStore configured with search, get_lesson_link,
        and other commonly used methods.
    """
    store = Mock()
    store.search.return_value = SearchResults(
        documents=["Test content about Claude API"],
        metadata=[{"course_title": "Test Course", "lesson_number": 1}],
        distances=[0.1],
        error=None,
    )
    store.get_lesson_link.return_value = "https://example.com/lesson"
    store._resolve_course_name.return_value = "Test Course"
    store.get_course_outline.return_value = {
        "title": "Test Course",
        "course_link": "https://example.com/course",
        "instructor": "Test Instructor",
        "lesson_count": 3,
        "lessons": [
            {
                "lesson_number": 0,
                "lesson_title": "Intro",
                "lesson_link": "https://example.com/0",
            },
            {
                "lesson_number": 1,
                "lesson_title": "Basics",
                "lesson_link": "https://example.com/1",
            },
            {
                "lesson_number": 2,
                "lesson_title": "Advanced",
                "lesson_link": "https://example.com/2",
            },
        ],
    }
    return store


@pytest.fixture
def mock_anthropic_response() -> Mock:
    """Create a mock successful Anthropic API response.

    Returns:
        Mock response object with text content.
    """
    response = Mock()
    response.stop_reason = "end_turn"
    text_block = Mock()
    text_block.type = "text"
    text_block.text = "This is a test response from Claude."
    response.content = [text_block]
    return response


@pytest.fixture
def mock_anthropic_tool_use_response() -> Mock:
    """Create a mock Anthropic response requesting tool use.

    Returns:
        Mock response with tool_use content block.
    """
    response = Mock()
    response.stop_reason = "tool_use"

    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.id = "tool_call_123"
    tool_block.name = "search_course_content"
    tool_block.input = {"query": "What is Claude?"}

    response.content = [tool_block]
    return response


def pytest_configure(config: pytest.Config) -> None:
    """Configure custom pytest markers.

    Args:
        config: pytest configuration object
    """
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (require API key)",
    )
