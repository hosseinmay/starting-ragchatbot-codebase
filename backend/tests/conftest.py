"""Shared pytest fixtures for RAG chatbot tests.

This module provides reusable fixtures for testing the RAG system components
including mock VectorStore responses, tool definitions, and Anthropic client mocks.
"""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

# Add backend to path for imports
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from vector_store import SearchResults  # noqa: E402


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
            {"lesson_number": 0, "lesson_title": "Intro", "lesson_link": "https://example.com/0"},
            {"lesson_number": 1, "lesson_title": "Basics", "lesson_link": "https://example.com/1"},
            {"lesson_number": 2, "lesson_title": "Advanced", "lesson_link": "https://example.com/2"},
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
