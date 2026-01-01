"""Tests for CourseSearchTool and ToolManager classes.

This module tests the search tool execution, error propagation,
empty results handling, and source tracking functionality.
"""

from typing import Any
from unittest.mock import Mock

import pytest

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute() method."""

    @pytest.fixture
    def search_tool(self, mock_vector_store: Mock) -> CourseSearchTool:
        """Create CourseSearchTool with mocked vector store.

        Args:
            mock_vector_store: Pytest fixture providing mock VectorStore

        Returns:
            CourseSearchTool instance with mocked dependencies
        """
        return CourseSearchTool(mock_vector_store)

    def test_successful_search_returns_formatted_results(
        self,
        search_tool: CourseSearchTool,
        mock_vector_store: Mock,
    ) -> None:
        """Test that successful search returns properly formatted content."""
        # Arrange
        mock_results = SearchResults(
            documents=["This is content about Claude API"],
            metadata=[{"course_title": "Building Towards Computer Use", "lesson_number": 1}],
            distances=[0.15],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results

        # Act
        result = search_tool.execute(query="Claude API")

        # Assert
        assert "[Building Towards Computer Use - Lesson 1]" in result
        assert "This is content about Claude API" in result
        mock_vector_store.search.assert_called_once_with(
            query="Claude API",
            course_name=None,
            lesson_number=None,
        )

    def test_search_with_course_filter(
        self,
        search_tool: CourseSearchTool,
        mock_vector_store: Mock,
    ) -> None:
        """Test that course_name filter is passed to VectorStore."""
        # Arrange
        mock_results = SearchResults(
            documents=["MCP Content"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 2}],
            distances=[0.1],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results

        # Act
        search_tool.execute(query="test", course_name="MCP")

        # Assert
        mock_vector_store.search.assert_called_once_with(
            query="test",
            course_name="MCP",
            lesson_number=None,
        )

    def test_search_with_lesson_filter(
        self,
        search_tool: CourseSearchTool,
        mock_vector_store: Mock,
    ) -> None:
        """Test that lesson_number filter is passed correctly."""
        # Arrange
        mock_results = SearchResults(
            documents=["Lesson 3 content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 3}],
            distances=[0.2],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results

        # Act
        search_tool.execute(query="test", lesson_number=3)

        # Assert
        mock_vector_store.search.assert_called_once_with(
            query="test",
            course_name=None,
            lesson_number=3,
        )

    def test_search_with_both_filters(
        self,
        search_tool: CourseSearchTool,
        mock_vector_store: Mock,
    ) -> None:
        """Test combining course and lesson filters."""
        # Arrange
        mock_results = SearchResults(
            documents=["Specific content"],
            metadata=[{"course_title": "Computer Use", "lesson_number": 5}],
            distances=[0.05],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results

        # Act
        search_tool.execute(
            query="computer use",
            course_name="Computer Use",
            lesson_number=5,
        )

        # Assert
        mock_vector_store.search.assert_called_once_with(
            query="computer use",
            course_name="Computer Use",
            lesson_number=5,
        )

    def test_error_propagation_from_vector_store(
        self,
        search_tool: CourseSearchTool,
        mock_vector_store: Mock,
    ) -> None:
        """Test that VectorStore errors are returned as error strings."""
        # Arrange
        mock_results = SearchResults.empty("Search error: ChromaDB connection failed")
        mock_vector_store.search.return_value = mock_results

        # Act
        result = search_tool.execute(query="test query")

        # Assert
        assert result == "Search error: ChromaDB connection failed"

    def test_course_not_found_error(
        self,
        search_tool: CourseSearchTool,
        mock_vector_store: Mock,
    ) -> None:
        """Test handling of course name resolution failure."""
        # Arrange
        mock_results = SearchResults.empty("No course found matching 'NonExistentCourse'")
        mock_vector_store.search.return_value = mock_results

        # Act
        result = search_tool.execute(query="test", course_name="NonExistentCourse")

        # Assert
        assert "No course found matching" in result
        assert "NonExistentCourse" in result

    def test_empty_results_handling(
        self,
        search_tool: CourseSearchTool,
        mock_vector_store: Mock,
    ) -> None:
        """Test that empty results return appropriate message."""
        # Arrange
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results

        # Act
        result = search_tool.execute(query="obscure topic")

        # Assert
        assert "No relevant content found" in result

    def test_empty_results_with_course_filter_shows_filter_info(
        self,
        search_tool: CourseSearchTool,
        mock_vector_store: Mock,
    ) -> None:
        """Test that empty results message includes course filter context."""
        # Arrange
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results

        # Act
        result = search_tool.execute(query="test", course_name="MCP Course")

        # Assert
        assert "No relevant content found" in result
        assert "MCP Course" in result

    def test_empty_results_with_lesson_filter_shows_filter_info(
        self,
        search_tool: CourseSearchTool,
        mock_vector_store: Mock,
    ) -> None:
        """Test that empty results message includes lesson filter context."""
        # Arrange
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results

        # Act
        result = search_tool.execute(query="test", lesson_number=5)

        # Assert
        assert "No relevant content found" in result
        assert "lesson 5" in result

    def test_sources_tracking(
        self,
        search_tool: CourseSearchTool,
        mock_vector_store: Mock,
    ) -> None:
        """Test that last_sources is populated correctly after search."""
        # Arrange
        mock_results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"

        # Act
        search_tool.execute(query="test")

        # Assert
        assert len(search_tool.last_sources) == 2
        assert search_tool.last_sources[0]["title"] == "Course A - Lesson 1"
        assert search_tool.last_sources[1]["title"] == "Course B - Lesson 2"

    def test_sources_include_lesson_link(
        self,
        search_tool: CourseSearchTool,
        mock_vector_store: Mock,
    ) -> None:
        """Test that sources include lesson URLs from vector store."""
        # Arrange
        mock_results = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson-1"

        # Act
        search_tool.execute(query="test")

        # Assert
        assert search_tool.last_sources[0]["url"] == "https://example.com/lesson-1"
        mock_vector_store.get_lesson_link.assert_called_with("Test Course", 1)

    def test_multiple_results_formatted_separately(
        self,
        search_tool: CourseSearchTool,
        mock_vector_store: Mock,
    ) -> None:
        """Test that multiple results are formatted with separators."""
        # Arrange
        mock_results = SearchResults(
            documents=["First result content", "Second result content"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results

        # Act
        result = search_tool.execute(query="test")

        # Assert
        assert "[Course A - Lesson 1]" in result
        assert "[Course B - Lesson 2]" in result
        assert "First result content" in result
        assert "Second result content" in result
        # Results should be separated by double newline
        assert "\n\n" in result


class TestCourseOutlineTool:
    """Tests for CourseOutlineTool.execute() method."""

    @pytest.fixture
    def outline_tool(self, mock_vector_store: Mock) -> CourseOutlineTool:
        """Create CourseOutlineTool with mocked vector store.

        Args:
            mock_vector_store: Pytest fixture providing mock VectorStore

        Returns:
            CourseOutlineTool instance with mocked dependencies
        """
        return CourseOutlineTool(mock_vector_store)

    def test_successful_outline_returns_formatted_content(
        self,
        outline_tool: CourseOutlineTool,
        mock_vector_store: Mock,
    ) -> None:
        """Test that successful outline retrieval returns formatted content."""
        # Arrange
        mock_vector_store._resolve_course_name.return_value = "Test Course"
        mock_vector_store.get_course_outline.return_value = {
            "title": "Test Course",
            "course_link": "https://example.com/course",
            "instructor": "Test Instructor",
            "lesson_count": 2,
            "lessons": [
                {"lesson_number": 0, "lesson_title": "Introduction"},
                {"lesson_number": 1, "lesson_title": "Basics"},
            ],
        }

        # Act
        result = outline_tool.execute(course_title="Test")

        # Assert
        assert "Course: Test Course" in result
        assert "Course Link: https://example.com/course" in result
        assert "Total Lessons: 2" in result
        assert "0. Introduction" in result
        assert "1. Basics" in result

    def test_course_not_found_returns_error(
        self,
        outline_tool: CourseOutlineTool,
        mock_vector_store: Mock,
    ) -> None:
        """Test that unresolvable course name returns error message."""
        # Arrange
        mock_vector_store._resolve_course_name.return_value = None

        # Act
        result = outline_tool.execute(course_title="NonExistent")

        # Assert
        assert "No course found matching 'NonExistent'" in result

    def test_outline_sources_tracked(
        self,
        outline_tool: CourseOutlineTool,
        mock_vector_store: Mock,
    ) -> None:
        """Test that sources are tracked for outline requests."""
        # Arrange
        mock_vector_store._resolve_course_name.return_value = "Test Course"
        mock_vector_store.get_course_outline.return_value = {
            "title": "Test Course",
            "course_link": "https://example.com/course",
            "lessons": [],
        }

        # Act
        outline_tool.execute(course_title="Test")

        # Assert
        assert len(outline_tool.last_sources) == 1
        assert outline_tool.last_sources[0]["title"] == "Test Course"
        assert outline_tool.last_sources[0]["url"] == "https://example.com/course"


class TestToolManager:
    """Tests for ToolManager class."""

    def test_register_tool_stores_by_name(self, mock_vector_store: Mock) -> None:
        """Test that tools are registered with correct names."""
        # Arrange
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        # Act
        manager.register_tool(tool)

        # Assert
        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] is tool

    def test_register_tool_raises_on_missing_name(self) -> None:
        """Test that registering tool without name raises ValueError."""
        # Arrange
        manager = ToolManager()
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"description": "No name field"}

        # Act & Assert
        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            manager.register_tool(mock_tool)

    def test_execute_tool_not_found_returns_error(self) -> None:
        """Test executing non-existent tool returns error message."""
        # Arrange
        manager = ToolManager()

        # Act
        result = manager.execute_tool("nonexistent", query="test")

        # Assert
        assert "not found" in result.lower()
        assert "nonexistent" in result.lower()

    def test_execute_tool_calls_correct_tool(
        self,
        mock_vector_store: Mock,
    ) -> None:
        """Test that execute_tool passes kwargs correctly to tool."""
        # Arrange
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = SearchResults(
            documents=["Result"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )
        manager.register_tool(tool)

        # Act
        result = manager.execute_tool("search_course_content", query="test", course_name="MCP")

        # Assert
        mock_vector_store.search.assert_called_once_with(
            query="test",
            course_name="MCP",
            lesson_number=None,
        )
        assert "[Test - Lesson 1]" in result

    def test_get_tool_definitions_returns_all_schemas(
        self,
        mock_vector_store: Mock,
    ) -> None:
        """Test that get_tool_definitions returns all registered tool schemas."""
        # Arrange
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        # Act
        definitions = manager.get_tool_definitions()

        # Assert
        assert len(definitions) == 2
        names = [d["name"] for d in definitions]
        assert "search_course_content" in names
        assert "get_course_outline" in names

    def test_get_last_sources_returns_tool_sources(
        self,
        mock_vector_store: Mock,
    ) -> None:
        """Test that get_last_sources retrieves sources from tools."""
        # Arrange
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        tool.last_sources = [{"title": "Test Source", "url": "https://example.com"}]
        manager.register_tool(tool)

        # Act
        sources = manager.get_last_sources()

        # Assert
        assert len(sources) == 1
        assert sources[0]["title"] == "Test Source"

    def test_get_last_sources_returns_empty_when_no_sources(
        self,
        mock_vector_store: Mock,
    ) -> None:
        """Test that get_last_sources returns empty list when no sources exist."""
        # Arrange
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        tool.last_sources = []
        manager.register_tool(tool)

        # Act
        sources = manager.get_last_sources()

        # Assert
        assert sources == []

    def test_reset_sources_clears_all_tool_sources(
        self,
        mock_vector_store: Mock,
    ) -> None:
        """Test that reset_sources clears sources from all tools."""
        # Arrange
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)
        search_tool.last_sources = [{"title": "Source 1"}]
        outline_tool.last_sources = [{"title": "Source 2"}]
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        # Act
        manager.reset_sources()

        # Assert
        assert search_tool.last_sources == []
        assert outline_tool.last_sources == []
