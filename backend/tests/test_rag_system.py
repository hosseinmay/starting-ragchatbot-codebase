"""Integration tests for RAGSystem.

This module tests the complete query flow through RAGSystem,
including session management, source retrieval, and error propagation.
Includes both unit tests (mocked) and integration tests (real API).
"""

import os
from typing import Any
from unittest.mock import Mock, patch

import anthropic
import httpx
import pytest

from ai_generator import AIGeneratorError
from rag_system import RAGSystem


def _create_mock_request() -> httpx.Request:
    """Create a mock httpx.Request for exception testing.

    Returns:
        A mock httpx Request object.
    """
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


class TestRAGSystemQuery:
    """Integration tests for RAGSystem.query() method."""

    @pytest.fixture
    def mock_config(self) -> Mock:
        """Create mock configuration object.

        Returns:
            Mock config with all required attributes.
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
    def mock_rag_system(self, mock_config: Mock) -> RAGSystem:
        """Create RAGSystem with mocked dependencies.

        Args:
            mock_config: Mock configuration fixture

        Returns:
            RAGSystem with mocked VectorStore, AIGenerator, and SessionManager
        """
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as MockVectorStore,
            patch("rag_system.AIGenerator") as MockAIGenerator,
            patch("rag_system.SessionManager") as MockSessionManager,
        ):
            mock_vector_store = MockVectorStore.return_value
            mock_ai_generator = MockAIGenerator.return_value
            mock_session_manager = MockSessionManager.return_value

            # Configure default behaviors
            mock_ai_generator.generate_response.return_value = "Test response"
            mock_session_manager.get_conversation_history.return_value = None
            mock_session_manager.create_session.return_value = "session_1"

            rag = RAGSystem(mock_config)

            # Store mocks for test assertions
            rag._mock_vector_store = mock_vector_store
            rag._mock_ai_generator = mock_ai_generator
            rag._mock_session_manager = mock_session_manager

            return rag

    def test_query_returns_response_and_sources_tuple(
        self,
        mock_rag_system: RAGSystem,
    ) -> None:
        """Test that query returns (response, sources) tuple."""
        # Arrange
        mock_rag_system.ai_generator.generate_response.return_value = "Test response"
        mock_rag_system.tool_manager.get_last_sources = Mock(
            return_value=[{"title": "Course A", "url": "https://example.com"}]
        )
        mock_rag_system.tool_manager.reset_sources = Mock()

        # Act
        result = mock_rag_system.query("What is Claude?")

        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 2
        response, sources = result
        assert response == "Test response"
        assert len(sources) == 1
        assert sources[0]["title"] == "Course A"

    def test_query_passes_tools_to_ai_generator(
        self,
        mock_rag_system: RAGSystem,
    ) -> None:
        """Test that tools are passed to generate_response."""
        # Arrange
        mock_rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        mock_rag_system.tool_manager.reset_sources = Mock()

        # Act
        mock_rag_system.query("test query")

        # Assert
        call_kwargs = mock_rag_system.ai_generator.generate_response.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tool_manager"] is mock_rag_system.tool_manager

    def test_query_uses_session_history(
        self,
        mock_rag_system: RAGSystem,
    ) -> None:
        """Test that session history is retrieved and passed to AI."""
        # Arrange
        mock_rag_system._mock_session_manager.get_conversation_history.return_value = (
            "User: Previous\nAssistant: Answer"
        )
        mock_rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        mock_rag_system.tool_manager.reset_sources = Mock()

        # Act
        mock_rag_system.query("new question", session_id="session_1")

        # Assert
        mock_rag_system._mock_session_manager.get_conversation_history.assert_called_with(
            "session_1"
        )
        call_kwargs = mock_rag_system.ai_generator.generate_response.call_args[1]
        assert call_kwargs["conversation_history"] == "User: Previous\nAssistant: Answer"

    def test_query_adds_exchange_to_session(
        self,
        mock_rag_system: RAGSystem,
    ) -> None:
        """Test that query/response is added to session history."""
        # Arrange
        mock_rag_system.ai_generator.generate_response.return_value = "AI response"
        mock_rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        mock_rag_system.tool_manager.reset_sources = Mock()

        # Act
        mock_rag_system.query("User question", session_id="session_2")

        # Assert
        mock_rag_system._mock_session_manager.add_exchange.assert_called_once()
        call_args = mock_rag_system._mock_session_manager.add_exchange.call_args[0]
        assert call_args[0] == "session_2"
        assert "User question" in call_args[1]  # Query wrapped in prompt
        assert call_args[2] == "AI response"

    def test_query_resets_sources_after_retrieval(
        self,
        mock_rag_system: RAGSystem,
    ) -> None:
        """Test that sources are reset after being retrieved."""
        # Arrange
        mock_rag_system.tool_manager.get_last_sources = Mock(
            return_value=[{"title": "Test"}]
        )
        mock_rag_system.tool_manager.reset_sources = Mock()

        # Act
        mock_rag_system.query("test")

        # Assert
        mock_rag_system.tool_manager.reset_sources.assert_called_once()

    def test_query_without_session_id_skips_history(
        self,
        mock_rag_system: RAGSystem,
    ) -> None:
        """Test query without session_id doesn't fetch or save history."""
        # Arrange
        mock_rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        mock_rag_system.tool_manager.reset_sources = Mock()

        # Act
        mock_rag_system.query("test")

        # Assert: History not fetched, exchange not added
        call_kwargs = mock_rag_system.ai_generator.generate_response.call_args[1]
        assert call_kwargs["conversation_history"] is None
        mock_rag_system._mock_session_manager.add_exchange.assert_not_called()

    def test_query_prompt_format(
        self,
        mock_rag_system: RAGSystem,
    ) -> None:
        """Test that query is wrapped in expected prompt format."""
        # Arrange
        mock_rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        mock_rag_system.tool_manager.reset_sources = Mock()

        # Act
        mock_rag_system.query("What is computer use?")

        # Assert
        call_kwargs = mock_rag_system.ai_generator.generate_response.call_args[1]
        assert "Answer this question about course materials:" in call_kwargs["query"]
        assert "What is computer use?" in call_kwargs["query"]


class TestRAGSystemErrorPropagation:
    """Tests for error propagation through RAGSystem.

    CRITICAL: These tests verify that AIGeneratorError from AIGenerator
    propagates correctly to the caller (app.py).
    """

    @pytest.fixture
    def mock_config(self) -> Mock:
        """Create mock configuration."""
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

    def test_ai_generator_error_propagates(
        self,
        mock_config: Mock,
    ) -> None:
        """Test that AIGeneratorError propagates to caller.

        This is how 'query failed' was happening - now errors are wrapped
        in AIGeneratorError and caught by app.py for user-friendly display.
        """
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as MockAIGenerator,
            patch("rag_system.SessionManager"),
        ):
            mock_ai = MockAIGenerator.return_value
            mock_ai.generate_response.side_effect = AIGeneratorError(
                "The AI service encountered an error. Please try again.",
            )

            rag = RAGSystem(mock_config)

            # Act & Assert: AIGeneratorError should propagate
            with pytest.raises(AIGeneratorError):
                rag.query("test query")

    def test_ai_generator_auth_error_wrapped(
        self,
        mock_config: Mock,
    ) -> None:
        """Test that authentication errors are wrapped in AIGeneratorError."""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as MockAIGenerator,
            patch("rag_system.SessionManager"),
        ):
            mock_ai = MockAIGenerator.return_value
            mock_ai.generate_response.side_effect = AIGeneratorError(
                "API authentication failed. Please check the API key configuration.",
            )

            rag = RAGSystem(mock_config)

            with pytest.raises(AIGeneratorError, match="authentication failed"):
                rag.query("test query")

    def test_tool_manager_error_propagates(
        self,
        mock_config: Mock,
    ) -> None:
        """Test error when ToolManager.get_last_sources fails."""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as MockAIGenerator,
            patch("rag_system.SessionManager"),
        ):
            mock_ai = MockAIGenerator.return_value
            mock_ai.generate_response.return_value = "Response"

            rag = RAGSystem(mock_config)
            rag.tool_manager.get_last_sources = Mock(
                side_effect=RuntimeError("ToolManager error")
            )

            with pytest.raises(RuntimeError, match="ToolManager error"):
                rag.query("test")

    def test_session_manager_error_propagates(
        self,
        mock_config: Mock,
    ) -> None:
        """Test that SessionManager errors propagate."""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager") as MockSessionManager,
        ):
            mock_session = MockSessionManager.return_value
            mock_session.get_conversation_history.side_effect = RuntimeError(
                "Session error"
            )

            rag = RAGSystem(mock_config)

            with pytest.raises(RuntimeError, match="Session error"):
                rag.query("test", session_id="session_1")


class TestRAGSystemRealIntegration:
    """Integration tests that run against the real system.

    These tests require ANTHROPIC_API_KEY to be set and will make
    real API calls. Use to diagnose actual 'query failed' issues.
    """

    @pytest.fixture
    def real_rag_system(self) -> RAGSystem:
        """Create RAGSystem with real configuration.

        Returns:
            RAGSystem instance using real config (requires API key)

        Raises:
            pytest.skip: If ANTHROPIC_API_KEY not set
        """
        from config import config

        if not config.ANTHROPIC_API_KEY:
            pytest.skip("ANTHROPIC_API_KEY not set")

        return RAGSystem(config)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    )
    def test_real_query_completes_without_exception(
        self,
        real_rag_system: RAGSystem,
    ) -> None:
        """Test that a real query completes without raising exception.

        This test identifies if the real system raises exceptions that
        would cause 'query failed' in the frontend.
        """
        try:
            response, sources = real_rag_system.query("What courses are available?")
            assert response is not None
            assert isinstance(response, str)
            assert len(response) > 0
        except anthropic.AuthenticationError as e:
            pytest.fail(f"Authentication failed - check API key: {e}")
        except anthropic.RateLimitError as e:
            pytest.fail(f"Rate limited: {e}")
        except anthropic.APIError as e:
            pytest.fail(f"API error - this causes 'query failed': {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error - this causes 'query failed': {type(e).__name__}: {e}")

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    )
    def test_real_query_with_tool_use(
        self,
        real_rag_system: RAGSystem,
    ) -> None:
        """Test query that should trigger tool use (search).

        This tests the complete flow including tool execution.
        """
        try:
            # Query that should trigger search tool
            response, sources = real_rag_system.query(
                "What does the course say about computer use?"
            )
            assert response is not None
            assert isinstance(response, str)
            # Sources may or may not be populated depending on tool use
            assert isinstance(sources, list)
        except anthropic.AuthenticationError as e:
            pytest.fail(f"Authentication failed - check API key: {e}")
        except anthropic.APIError as e:
            pytest.fail(f"API error during tool use - this causes 'query failed': {e}")
        except Exception as e:
            pytest.fail(f"Error during tool use: {type(e).__name__}: {e}")

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    )
    def test_real_query_with_session(
        self,
        real_rag_system: RAGSystem,
    ) -> None:
        """Test query with session management.

        Tests the session creation and history flow.
        """
        try:
            # Create a session
            session_id = real_rag_system.session_manager.create_session()

            # First query
            response1, _ = real_rag_system.query("Hello", session_id=session_id)
            assert response1 is not None

            # Second query with context
            response2, _ = real_rag_system.query(
                "What did I just say?",
                session_id=session_id,
            )
            assert response2 is not None
        except Exception as e:
            pytest.fail(f"Session test failed: {type(e).__name__}: {e}")


class TestRAGSystemInitialization:
    """Tests for RAGSystem initialization."""

    @pytest.fixture
    def mock_config(self) -> Mock:
        """Create mock configuration."""
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

    def test_initialization_creates_all_components(
        self,
        mock_config: Mock,
    ) -> None:
        """Test that RAGSystem initializes all required components."""
        with (
            patch("rag_system.DocumentProcessor") as MockDocProcessor,
            patch("rag_system.VectorStore") as MockVectorStore,
            patch("rag_system.AIGenerator") as MockAIGenerator,
            patch("rag_system.SessionManager") as MockSessionManager,
        ):
            rag = RAGSystem(mock_config)

            # Assert all components initialized
            MockDocProcessor.assert_called_once_with(800, 100)
            MockVectorStore.assert_called_once_with(
                "./test_chroma_db",
                "all-MiniLM-L6-v2",
                5,
            )
            MockAIGenerator.assert_called_once_with(
                "test-api-key",
                "claude-sonnet-4-20250514",
            )
            MockSessionManager.assert_called_once_with(2)

    def test_initialization_registers_tools(
        self,
        mock_config: Mock,
    ) -> None:
        """Test that RAGSystem registers search and outline tools."""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
        ):
            rag = RAGSystem(mock_config)

            # Assert tools are registered
            tool_definitions = rag.tool_manager.get_tool_definitions()
            tool_names = [t["name"] for t in tool_definitions]

            assert "search_course_content" in tool_names
            assert "get_course_outline" in tool_names
