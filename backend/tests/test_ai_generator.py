"""Tests for AIGenerator class.

This module tests the AI response generation, tool calling flow,
and critically, error propagation from the Anthropic API.
These tests help identify where 'query failed' errors originate.
"""

from typing import Any
from unittest.mock import Mock, patch

import anthropic
import httpx
import pytest

from ai_generator import AIGenerator, AIGeneratorError


def _create_mock_request() -> httpx.Request:
    """Create a mock httpx.Request for exception testing.

    Returns:
        A mock httpx Request object.
    """
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


class TestAIGeneratorConfiguration:
    """Tests for AIGenerator initialization and configuration."""

    def test_temperature_set_to_zero(self) -> None:
        """Verify temperature is set to 0 for deterministic responses."""
        with patch("ai_generator.anthropic.Anthropic"):
            generator = AIGenerator(api_key="test-key", model="test-model")

            assert generator.base_params["temperature"] == 0

    def test_max_tokens_set_correctly(self) -> None:
        """Verify max_tokens is set to 800."""
        with patch("ai_generator.anthropic.Anthropic"):
            generator = AIGenerator(api_key="test-key", model="test-model")

            assert generator.base_params["max_tokens"] == 800

    def test_model_name_preserved(self) -> None:
        """Test that model name is stored correctly in params."""
        with patch("ai_generator.anthropic.Anthropic"):
            generator = AIGenerator(
                api_key="test-key", model="claude-sonnet-4-20250514"
            )

            assert generator.model == "claude-sonnet-4-20250514"
            assert generator.base_params["model"] == "claude-sonnet-4-20250514"


class TestAIGeneratorGenerateResponse:
    """Tests for AIGenerator.generate_response() method."""

    @pytest.fixture
    def mock_anthropic_client(self) -> Mock:
        """Create a mock Anthropic client.

        Returns:
            Mock Anthropic client with messages.create method.
        """
        client = Mock()
        client.messages = Mock()
        return client

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client: Mock) -> AIGenerator:
        """Create AIGenerator with mocked Anthropic client.

        Args:
            mock_anthropic_client: Mocked Anthropic API client

        Returns:
            AIGenerator instance with mocked dependencies
        """
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic_class:
            mock_anthropic_class.return_value = mock_anthropic_client
            generator = AIGenerator(
                api_key="test-api-key",
                model="claude-sonnet-4-20250514",
            )
            generator.client = mock_anthropic_client
            return generator

    @pytest.fixture
    def mock_tool_manager(self) -> Mock:
        """Create a mock ToolManager.

        Returns:
            Mock ToolManager with execute_tool method.
        """
        manager = Mock()
        manager.execute_tool.return_value = "Tool execution result"
        return manager

    @pytest.fixture
    def sample_tools(self) -> list[dict[str, Any]]:
        """Sample tool definitions for testing.

        Returns:
            List of Anthropic tool definition dicts.
        """
        return [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
        ]

    def test_direct_response_returned(
        self,
        ai_generator: AIGenerator,
        mock_anthropic_client: Mock,
    ) -> None:
        """Test that direct text response is returned when no tool use."""
        # Arrange
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(type="text", text="Direct answer from Claude")]
        mock_anthropic_client.messages.create.return_value = mock_response

        # Act
        result = ai_generator.generate_response(query="What is Python?")

        # Assert
        assert result == "Direct answer from Claude"

    def test_tools_passed_to_api_call(
        self,
        ai_generator: AIGenerator,
        mock_anthropic_client: Mock,
        sample_tools: list[dict[str, Any]],
    ) -> None:
        """Test that tools are included in API call when provided."""
        # Arrange
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(type="text", text="Response")]
        mock_anthropic_client.messages.create.return_value = mock_response

        # Act
        ai_generator.generate_response(query="What is Claude?", tools=sample_tools)

        # Assert
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == sample_tools

    def test_tool_choice_set_to_auto(
        self,
        ai_generator: AIGenerator,
        mock_anthropic_client: Mock,
        sample_tools: list[dict[str, Any]],
    ) -> None:
        """Test that tool_choice is set to auto when tools provided."""
        # Arrange
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(type="text", text="Response")]
        mock_anthropic_client.messages.create.return_value = mock_response

        # Act
        ai_generator.generate_response(query="test", tools=sample_tools)

        # Assert
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    def test_no_tools_omits_tool_params(
        self,
        ai_generator: AIGenerator,
        mock_anthropic_client: Mock,
    ) -> None:
        """Test that API call without tools doesn't include tool params."""
        # Arrange
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(type="text", text="Response")]
        mock_anthropic_client.messages.create.return_value = mock_response

        # Act
        ai_generator.generate_response(query="General question")

        # Assert
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs

    def test_conversation_history_included_in_system(
        self,
        ai_generator: AIGenerator,
        mock_anthropic_client: Mock,
    ) -> None:
        """Test that conversation history is appended to system prompt."""
        # Arrange
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(type="text", text="Response")]
        mock_anthropic_client.messages.create.return_value = mock_response

        history = "User: Previous question\nAssistant: Previous answer"

        # Act
        ai_generator.generate_response(query="test", conversation_history=history)

        # Assert
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert "Previous conversation:" in call_kwargs["system"]
        assert history in call_kwargs["system"]

    def test_no_history_uses_base_system_prompt(
        self,
        ai_generator: AIGenerator,
        mock_anthropic_client: Mock,
    ) -> None:
        """Test that system prompt is used alone when no history."""
        # Arrange
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(type="text", text="Response")]
        mock_anthropic_client.messages.create.return_value = mock_response

        # Act
        ai_generator.generate_response(query="test")

        # Assert
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert "Previous conversation:" not in call_kwargs["system"]
        assert "AI assistant specialized in course materials" in call_kwargs["system"]


class TestAIGeneratorToolExecution:
    """Tests for tool execution flow in AIGenerator."""

    @pytest.fixture
    def mock_anthropic_client(self) -> Mock:
        """Create a mock Anthropic client."""
        client = Mock()
        client.messages = Mock()
        return client

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client: Mock) -> AIGenerator:
        """Create AIGenerator with mocked client."""
        with patch("ai_generator.anthropic.Anthropic") as mock_class:
            mock_class.return_value = mock_anthropic_client
            generator = AIGenerator(api_key="test-key", model="test-model")
            generator.client = mock_anthropic_client
            return generator

    @pytest.fixture
    def sample_tools(self) -> list[dict[str, Any]]:
        """Sample tool definitions."""
        return [{"name": "search_course_content", "input_schema": {"type": "object"}}]

    def test_tool_execution_calls_tool_manager(
        self,
        ai_generator: AIGenerator,
        mock_anthropic_client: Mock,
        sample_tools: list[dict[str, Any]],
    ) -> None:
        """Test that tool execution calls ToolManager.execute_tool."""
        # Arrange: First response requests tool use
        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "search_course_content"
        tool_use_block.id = "tool_123"
        tool_use_block.input = {"query": "Claude API"}

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_block]

        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [
            Mock(type="text", text="Final answer based on search")
        ]

        mock_anthropic_client.messages.create.side_effect = [
            first_response,
            final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results here"

        # Act
        result = ai_generator.generate_response(
            query="What is Claude API?",
            tools=sample_tools,
            tool_manager=mock_tool_manager,
        )

        # Assert
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Claude API",
        )
        assert result == "Final answer based on search"

    def test_tool_results_sent_to_api(
        self,
        ai_generator: AIGenerator,
        mock_anthropic_client: Mock,
        sample_tools: list[dict[str, Any]],
    ) -> None:
        """Test that tool results are included in follow-up API call."""
        # Arrange
        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "search_course_content"
        tool_use_block.id = "tool_456"
        tool_use_block.input = {"query": "test"}

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_block]

        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(type="text", text="Answer")]

        mock_anthropic_client.messages.create.side_effect = [
            first_response,
            final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results here"

        # Act
        ai_generator.generate_response(
            query="test",
            tools=sample_tools,
            tool_manager=mock_tool_manager,
        )

        # Assert: Check second API call includes tool results
        second_call = mock_anthropic_client.messages.create.call_args_list[1]
        messages = second_call[1]["messages"]

        # Should have: original user message, assistant tool_use, user tool_result
        assert len(messages) == 3
        assert messages[2]["role"] == "user"

        tool_result = messages[2]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "tool_456"
        assert tool_result["content"] == "Search results here"

    def test_no_tool_execution_without_tool_manager(
        self,
        ai_generator: AIGenerator,
        mock_anthropic_client: Mock,
        sample_tools: list[dict[str, Any]],
    ) -> None:
        """Test that tool use response is returned directly if no tool_manager."""
        # Arrange: Response requests tool use but no manager provided
        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.text = "Tool use block"

        mock_response = Mock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [Mock(type="text", text="Response text")]
        mock_anthropic_client.messages.create.return_value = mock_response

        # Act
        result = ai_generator.generate_response(
            query="test",
            tools=sample_tools,
            tool_manager=None,  # No tool manager
        )

        # Assert: Should return the text content, not execute tools
        assert result == "Response text"


class TestAIGeneratorErrorPropagation:
    """Tests for error handling in AIGenerator.

    CRITICAL: These tests verify that API errors are wrapped in
    AIGeneratorError with user-friendly messages.
    """

    @pytest.fixture
    def ai_generator(self) -> AIGenerator:
        """Create AIGenerator with mocked client initialization."""
        with patch("ai_generator.anthropic.Anthropic") as mock_class:
            mock_client = Mock()
            mock_class.return_value = mock_client
            generator = AIGenerator(api_key="test-key", model="test-model")
            generator.client = mock_client
            return generator

    def test_api_authentication_error_wrapped(
        self,
        ai_generator: AIGenerator,
    ) -> None:
        """Test that AuthenticationError is wrapped in AIGeneratorError.

        This test verifies that invalid API key errors are wrapped
        with user-friendly messages instead of raw exceptions.
        """
        # Arrange: Simulate API key error
        ai_generator.client.messages.create.side_effect = anthropic.AuthenticationError(
            "Invalid API Key",
            response=Mock(status_code=401),
            body={"error": {"message": "Invalid API Key"}},
        )

        # Act & Assert: Error should be wrapped in AIGeneratorError
        with pytest.raises(AIGeneratorError, match="authentication failed"):
            ai_generator.generate_response(query="test")

    def test_api_rate_limit_error_wrapped(
        self,
        ai_generator: AIGenerator,
    ) -> None:
        """Test that RateLimitError is wrapped in AIGeneratorError."""
        # Arrange
        ai_generator.client.messages.create.side_effect = anthropic.RateLimitError(
            "Rate limit exceeded",
            response=Mock(status_code=429),
            body={"error": {"message": "Rate limit exceeded"}},
        )

        # Act & Assert
        with pytest.raises(AIGeneratorError, match="temporarily busy"):
            ai_generator.generate_response(query="test")

    def test_api_internal_error_wrapped(
        self,
        ai_generator: AIGenerator,
    ) -> None:
        """Test that APIError (500) is wrapped in AIGeneratorError."""
        # Arrange
        ai_generator.client.messages.create.side_effect = anthropic.APIError(
            "Internal server error",
            request=_create_mock_request(),
            body={"error": {"message": "Internal server error"}},
        )

        # Act & Assert
        with pytest.raises(AIGeneratorError, match="encountered an error"):
            ai_generator.generate_response(query="test")

    def test_connection_error_wrapped(
        self,
        ai_generator: AIGenerator,
    ) -> None:
        """Test that connection errors are wrapped in AIGeneratorError."""
        # Arrange
        ai_generator.client.messages.create.side_effect = anthropic.APIConnectionError(
            message="Connection failed",
            request=_create_mock_request(),
        )

        # Act & Assert
        with pytest.raises(AIGeneratorError, match="Could not connect"):
            ai_generator.generate_response(query="test")

    def test_empty_response_handled_gracefully(
        self,
        ai_generator: AIGenerator,
    ) -> None:
        """Test that empty response content returns a helpful message.

        Instead of crashing with IndexError, return a user-friendly message.
        """
        # Arrange: Response with empty content list
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = []  # Empty content list
        ai_generator.client.messages.create.return_value = mock_response

        # Act
        result = ai_generator.generate_response(query="test")

        # Assert: Returns helpful message instead of crashing
        assert "apologize" in result.lower() or "couldn't generate" in result.lower()

    def test_tool_execution_second_api_call_error_wrapped(
        self,
        ai_generator: AIGenerator,
    ) -> None:
        """Test error during follow-up API call after tool execution.

        The second API call (after tool execution) errors are wrapped.
        """
        # Arrange
        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "search"
        tool_use_block.id = "t1"
        tool_use_block.input = {"query": "test"}

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_block]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "results"

        # Second call fails
        ai_generator.client.messages.create.side_effect = [
            first_response,
            anthropic.APIError(
                "Error on second call",
                request=_create_mock_request(),
                body={},
            ),
        ]

        # Act & Assert: Error from second API call should be wrapped
        with pytest.raises(AIGeneratorError, match="encountered an error"):
            ai_generator.generate_response(
                query="test",
                tools=[{"name": "search"}],
                tool_manager=mock_tool_manager,
            )

    def test_tool_manager_exception_propagates(
        self,
        ai_generator: AIGenerator,
    ) -> None:
        """Test that ToolManager exceptions propagate through.

        If the tool manager raises an exception, it should propagate
        (these are not API errors, so they're not wrapped).
        """
        # Arrange
        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "search"
        tool_use_block.id = "t1"
        tool_use_block.input = {"query": "test"}

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use_block]

        ai_generator.client.messages.create.return_value = first_response

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = RuntimeError(
            "Tool execution failed"
        )

        # Act & Assert
        with pytest.raises(RuntimeError, match="Tool execution failed"):
            ai_generator.generate_response(
                query="test",
                tools=[{"name": "search"}],
                tool_manager=mock_tool_manager,
            )

    def test_ai_generator_error_contains_original_error(
        self,
        ai_generator: AIGenerator,
    ) -> None:
        """Test that AIGeneratorError preserves the original exception."""
        # Arrange
        original_error = anthropic.AuthenticationError(
            "Invalid API Key",
            response=Mock(status_code=401),
            body={},
        )
        ai_generator.client.messages.create.side_effect = original_error

        # Act & Assert
        with pytest.raises(AIGeneratorError) as exc_info:
            ai_generator.generate_response(query="test")

        assert exc_info.value.original_error is original_error


class TestSequentialToolCalling:
    """Tests for sequential tool calling functionality (up to 2 rounds)."""

    @pytest.fixture
    def mock_anthropic_client(self) -> Mock:
        """Create a mock Anthropic client."""
        client = Mock()
        client.messages = Mock()
        return client

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client: Mock) -> AIGenerator:
        """Create AIGenerator with mocked client."""
        with patch("ai_generator.anthropic.Anthropic") as mock_class:
            mock_class.return_value = mock_anthropic_client
            generator = AIGenerator(api_key="test-key", model="test-model")
            generator.client = mock_anthropic_client
            return generator

    @pytest.fixture
    def sample_tools(self) -> list[dict[str, Any]]:
        """Sample tool definitions."""
        return [
            {"name": "get_course_outline", "input_schema": {"type": "object"}},
            {"name": "search_course_content", "input_schema": {"type": "object"}},
        ]

    def test_two_sequential_tool_calls(
        self,
        ai_generator: AIGenerator,
        mock_anthropic_client: Mock,
        sample_tools: list[dict[str, Any]],
    ) -> None:
        """Test that Claude can make 2 tool calls sequentially.

        Scenario: User asks complex question requiring 2 tools.
        Round 1: Claude calls get_course_outline
        Round 2: Claude calls search_course_content with info from round 1
        Final: Claude synthesizes answer from both results
        """
        # Arrange: Create tool use blocks
        tool_use_1 = Mock()
        tool_use_1.type = "tool_use"
        tool_use_1.name = "get_course_outline"
        tool_use_1.id = "tool_1"
        tool_use_1.input = {"course_title": "MCP"}

        tool_use_2 = Mock()
        tool_use_2.type = "tool_use"
        tool_use_2.name = "search_course_content"
        tool_use_2.id = "tool_2"
        tool_use_2.input = {"query": "lesson 4 topic"}

        # Mock responses: tool_use -> tool_use -> final text
        response_1 = Mock()
        response_1.stop_reason = "tool_use"
        response_1.content = [tool_use_1]

        response_2 = Mock()
        response_2.stop_reason = "tool_use"
        response_2.content = [tool_use_2]

        response_3 = Mock()
        response_3.stop_reason = "end_turn"
        response_3.content = [Mock(type="text", text="Final answer using both tools")]

        mock_anthropic_client.messages.create.side_effect = [
            response_1,
            response_2,
            response_3,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Outline: Lesson 4 is about MCP Servers",
            "Search results about MCP Servers",
        ]

        # Act
        result = ai_generator.generate_response(
            query="What does lesson 4 of the MCP course cover?",
            tools=sample_tools,
            tool_manager=mock_tool_manager,
        )

        # Assert
        assert result == "Final answer using both tools"
        assert mock_anthropic_client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2

    def test_max_two_rounds_enforced(
        self,
        ai_generator: AIGenerator,
        mock_anthropic_client: Mock,
        sample_tools: list[dict[str, Any]],
    ) -> None:
        """Test that system enforces MAX_TOOL_ROUNDS=2 limit.

        Even if Claude keeps requesting tools, only 2 rounds are allowed.
        After 2 rounds, a final call WITHOUT tools forces a text response.
        """
        # Arrange: Claude keeps requesting tool_use
        tool_use = Mock()
        tool_use.type = "tool_use"
        tool_use.name = "search_course_content"
        tool_use.id = "tool_1"
        tool_use.input = {"query": "test"}

        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_response.content = [tool_use]

        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(type="text", text="Forced final answer")]

        # 2 tool rounds + 1 final call without tools = 3 total calls
        mock_anthropic_client.messages.create.side_effect = [
            tool_response,
            tool_response,
            final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        # Act
        result = ai_generator.generate_response(
            query="Keep searching",
            tools=sample_tools,
            tool_manager=mock_tool_manager,
        )

        # Assert
        assert result == "Forced final answer"
        assert mock_anthropic_client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2

    def test_tools_available_in_second_round(
        self,
        ai_generator: AIGenerator,
        mock_anthropic_client: Mock,
        sample_tools: list[dict[str, Any]],
    ) -> None:
        """Test that tools are still available in the second API call.

        This is the core fix - previously tools were removed after round 1.
        """
        # Arrange
        tool_use = Mock()
        tool_use.type = "tool_use"
        tool_use.name = "get_course_outline"
        tool_use.id = "tool_1"
        tool_use.input = {"course_title": "MCP"}

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use]

        second_response = Mock()
        second_response.stop_reason = "end_turn"
        second_response.content = [Mock(type="text", text="Answer")]

        mock_anthropic_client.messages.create.side_effect = [
            first_response,
            second_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Outline result"

        # Act
        ai_generator.generate_response(
            query="test",
            tools=sample_tools,
            tool_manager=mock_tool_manager,
        )

        # Assert: Verify tools were passed in BOTH API calls
        call_1 = mock_anthropic_client.messages.create.call_args_list[0][1]
        call_2 = mock_anthropic_client.messages.create.call_args_list[1][1]

        assert "tools" in call_1
        assert call_1["tools"] == sample_tools
        assert "tools" in call_2
        assert call_2["tools"] == sample_tools

    def test_message_history_accumulates_across_rounds(
        self,
        ai_generator: AIGenerator,
        mock_anthropic_client: Mock,
        sample_tools: list[dict[str, Any]],
    ) -> None:
        """Test that messages list properly accumulates through rounds.

        After 2 tool rounds, messages should be:
        [user query, assistant tool_use, user tool_result,
         assistant tool_use, user tool_result]
        """
        # Arrange
        tool_use_1 = Mock()
        tool_use_1.type = "tool_use"
        tool_use_1.name = "get_course_outline"
        tool_use_1.id = "tool_1"
        tool_use_1.input = {"course_title": "MCP"}

        tool_use_2 = Mock()
        tool_use_2.type = "tool_use"
        tool_use_2.name = "search_course_content"
        tool_use_2.id = "tool_2"
        tool_use_2.input = {"query": "topic"}

        response_1 = Mock()
        response_1.stop_reason = "tool_use"
        response_1.content = [tool_use_1]

        response_2 = Mock()
        response_2.stop_reason = "tool_use"
        response_2.content = [tool_use_2]

        response_3 = Mock()
        response_3.stop_reason = "end_turn"
        response_3.content = [Mock(type="text", text="Final answer")]

        mock_anthropic_client.messages.create.side_effect = [
            response_1,
            response_2,
            response_3,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        # Act
        ai_generator.generate_response(
            query="Complex query",
            tools=sample_tools,
            tool_manager=mock_tool_manager,
        )

        # Assert: Check final API call's messages
        final_call = mock_anthropic_client.messages.create.call_args_list[2][1]
        messages = final_call["messages"]

        # Should have 5 messages after 2 tool rounds
        assert len(messages) == 5
        assert messages[0]["role"] == "user"  # Original query
        assert messages[1]["role"] == "assistant"  # Round 1 tool use
        assert messages[2]["role"] == "user"  # Round 1 tool result
        assert messages[3]["role"] == "assistant"  # Round 2 tool use
        assert messages[4]["role"] == "user"  # Round 2 tool result

    def test_single_tool_call_still_works(
        self,
        ai_generator: AIGenerator,
        mock_anthropic_client: Mock,
        sample_tools: list[dict[str, Any]],
    ) -> None:
        """Test backward compatibility: single tool call still works.

        When Claude only needs one tool call and then answers, it should work.
        """
        # Arrange
        tool_use = Mock()
        tool_use.type = "tool_use"
        tool_use.name = "search_course_content"
        tool_use.id = "tool_1"
        tool_use.input = {"query": "test"}

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_use]

        second_response = Mock()
        second_response.stop_reason = "end_turn"
        second_response.content = [Mock(type="text", text="Answer from search")]

        mock_anthropic_client.messages.create.side_effect = [
            first_response,
            second_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results"

        # Act
        result = ai_generator.generate_response(
            query="Simple question",
            tools=sample_tools,
            tool_manager=mock_tool_manager,
        )

        # Assert
        assert result == "Answer from search"
        assert mock_anthropic_client.messages.create.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 1

    def test_no_tool_use_returns_immediately(
        self,
        ai_generator: AIGenerator,
        mock_anthropic_client: Mock,
        sample_tools: list[dict[str, Any]],
    ) -> None:
        """Test that direct answer (no tool_use) returns without looping."""
        # Arrange
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(type="text", text="Direct answer")]
        mock_anthropic_client.messages.create.return_value = mock_response

        mock_tool_manager = Mock()

        # Act
        result = ai_generator.generate_response(
            query="General knowledge question",
            tools=sample_tools,
            tool_manager=mock_tool_manager,
        )

        # Assert
        assert result == "Direct answer"
        assert mock_anthropic_client.messages.create.call_count == 1
        mock_tool_manager.execute_tool.assert_not_called()

    def test_max_tool_rounds_constant_is_two(self) -> None:
        """Verify MAX_TOOL_ROUNDS class constant is set to 2."""
        assert AIGenerator.MAX_TOOL_ROUNDS == 2

    def test_final_call_omits_tools_after_max_rounds(
        self,
        ai_generator: AIGenerator,
        mock_anthropic_client: Mock,
        sample_tools: list[dict[str, Any]],
    ) -> None:
        """Test that the final API call after max rounds does not include tools.

        This forces Claude to provide a text response instead of
        requesting more tools.
        """
        # Arrange: Claude keeps requesting tools
        tool_use = Mock()
        tool_use.type = "tool_use"
        tool_use.name = "search_course_content"
        tool_use.id = "tool_1"
        tool_use.input = {"query": "test"}

        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_response.content = [tool_use]

        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(type="text", text="Final")]

        mock_anthropic_client.messages.create.side_effect = [
            tool_response,
            tool_response,
            final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Result"

        # Act
        ai_generator.generate_response(
            query="test",
            tools=sample_tools,
            tool_manager=mock_tool_manager,
        )

        # Assert: Final call (3rd) should NOT include tools
        final_call = mock_anthropic_client.messages.create.call_args_list[2][1]
        assert "tools" not in final_call
        assert "tool_choice" not in final_call
