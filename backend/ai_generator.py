import logging
from typing import List, Optional, Dict, Any

import anthropic

logger = logging.getLogger(__name__)


class AIGeneratorError(Exception):
    """Custom exception for AI generation errors with user-friendly messages."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Maximum number of sequential tool rounds per query
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials with access to tools for course information.

Available Tools:
1. **search_course_content**: Search course materials for specific content
   - Use for: questions about topics, explanations, concepts within courses

2. **get_course_outline**: Get course structure with lesson list
   - Use for: questions about course structure, what lessons exist, lesson titles, course navigation

Tool Selection:
- **Outline questions** (use get_course_outline): "What lessons are in X?", "Show me the outline", "What topics does X cover?", "How many lessons?"
- **Content questions** (use search_course_content): "Explain X", "How do I do X?", "What does the course say about X?"
- **General knowledge**: Answer directly without tools

Sequential Tool Usage:
- **Up to 2 tool calls per query** when needed for complex queries requiring:
  * Comparing information across courses
  * First getting structure, then searching within specific lessons
  * Multi-part questions needing different data sources
- Use results from the first tool call to inform the second

Response Protocol:
- Synthesize tool results into accurate, fact-based responses
- If tool yields no results, state this clearly without offering alternatives
- **No meta-commentary**: Provide direct answers only - no reasoning or tool explanations

All responses must be:
1. **Brief and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include examples when helpful
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_manager: Optional[Any] = None,
    ) -> str:
        """
        Generate AI response with optional multi-round tool usage.

        Supports up to MAX_TOOL_ROUNDS sequential tool calls where Claude
        can reason about results from one tool before calling another.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string

        Raises:
            AIGeneratorError: On API errors (auth, rate limit, connection, etc.)
        """
        # Build system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize message list with user query
        messages: List[Dict[str, Any]] = [{"role": "user", "content": query}]

        # Track tool rounds
        tool_rounds_completed = 0

        # Main loop - allows up to MAX_TOOL_ROUNDS sequential tool calls
        while tool_rounds_completed < self.MAX_TOOL_ROUNDS:
            # Prepare API parameters
            api_params: Dict[str, Any] = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
            }

            # Add tools if available
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Make API call
            response = self._make_api_call(api_params)

            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use" and tool_manager:
                # Add Claude's tool request to messages
                messages.append({"role": "assistant", "content": response.content})

                # Execute all tool calls and collect results
                tool_results = self._execute_tools(response.content, tool_manager)

                # Add tool results to messages
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})

                # Increment round counter and continue loop
                tool_rounds_completed += 1
                continue

            # Claude chose to answer directly (no tool_use) - return response
            return self._extract_text_response(response)

        # MAX_TOOL_ROUNDS reached - make final call WITHOUT tools to force text response
        final_params: Dict[str, Any] = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        final_response = self._make_api_call(final_params)
        return self._extract_text_response(final_response)

    def _make_api_call(self, api_params: Dict[str, Any]) -> Any:
        """
        Make an API call to Claude with error handling.

        Args:
            api_params: Parameters for the API call

        Returns:
            API response object

        Raises:
            AIGeneratorError: On API errors
        """
        try:
            return self.client.messages.create(**api_params)
        except anthropic.AuthenticationError as e:
            logger.error("Anthropic API authentication failed: %s", e)
            raise AIGeneratorError(
                "API authentication failed. Please check the API key configuration.",
                original_error=e,
            ) from e
        except anthropic.RateLimitError as e:
            logger.warning("Anthropic API rate limit exceeded: %s", e)
            raise AIGeneratorError(
                "The AI service is temporarily busy. Please try again in a moment.",
                original_error=e,
            ) from e
        except anthropic.APIConnectionError as e:
            logger.error("Failed to connect to Anthropic API: %s", e)
            raise AIGeneratorError(
                "Could not connect to the AI service. Please check your internet connection.",
                original_error=e,
            ) from e
        except anthropic.APIError as e:
            logger.error("Anthropic API error: %s", e)
            raise AIGeneratorError(
                "The AI service encountered an error. Please try again.",
                original_error=e,
            ) from e

    def _execute_tools(
        self, content_blocks: List[Any], tool_manager: Any
    ) -> List[Dict[str, Any]]:
        """
        Execute all tool calls from response content blocks.

        Args:
            content_blocks: Content blocks from API response
            tool_manager: Manager to execute tools

        Returns:
            List of tool result dictionaries
        """
        tool_results: List[Dict[str, Any]] = []
        for content_block in content_blocks:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name,
                    **content_block.input,
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result,
                    }
                )
        return tool_results

    def _extract_text_response(self, response: Any) -> str:
        """
        Extract text from API response, handling empty content gracefully.

        Args:
            response: API response object

        Returns:
            Text content from response
        """
        if not response.content:
            logger.warning("Received empty response content from API")
            return (
                "I apologize, but I couldn't generate a response. "
                "Please try rephrasing your question."
            )
        return response.content[0].text
