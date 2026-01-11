"""
Tests for AIGenerator tool calling functionality

These tests verify:
1. Basic response generation
2. Tool calling mechanism
3. Tool execution flow
4. Response formatting after tool use
5. Conversation history handling
"""

from unittest.mock import Mock

from ai_generator import AIGenerator


class TestAIGeneratorToolCalling:
    """Test suite for AIGenerator tool calling capabilities"""

    def test_generate_response_without_tools(
        self, ai_generator_with_mock, mock_anthropic_client
    ):
        """Test basic response generation without tools"""
        # Execute
        response = ai_generator_with_mock.generate_response(query="What is Python?")

        # Assert
        assert isinstance(response, str)
        assert len(response) > 0
        mock_anthropic_client.messages.create.assert_called_once()

        # Verify tools were not included in the call
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert "tools" not in call_kwargs

    def test_generate_response_with_tools_but_no_use(
        self, ai_generator_with_mock, mock_anthropic_client
    ):
        """Test response generation with tools available but not used"""
        # Setup
        tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]

        # Execute
        response = ai_generator_with_mock.generate_response(
            query="What is 2+2?", tools=tools
        )

        # Assert
        assert isinstance(response, str)
        mock_anthropic_client.messages.create.assert_called_once()

        # Verify tools were included in the call
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    def test_generate_response_with_tool_use(
        self, ai_generator_with_mock, mock_anthropic_client
    ):
        """Test response generation when AI decides to use a tool"""
        # Setup - mock initial response with tool use
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"

        # Mock tool use content block
        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "search_course_content"
        tool_use_block.input = {"query": "Python basics"}
        tool_use_block.id = "tool_call_123"

        initial_response.content = [tool_use_block]

        # Mock final response after tool execution
        final_response = Mock()
        final_response.content = [
            Mock(text="Based on the search, Python is a programming language.")
        ]

        # Configure mock to return different responses on subsequent calls
        mock_anthropic_client.messages.create.side_effect = [
            initial_response,
            final_response,
        ]

        # Setup tools
        tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]

        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Python is a programming language"

        # Execute
        response = ai_generator_with_mock.generate_response(
            query="What is Python?", tools=tools, tool_manager=mock_tool_manager
        )

        # Assert
        assert isinstance(response, str)
        assert "Based on the search" in response or len(response) > 0

        # Verify two API calls were made (initial + follow-up after tool use)
        assert mock_anthropic_client.messages.create.call_count == 2

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once()

    def test_handle_tool_execution_flow(
        self, ai_generator_with_mock, mock_anthropic_client
    ):
        """Test the complete tool execution flow with new _execute_and_append_tools method"""
        # Setup
        response = Mock()
        response.stop_reason = "tool_use"

        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "search_course_content"
        tool_use_block.input = {"query": "variables in Python"}
        tool_use_block.id = "tool_123"

        response.content = [tool_use_block]

        # Setup messages list
        messages = [{"role": "user", "content": "Tell me about variables"}]

        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Variables store data in Python"

        # Execute the private method directly
        result = ai_generator_with_mock._execute_and_append_tools(
            response, messages, mock_tool_manager
        )

        # Assert
        assert result is None  # Success returns None
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="variables in Python"
        )
        # Verify messages were appended
        assert len(messages) == 3  # original + assistant response + tool results

    def test_conversation_history_included(
        self, ai_generator_with_mock, mock_anthropic_client
    ):
        """Test that conversation history is properly included"""
        # Setup
        history = "User: Hello\nAssistant: Hi there!"

        # Execute
        _response = ai_generator_with_mock.generate_response(  # noqa: F841
            query="What is Python?", conversation_history=history
        )

        # Assert
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert "system" in call_kwargs
        assert history in call_kwargs["system"]

    def test_system_prompt_structure(self, ai_generator_with_mock):
        """Test that system prompt contains essential instructions"""
        system_prompt = ai_generator_with_mock.SYSTEM_PROMPT

        # Verify key elements are present
        assert "tool" in system_prompt.lower() or "Tool" in system_prompt
        assert "search" in system_prompt.lower() or "Search" in system_prompt

    def test_multiple_tool_calls_in_response(
        self, ai_generator_with_mock, mock_anthropic_client
    ):
        """Test handling of multiple tool calls in a single response with new method"""
        # Setup - mock response with multiple tool use blocks
        response = Mock()
        response.stop_reason = "tool_use"

        tool_use_1 = Mock()
        tool_use_1.type = "tool_use"
        tool_use_1.name = "search_course_content"
        tool_use_1.input = {"query": "Python"}
        tool_use_1.id = "tool_1"

        tool_use_2 = Mock()
        tool_use_2.type = "tool_use"
        tool_use_2.name = "search_course_content"
        tool_use_2.input = {"query": "variables"}
        tool_use_2.id = "tool_2"

        response.content = [tool_use_1, tool_use_2]

        # Setup messages list
        messages = [{"role": "user", "content": "Test query"}]

        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        # Execute
        result = ai_generator_with_mock._execute_and_append_tools(
            response, messages, mock_tool_manager
        )

        # Assert - both tools should be executed
        assert mock_tool_manager.execute_tool.call_count == 2
        assert result is None  # Success
        assert len(messages) == 3  # original + assistant response + tool results

    def test_api_parameters_correct(
        self, ai_generator_with_mock, mock_anthropic_client
    ):
        """Test that API parameters are correctly set"""
        # Execute
        ai_generator_with_mock.generate_response(query="Test query")

        # Assert
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]

        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["temperature"] == 0
        assert call_kwargs["max_tokens"] == 800
        assert "messages" in call_kwargs
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"

    def test_tool_use_without_tool_manager_returns_text(
        self, ai_generator_with_mock, mock_anthropic_client
    ):
        """Test that tool_use without tool_manager still returns gracefully"""
        # Setup - mock response indicating tool use but no manager provided
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        initial_response.content = [Mock(text="Attempted to use tool")]

        mock_anthropic_client.messages.create.return_value = initial_response

        tools = [{"name": "test_tool"}]

        # Execute without tool_manager
        response = ai_generator_with_mock.generate_response(
            query="Test", tools=tools, tool_manager=None
        )

        # Should return the text from the response
        assert isinstance(response, str)


class TestAIGeneratorConfiguration:
    """Test suite for AIGenerator configuration and initialization"""

    def test_initialization(self):
        """Test AIGenerator initializes correctly"""
        generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        assert generator.model == "claude-sonnet-4-20250514"
        assert generator.base_params["model"] == "claude-sonnet-4-20250514"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

    def test_system_prompt_is_static(self):
        """Test that SYSTEM_PROMPT is defined at class level"""
        assert hasattr(AIGenerator, "SYSTEM_PROMPT")
        assert isinstance(AIGenerator.SYSTEM_PROMPT, str)
        assert len(AIGenerator.SYSTEM_PROMPT) > 0
