"""
Tests for sequential/iterative tool calling in AIGenerator

These tests verify:
1. Single-round tool use (backward compatibility)
2. Two-round sequential tool use
3. Max rounds enforcement
4. Natural termination
5. Error handling
6. Message history accumulation
7. Tool availability across rounds
"""
import pytest
from unittest.mock import Mock
from ai_generator import AIGenerator


class TestSequentialToolCalling:
    """Test suite for sequential/iterative tool calling"""

    def test_single_round_tool_use(self, ai_generator_with_mock, mock_anthropic_client):
        """Verify single tool call still works (backward compatibility)"""
        # Setup: Mock tool_use → text response
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_use = Mock()
        tool_use.type = "tool_use"
        tool_use.name = "search_course_content"
        tool_use.input = {"query": "Python"}
        tool_use.id = "tool_1"
        tool_response.content = [tool_use]

        # Final response after tool execution
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="Python is a programming language")]

        # Configure mock to return responses in sequence
        mock_anthropic_client.messages.create.side_effect = [
            tool_response,      # Round 1: tool use
            final_response      # Round 1: response after tool
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Python is a high-level language"

        tools = [{"name": "search_course_content"}]

        # Execute
        response = ai_generator_with_mock.generate_response(
            query="What is Python?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert
        assert isinstance(response, str)
        assert "Python" in response or "programming" in response.lower()
        assert mock_anthropic_client.messages.create.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 1

    def test_two_round_sequential_tool_use(self, ai_generator_with_mock, mock_anthropic_client):
        """Verify Claude can call tools twice sequentially"""
        # Round 1: Tool use for course outline
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        tool_use_1 = Mock()
        tool_use_1.type = "tool_use"
        tool_use_1.name = "get_course_outline"
        tool_use_1.input = {"course_title": "Python"}
        tool_use_1.id = "tool_1"
        round1_response.content = [tool_use_1]

        # Round 2: Tool use for content search
        round2_response = Mock()
        round2_response.stop_reason = "tool_use"
        tool_use_2 = Mock()
        tool_use_2.type = "tool_use"
        tool_use_2.name = "search_course_content"
        tool_use_2.input = {"query": "variables", "lesson_number": 2}
        tool_use_2.id = "tool_2"
        round2_response.content = [tool_use_2]

        # Final response after tools
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="The Python course has 5 lessons. Lesson 2 covers variables and data types.")]

        # Configure mock to return responses in sequence
        mock_anthropic_client.messages.create.side_effect = [
            round1_response,  # Round 1: get outline
            round2_response,  # Round 2: search content
            final_response    # Final: answer
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline: Lesson 1, Lesson 2, Lesson 3, Lesson 4, Lesson 5",
            "Lesson 2 content: Variables are containers for storing data values..."
        ]

        tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]

        # Execute
        response = ai_generator_with_mock.generate_response(
            query="What lessons are in Python course and tell me about variables?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert
        assert isinstance(response, str)
        assert "lesson" in response.lower() or "variable" in response.lower()
        assert mock_anthropic_client.messages.create.call_count == 3  # 2 tool rounds + 1 final
        assert mock_tool_manager.execute_tool.call_count == 2

    def test_max_rounds_enforced(self, ai_generator_with_mock, mock_anthropic_client):
        """Verify loop stops after 2 rounds"""
        # Both rounds return tool_use
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_use = Mock()
        tool_use.type = "tool_use"
        tool_use.name = "search_course_content"
        tool_use.input = {"query": "test"}
        tool_use.id = "tool_id"
        tool_response.content = [tool_use]

        # Final response (without tools)
        final_response = Mock()
        final_response.content = [Mock(text="Based on the searches, here's the answer")]

        mock_anthropic_client.messages.create.side_effect = [
            tool_response,  # Round 1
            tool_response,  # Round 2
            final_response  # Final call without tools
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search result"

        response = ai_generator_with_mock.generate_response(
            query="Test",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Should call API 3 times: 2 tool rounds + 1 final without tools
        assert mock_anthropic_client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2
        assert isinstance(response, str)

    def test_natural_termination_after_first_tool(self, ai_generator_with_mock, mock_anthropic_client):
        """Verify loop exits when Claude doesn't use tools in second response"""
        # Round 1: Tool use
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_use = Mock()
        tool_use.type = "tool_use"
        tool_use.name = "search_course_content"
        tool_use.input = {"query": "Python"}
        tool_use.id = "tool_1"
        tool_response.content = [tool_use]

        # Round 2: Text response (no tool use)
        text_response = Mock()
        text_response.stop_reason = "end_turn"
        text_response.content = [Mock(text="Python is a programming language")]

        mock_anthropic_client.messages.create.side_effect = [
            tool_response,   # Round 1
            text_response    # Round 2 - natural termination
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Python info..."

        response = ai_generator_with_mock.generate_response(
            query="What is Python?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Should only call API twice (tool + response)
        assert mock_anthropic_client.messages.create.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 1
        assert "Python" in response

    def test_early_termination_no_tool_use(self, ai_generator_with_mock, mock_anthropic_client):
        """Verify early termination when AI doesn't use tools"""
        # First call: direct answer, no tools
        direct_response = Mock()
        direct_response.stop_reason = "end_turn"
        direct_response.content = [Mock(text="2+2=4")]

        mock_anthropic_client.messages.create.return_value = direct_response

        response = ai_generator_with_mock.generate_response(
            query="What is 2+2?",
            tools=[{"name": "search_course_content"}],
            tool_manager=Mock()
        )

        # Should only call once - no tools used
        assert mock_anthropic_client.messages.create.call_count == 1
        assert "4" in response

    def test_tool_error_terminates_sequence(self, ai_generator_with_mock, mock_anthropic_client):
        """Verify tool errors prevent further rounds"""
        # Round 1: Tool use
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_use = Mock()
        tool_use.type = "tool_use"
        tool_use.name = "search_course_content"
        tool_use.input = {"query": "test"}
        tool_use.id = "tool_id"
        tool_response.content = [tool_use]

        mock_anthropic_client.messages.create.return_value = tool_response

        # Tool manager raises exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Database connection failed")

        response = ai_generator_with_mock.generate_response(
            query="Test",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Should return error message
        assert isinstance(response, str)
        assert "error" in response.lower() or "failed" in response.lower()
        # Should not make additional API calls after error
        assert mock_anthropic_client.messages.create.call_count == 1

    def test_message_history_accumulates_correctly(self, ai_generator_with_mock, mock_anthropic_client):
        """Verify message array grows correctly across rounds"""
        # Track all API calls with deep copy of messages to verify message history
        api_calls = []

        def capture_call(**kwargs):
            import copy
            # Deep copy messages to capture state at call time
            call_record = kwargs.copy()
            call_record["messages"] = copy.deepcopy(kwargs["messages"])
            api_calls.append(call_record)

            if len(api_calls) == 1:
                # Round 1 response
                response = Mock()
                response.stop_reason = "tool_use"
                tool = Mock()
                tool.type = "tool_use"
                tool.name = "search"
                tool.input = {"query": "test"}
                tool.id = "id1"
                response.content = [tool]
                return response
            elif len(api_calls) == 2:
                # Round 2 response (still using tools)
                response = Mock()
                response.stop_reason = "tool_use"
                tool = Mock()
                tool.type = "tool_use"
                tool.name = "search"
                tool.input = {"query": "test2"}
                tool.id = "id2"
                response.content = [tool]
                return response
            else:
                # Final response
                response = Mock()
                response.stop_reason = "end_turn"
                response.content = [Mock(text="Final answer")]
                return response

        mock_anthropic_client.messages.create.side_effect = capture_call

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Result"

        ai_generator_with_mock.generate_response(
            query="Test",
            tools=[{"name": "search"}],
            tool_manager=mock_tool_manager
        )

        # Verify message history growth
        assert len(api_calls) == 3

        # First call: 1 message (user query)
        assert len(api_calls[0]["messages"]) == 1
        assert api_calls[0]["messages"][0]["role"] == "user"

        # Second call: 3 messages (user query + assistant tool_use + user tool_result)
        assert len(api_calls[1]["messages"]) == 3
        assert api_calls[1]["messages"][0]["role"] == "user"
        assert api_calls[1]["messages"][1]["role"] == "assistant"
        assert api_calls[1]["messages"][2]["role"] == "user"

        # Third call (final, no tools): 5 messages
        # (user + assistant + user + assistant + user)
        assert len(api_calls[2]["messages"]) == 5

    def test_tools_available_in_each_round(self, ai_generator_with_mock, mock_anthropic_client):
        """Verify tools param present in all round API calls"""
        # Track API calls
        api_calls = []

        def capture_call(**kwargs):
            api_calls.append(kwargs)
            if len(api_calls) <= 2:
                # Tool use responses
                response = Mock()
                response.stop_reason = "tool_use"
                tool = Mock()
                tool.type = "tool_use"
                tool.name = "search"
                tool.input = {"query": "test"}
                tool.id = f"id{len(api_calls)}"
                response.content = [tool]
                return response
            else:
                # Final response
                response = Mock()
                response.stop_reason = "end_turn"
                response.content = [Mock(text="Answer")]
                return response

        mock_anthropic_client.messages.create.side_effect = capture_call

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Result"

        tools = [{"name": "search"}]

        ai_generator_with_mock.generate_response(
            query="Test",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Both round 1 and round 2 should have tools
        assert "tools" in api_calls[0]
        assert api_calls[0]["tools"] == tools
        assert "tools" in api_calls[1]
        assert api_calls[1]["tools"] == tools

        # Final call (after max rounds) should NOT have tools
        assert "tools" not in api_calls[2]

    def test_no_tools_provided_returns_direct_response(self, ai_generator_with_mock, mock_anthropic_client):
        """Verify behavior when no tools given"""
        # Setup: direct response
        direct_response = Mock()
        direct_response.stop_reason = "end_turn"
        direct_response.content = [Mock(text="Direct answer without tools")]

        mock_anthropic_client.messages.create.return_value = direct_response

        # Call without tools
        response = ai_generator_with_mock.generate_response(
            query="What is 2+2?",
            tools=None,
            tool_manager=None
        )

        # Should get direct response
        assert isinstance(response, str)
        assert "Direct answer" in response
        assert mock_anthropic_client.messages.create.call_count == 1

        # Verify tools were not in the API call
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert "tools" not in call_kwargs
