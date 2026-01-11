from typing import List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Maximum number of sequential tool calling rounds
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Tool Usage:
- **get_course_outline tool**: Use for questions about course structure, lesson lists, course overview, or when users ask "what lessons does X have" or "what's in the course"
  - Returns: course title, course link, and complete list of lessons with numbers and titles
  - **Return the outline exactly as provided by the tool without adding extra text or descriptions**
- **search_course_content tool**: Use for questions about specific course content or detailed educational materials
  - Returns: relevant content chunks from course lessons
  - Synthesize search results into accurate, fact-based responses
- **Multiple sequential tool calls allowed**: You may call tools up to 2 times to gather complete information
  - Example: First get course outline, then search specific lesson content based on the outline
  - Each tool call result will be returned to you, and you can analyze and call tools again if needed
- If tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
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
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional multi-round tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize message history
        messages = [{"role": "user", "content": query}]

        # Iterative loop for sequential tool calling
        for round_num in range(self.MAX_TOOL_ROUNDS):
            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
            }

            # Add tools if available
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Get response from Claude
            response = self.client.messages.create(**api_params)

            # Check if tool use is requested
            if response.stop_reason == "tool_use" and tool_manager:
                # Execute tools and append results to messages
                error_result = self._execute_and_append_tools(
                    response, messages, tool_manager
                )

                # Check for tool execution errors
                if error_result:
                    return error_result

                # Continue to next round with updated messages
                continue

            # Natural termination: no tool use, return response
            return (
                response.content[0].text
                if response.content
                else "No response generated"
            )

        # Max rounds reached - make final call without tools to force answer
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        final_response = self.client.messages.create(**final_params)
        return (
            final_response.content[0].text
            if final_response.content
            else "No response generated"
        )

    def _execute_and_append_tools(
        self, response, messages: List, tool_manager
    ) -> Optional[str]:
        """
        Execute tool calls and append results to message history in-place.

        This method modifies the messages list by appending:
        1. Assistant's response with tool_use content blocks
        2. User message with tool_result content blocks

        Args:
            response: The API response containing tool use requests
            messages: Message history list (modified in-place)
            tool_manager: Manager to execute tools

        Returns:
            Error message string if tool execution failed, None if successful
        """

        # Add AI's tool use response to messages
        messages.append({"role": "assistant", "content": response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )
                except Exception as e:
                    # Tool execution error - return error message
                    error_msg = f"Tool execution error: {content_block.name} failed with {str(e)}"

                    # Add error result to messages for context
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": error_msg,
                            "is_error": True,
                        }
                    )
                    messages.append({"role": "user", "content": tool_results})

                    # Return error to terminate loop
                    return error_msg

        # Add successful tool results to messages
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        return None  # Success - no error
