"""
Tests for CourseSearchTool.execute() method

These tests verify:
1. Basic search functionality
2. Search with course name filtering
3. Search with lesson number filtering
4. Error handling
5. Empty results handling
6. Source tracking
"""
import pytest
from unittest.mock import Mock
from vector_store import SearchResults
from search_tools import CourseSearchTool, ToolManager


class TestCourseSearchToolExecute:
    """Test suite for CourseSearchTool.execute() method"""

    def test_execute_basic_search_success(self, course_search_tool, mock_vector_store):
        """Test basic search returns formatted results"""
        # Setup
        query = "What is Python?"

        # Execute
        result = course_search_tool.execute(query=query)

        # Assert
        assert isinstance(result, str)
        assert len(result) > 0
        mock_vector_store.search.assert_called_once_with(
            query=query,
            course_name=None,
            lesson_number=None
        )
        # Check that result contains the course title in header format
        assert "[Introduction to Python" in result

    def test_execute_with_course_name_filter(self, course_search_tool, mock_vector_store):
        """Test search with course name filtering"""
        # Setup
        query = "variables"
        course_name = "Python"

        # Execute
        result = course_search_tool.execute(query=query, course_name=course_name)

        # Assert
        mock_vector_store.search.assert_called_once_with(
            query=query,
            course_name=course_name,
            lesson_number=None
        )
        assert isinstance(result, str)

    def test_execute_with_lesson_filter(self, course_search_tool, mock_vector_store):
        """Test search with lesson number filtering"""
        # Setup
        query = "control flow"
        lesson_number = 3

        # Execute
        result = course_search_tool.execute(query=query, lesson_number=lesson_number)

        # Assert
        mock_vector_store.search.assert_called_once_with(
            query=query,
            course_name=None,
            lesson_number=lesson_number
        )
        assert isinstance(result, str)

    def test_execute_with_both_filters(self, course_search_tool, mock_vector_store):
        """Test search with both course name and lesson number filters"""
        # Setup
        query = "variables"
        course_name = "Python"
        lesson_number = 2

        # Execute
        result = course_search_tool.execute(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )

        # Assert
        mock_vector_store.search.assert_called_once_with(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )

    def test_execute_handles_empty_results(self, course_search_tool, mock_vector_store):
        """Test that empty results return appropriate message"""
        # Setup
        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )

        # Execute
        result = course_search_tool.execute(query="nonexistent topic")

        # Assert
        assert "No relevant content found" in result

    def test_execute_handles_empty_results_with_course_filter(self, course_search_tool, mock_vector_store):
        """Test empty results message includes filter information"""
        # Setup
        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )

        # Execute
        result = course_search_tool.execute(query="test", course_name="Python")

        # Assert
        assert "No relevant content found" in result
        assert "Python" in result

    def test_execute_handles_search_errors(self, course_search_tool, mock_vector_store):
        """Test that search errors are properly returned"""
        # Setup
        error_message = "Search error: Database connection failed"
        mock_vector_store.search.return_value = SearchResults.empty(error_message)

        # Execute
        result = course_search_tool.execute(query="test")

        # Assert
        assert result == error_message

    def test_execute_tracks_sources(self, course_search_tool, mock_vector_store):
        """Test that sources are tracked after search"""
        # Setup - mock with lesson link available
        mock_vector_store.get_lesson_link.return_value = "https://example.com/python/lesson1"

        # Execute
        result = course_search_tool.execute(query="Python")

        # Assert
        assert len(course_search_tool.last_sources) > 0
        source = course_search_tool.last_sources[0]
        assert "text" in source
        assert "url" in source
        assert "Introduction to Python" in source["text"]

    def test_execute_formats_results_correctly(self, course_search_tool, mock_vector_store):
        """Test that results are formatted with proper headers and content"""
        # Setup
        mock_vector_store.search.return_value = SearchResults(
            documents=["This is lesson content about variables"],
            metadata=[{
                "course_title": "Introduction to Python",
                "lesson_number": 2,
                "chunk_index": 0
            }],
            distances=[0.3],
            error=None
        )

        # Execute
        result = course_search_tool.execute(query="variables")

        # Assert
        assert "[Introduction to Python - Lesson 2]" in result
        assert "This is lesson content about variables" in result

    def test_execute_multiple_results(self, course_search_tool, mock_vector_store):
        """Test handling of multiple search results"""
        # Setup
        mock_vector_store.search.return_value = SearchResults(
            documents=[
                "First result about Python",
                "Second result about Python"
            ],
            metadata=[
                {"course_title": "Introduction to Python", "lesson_number": 1, "chunk_index": 0},
                {"course_title": "Introduction to Python", "lesson_number": 2, "chunk_index": 1}
            ],
            distances=[0.2, 0.4],
            error=None
        )

        # Execute
        result = course_search_tool.execute(query="Python")

        # Assert
        assert "First result about Python" in result
        assert "Second result about Python" in result
        assert len(course_search_tool.last_sources) == 2


class TestToolManager:
    """Test suite for ToolManager functionality"""

    def test_tool_manager_registers_tool(self, course_search_tool):
        """Test that tools can be registered"""
        manager = ToolManager()
        manager.register_tool(course_search_tool)

        assert "search_course_content" in manager.tools

    def test_tool_manager_gets_definitions(self, tool_manager):
        """Test that tool definitions are retrieved correctly"""
        definitions = tool_manager.get_tool_definitions()

        assert len(definitions) > 0
        assert definitions[0]["name"] == "search_course_content"
        assert "input_schema" in definitions[0]

    def test_tool_manager_executes_tool(self, tool_manager, mock_vector_store):
        """Test that ToolManager can execute registered tools"""
        result = tool_manager.execute_tool(
            "search_course_content",
            query="Python"
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_tool_manager_handles_unknown_tool(self, tool_manager):
        """Test that ToolManager handles unknown tool names"""
        result = tool_manager.execute_tool("nonexistent_tool", query="test")

        assert "not found" in result

    def test_tool_manager_gets_last_sources(self, tool_manager, mock_vector_store):
        """Test that ToolManager retrieves sources from tools"""
        # Execute a search
        tool_manager.execute_tool("search_course_content", query="Python")

        # Get sources
        sources = tool_manager.get_last_sources()

        assert isinstance(sources, list)
        # Should have sources from the search
        assert len(sources) > 0

    def test_tool_manager_resets_sources(self, tool_manager, mock_vector_store):
        """Test that ToolManager can reset sources"""
        # Execute a search
        tool_manager.execute_tool("search_course_content", query="Python")
        assert len(tool_manager.get_last_sources()) > 0

        # Reset sources
        tool_manager.reset_sources()

        # Verify sources are cleared
        sources = tool_manager.get_last_sources()
        assert len(sources) == 0
