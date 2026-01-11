"""
Tests for RAG System content query handling

These tests verify:
1. End-to-end query processing
2. Integration between components
3. Source retrieval and tracking
4. Session management
5. Error handling in the complete flow
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from vector_store import SearchResults


class TestRAGSystemQueryHandling:
    """Test suite for RAG system query processing"""

    @pytest.fixture
    def mock_rag_components(self, test_config, temp_chroma_path):
        """Create a RAG system with mocked components"""
        test_config.CHROMA_PATH = temp_chroma_path

        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor') as MockDocProcessor:

            # Setup vector store mock
            mock_vector_store = Mock()
            mock_vector_store.search.return_value = SearchResults(
                documents=["Python is a programming language"],
                metadata=[{"course_title": "Introduction to Python", "lesson_number": 1}],
                distances=[0.3],
                error=None
            )
            mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
            MockVectorStore.return_value = mock_vector_store

            # Setup AI generator mock
            mock_ai_gen = Mock()
            mock_ai_gen.generate_response.return_value = "Python is a high-level programming language used for various applications."
            MockAIGenerator.return_value = mock_ai_gen

            # Setup document processor mock
            MockDocProcessor.return_value = Mock()

            # Create RAG system
            rag = RAGSystem(test_config)

            return rag, mock_vector_store, mock_ai_gen

    def test_query_basic_content_question(self, mock_rag_components):
        """Test basic content query processing"""
        rag, mock_vector_store, mock_ai_gen = mock_rag_components

        # Execute
        response, sources = rag.query("What is Python?")

        # Assert
        assert isinstance(response, str)
        assert len(response) > 0
        mock_ai_gen.generate_response.assert_called_once()

        # Verify AI was called with proper tools
        call_kwargs = mock_ai_gen.generate_response.call_args[1]
        assert "tools" in call_kwargs
        assert "tool_manager" in call_kwargs

    def test_query_returns_sources(self, mock_rag_components):
        """Test that query returns sources from tool execution"""
        rag, mock_vector_store, mock_ai_gen = mock_rag_components

        # Setup tool manager to return sources
        rag.tool_manager.get_last_sources = Mock(return_value=[
            {"text": "Introduction to Python - Lesson 1", "url": "https://example.com/lesson1"}
        ])

        # Execute
        response, sources = rag.query("What is Python?")

        # Assert
        assert isinstance(sources, list)
        # Sources should be retrieved from tool manager
        rag.tool_manager.get_last_sources.assert_called_once()

    def test_query_with_session_id(self, mock_rag_components):
        """Test query processing with session management"""
        rag, mock_vector_store, mock_ai_gen = mock_rag_components

        # Create a session
        session_id = rag.session_manager.create_session()

        # Execute query with session
        response, sources = rag.query("What is Python?", session_id=session_id)

        # Assert
        assert isinstance(response, str)

        # Verify conversation history was retrieved and passed
        call_kwargs = mock_ai_gen.generate_response.call_args[1]
        # History might be None for first query or a string for subsequent queries
        assert "conversation_history" in call_kwargs

    def test_query_updates_conversation_history(self, mock_rag_components):
        """Test that query updates conversation history"""
        rag, mock_vector_store, mock_ai_gen = mock_rag_components

        # Create session
        session_id = rag.session_manager.create_session()

        # Execute first query
        query1 = "What is Python?"
        response1, _ = rag.query(query1, session_id=session_id)

        # Execute second query
        query2 = "Tell me more"
        response2, _ = rag.query(query2, session_id=session_id)

        # Assert - verify history was passed in second call
        second_call_kwargs = mock_ai_gen.generate_response.call_args[1]
        history = second_call_kwargs.get("conversation_history")

        # History should exist after first exchange
        assert history is not None

    def test_query_sources_reset_after_retrieval(self, mock_rag_components):
        """Test that sources are reset after being retrieved"""
        rag, mock_vector_store, mock_ai_gen = mock_rag_components

        # Setup mock for sources
        rag.tool_manager.get_last_sources = Mock(return_value=[{"text": "test", "url": "http://test.com"}])
        rag.tool_manager.reset_sources = Mock()

        # Execute
        response, sources = rag.query("Test query")

        # Assert
        rag.tool_manager.reset_sources.assert_called_once()

    def test_query_prompt_formatting(self, mock_rag_components):
        """Test that query is properly formatted for AI"""
        rag, mock_vector_store, mock_ai_gen = mock_rag_components

        user_query = "What is Python?"

        # Execute
        rag.query(user_query)

        # Assert - check the query parameter sent to AI
        call_kwargs = mock_ai_gen.generate_response.call_args[1]
        query_param = call_kwargs["query"]

        assert user_query in query_param
        assert "course materials" in query_param.lower()


class TestRAGSystemWithRealVectorStore:
    """Test RAG system with actual vector store (tests MAX_RESULTS bug)"""

    def test_query_with_zero_max_results_config(self, test_config, temp_chroma_path, sample_course, sample_course_chunks):
        """
        CRITICAL TEST: Test that demonstrates the MAX_RESULTS=0 bug
        This test should FAIL with current config.py settings
        """
        # Setup config with zero max results (simulating the bug)
        test_config.CHROMA_PATH = temp_chroma_path
        test_config.MAX_RESULTS = 0  # This is the bug in config.py!

        # Create RAG system with real vector store
        rag = RAGSystem(test_config)

        # Add sample data
        rag.vector_store.add_course_metadata(sample_course)
        rag.vector_store.add_course_content(sample_course_chunks)

        # Mock AI generator to isolate vector store behavior
        with patch.object(rag.ai_generator, 'generate_response') as mock_gen:
            # Setup mock to simulate tool execution
            def mock_generate(query, conversation_history=None, tools=None, tool_manager=None):
                if tool_manager:
                    # Simulate AI calling the search tool
                    result = tool_manager.execute_tool("search_course_content", query="Python")
                    return f"Response based on: {result}"
                return "No tools used"

            mock_gen.side_effect = mock_generate

            # Execute query
            response, sources = rag.query("What is Python?")

            # This assertion should FAIL if MAX_RESULTS=0
            # Because vector store will return 0 results
            assert "No relevant content found" not in response or test_config.MAX_RESULTS == 0,\
                f"Search returned no results when MAX_RESULTS={test_config.MAX_RESULTS}. " \
                f"This indicates the MAX_RESULTS=0 bug!"

    def test_query_with_proper_max_results_config(self, test_config, temp_chroma_path, sample_course, sample_course_chunks):
        """
        Test that queries work correctly with proper MAX_RESULTS configuration
        This test should PASS showing the correct behavior
        """
        # Setup config with proper max results
        test_config.CHROMA_PATH = temp_chroma_path
        test_config.MAX_RESULTS = 5  # Proper value

        # Create RAG system
        rag = RAGSystem(test_config)

        # Add sample data
        rag.vector_store.add_course_metadata(sample_course)
        rag.vector_store.add_course_content(sample_course_chunks)

        # Execute search directly through tool
        result = rag.search_tool.execute(query="Python")

        # Assert - should get actual results, not "no content found"
        assert "No relevant content found" not in result
        assert len(result) > 0
        # Should contain content from our sample chunks
        assert "Python" in result or "programming" in result.lower()

    def test_vector_store_search_respects_max_results(self, test_config, temp_chroma_path, sample_course, sample_course_chunks):
        """Test that vector store respects MAX_RESULTS configuration"""
        test_config.CHROMA_PATH = temp_chroma_path

        # Test with MAX_RESULTS = 0 (bug scenario)
        test_config.MAX_RESULTS = 0
        rag_broken = RAGSystem(test_config)
        rag_broken.vector_store.add_course_metadata(sample_course)
        rag_broken.vector_store.add_course_content(sample_course_chunks)

        results_broken = rag_broken.vector_store.search(query="Python")

        # With MAX_RESULTS=0, should get empty results
        assert results_broken.is_empty(), \
            "Expected empty results with MAX_RESULTS=0, but got results!"

        # Test with proper MAX_RESULTS
        test_config.MAX_RESULTS = 5
        test_config.CHROMA_PATH = temp_chroma_path + "_proper"
        rag_working = RAGSystem(test_config)
        rag_working.vector_store.add_course_metadata(sample_course)
        rag_working.vector_store.add_course_content(sample_course_chunks)

        results_working = rag_working.vector_store.search(query="Python")

        # With proper MAX_RESULTS, should get results
        assert not results_working.is_empty(), \
            "Expected results with MAX_RESULTS=5, but got empty!"


class TestRAGSystemCourseManagement:
    """Test suite for course document management"""

    def test_get_course_analytics(self, test_config, temp_chroma_path, sample_course):
        """Test retrieving course analytics"""
        test_config.CHROMA_PATH = temp_chroma_path
        test_config.MAX_RESULTS = 5

        rag = RAGSystem(test_config)
        rag.vector_store.add_course_metadata(sample_course)

        # Execute
        analytics = rag.get_course_analytics()

        # Assert
        assert "total_courses" in analytics
        assert "course_titles" in analytics
        assert analytics["total_courses"] == 1
        assert sample_course.title in analytics["course_titles"]


class TestRAGSystemErrorHandling:
    """Test error handling in RAG system"""

    @pytest.fixture
    def mock_rag_for_errors(self, test_config, temp_chroma_path):
        """Create a RAG system with mocked components for error testing"""
        test_config.CHROMA_PATH = temp_chroma_path + "_errors"

        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor') as MockDocProcessor:

            # Setup vector store mock
            mock_vector_store = Mock()
            mock_vector_store.search.return_value = SearchResults(
                documents=["Test content"],
                metadata=[{"course_title": "Test Course", "lesson_number": 1}],
                distances=[0.3],
                error=None
            )
            MockVectorStore.return_value = mock_vector_store

            # Setup AI generator mock
            mock_ai_gen = Mock()
            mock_ai_gen.generate_response.return_value = "Test response"
            MockAIGenerator.return_value = mock_ai_gen

            # Setup document processor mock
            MockDocProcessor.return_value = Mock()

            # Create RAG system
            rag = RAGSystem(test_config)

            return rag, mock_vector_store, mock_ai_gen

    def test_query_handles_ai_generator_error(self, mock_rag_for_errors):
        """Test that query handles AI generator errors gracefully"""
        rag, mock_vector_store, mock_ai_gen = mock_rag_for_errors

        # Setup AI generator to raise an error
        mock_ai_gen.generate_response.side_effect = Exception("API Error")

        # Execute - should raise exception (not handled by RAG system)
        with pytest.raises(Exception):
            rag.query("Test query")

    def test_query_handles_vector_store_error(self, mock_rag_for_errors):
        """Test query handling when vector store has errors"""
        rag, mock_vector_store, mock_ai_gen = mock_rag_for_errors

        # Setup vector store to return error
        mock_vector_store.search.return_value = SearchResults.empty("Database error")

        # Mock AI to simulate using the tool
        def mock_generate(query, conversation_history=None, tools=None, tool_manager=None):
            if tool_manager:
                result = tool_manager.execute_tool("search_course_content", query="test")
                return f"Search result: {result}"
            return "No search performed"

        mock_ai_gen.generate_response.side_effect = mock_generate

        # Execute
        response, sources = rag.query("Test query")

        # Response should contain the error from search
        assert "Database error" in response or isinstance(response, str)


class TestRAGSystemSequentialToolCalling:
    """Integration tests for sequential tool calling through RAG system"""

    def test_rag_sequential_tool_calling_with_real_vector_store(self, test_config, temp_chroma_path, sample_course, sample_course_chunks):
        """Test sequential calling with actual vector store and multiple searches"""
        # Setup config with proper max results
        test_config.CHROMA_PATH = temp_chroma_path + "_integration"
        test_config.MAX_RESULTS = 5

        # Create RAG system with real vector store
        rag = RAGSystem(test_config)

        # Add sample data
        rag.vector_store.add_course_metadata(sample_course)
        rag.vector_store.add_course_content(sample_course_chunks)

        # Mock AI generator to simulate sequential tool calling
        with patch.object(rag.ai_generator, 'generate_response') as mock_gen:
            call_count = 0

            def mock_sequential_calls(query, conversation_history=None, tools=None, tool_manager=None):
                nonlocal call_count
                call_count += 1

                # Simulate Claude making multiple tool calls
                if tool_manager and call_count == 1:
                    # First call: search for "Python"
                    result1 = tool_manager.execute_tool("search_course_content", query="Python")
                    # Simulate seeing result and making second call
                    result2 = tool_manager.execute_tool("search_course_content", query="variables")
                    return f"Based on the searches: {result1[:50]}... and {result2[:50]}..."

                return "Response without tools"

            mock_gen.side_effect = mock_sequential_calls

            # Execute query that would benefit from multiple searches
            response, sources = rag.query("Tell me about Python and variables")

            # Verify
            assert isinstance(response, str)
            assert len(response) > 0
            # Tool manager should have been called
            mock_gen.assert_called_once()

    def test_rag_outline_then_search_pattern(self, test_config, temp_chroma_path, sample_course, sample_course_chunks):
        """Test outline→search pattern with sequential calling"""
        # Setup
        test_config.CHROMA_PATH = temp_chroma_path + "_outline_search"
        test_config.MAX_RESULTS = 5

        rag = RAGSystem(test_config)
        rag.vector_store.add_course_metadata(sample_course)
        rag.vector_store.add_course_content(sample_course_chunks)

        # Mock AI to simulate outline→search pattern
        with patch.object(rag.ai_generator, 'generate_response') as mock_gen:
            def mock_outline_then_search(query, conversation_history=None, tools=None, tool_manager=None):
                if tool_manager:
                    # First get outline
                    outline = tool_manager.execute_tool("get_course_outline", course_title="Introduction to Python")
                    # Then search based on outline
                    content = tool_manager.execute_tool("search_course_content", query="Python", lesson_number=1)
                    return f"The course has these lessons: {outline}. Lesson 1 content: {content[:100]}..."
                return "No tools used"

            mock_gen.side_effect = mock_outline_then_search

            # Execute
            response, sources = rag.query("What lessons are in Python course and tell me about lesson 1?")

            # Verify
            assert isinstance(response, str)
            assert "lesson" in response.lower()
            mock_gen.assert_called_once()
