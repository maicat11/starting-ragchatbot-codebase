"""
Pytest fixtures for RAG system testing
"""

import os
import shutil
import sys
import tempfile
from unittest.mock import Mock

import pytest

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator  # noqa: E402
from models import Course, CourseChunk, Lesson  # noqa: E402
from search_tools import CourseSearchTool, ToolManager  # noqa: E402
from vector_store import SearchResults, VectorStore  # noqa: E402


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Introduction to Python",
        course_link="https://example.com/python",
        instructor="John Doe",
        lessons=[
            Lesson(
                lesson_number=1,
                title="Getting Started",
                lesson_link="https://example.com/python/lesson1",
            ),
            Lesson(
                lesson_number=2,
                title="Variables and Types",
                lesson_link="https://example.com/python/lesson2",
            ),
            Lesson(
                lesson_number=3,
                title="Control Flow",
                lesson_link="https://example.com/python/lesson3",
            ),
        ],
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Lesson 1 content: Python is a high-level programming language. It is widely used for web development, data science, and automation.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="Variables in Python are created when you assign a value to them. Python is dynamically typed, meaning you don't need to declare variable types.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=1,
        ),
        CourseChunk(
            content="Control flow in Python includes if statements, for loops, and while loops. These allow you to control the execution flow of your program.",
            course_title=sample_course.title,
            lesson_number=3,
            chunk_index=2,
        ),
    ]


@pytest.fixture
def temp_chroma_path():
    """Create a temporary directory for ChromaDB testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing without actual ChromaDB"""
    mock = Mock(spec=VectorStore)

    # Setup default return values
    mock.search.return_value = SearchResults(
        documents=["Sample content from Python course"],
        metadata=[
            {
                "course_title": "Introduction to Python",
                "lesson_number": 1,
                "chunk_index": 0,
            }
        ],
        distances=[0.5],
        error=None,
    )

    mock._resolve_course_name.return_value = "Introduction to Python"
    mock.get_lesson_link.return_value = "https://example.com/python/lesson1"

    return mock


@pytest.fixture
def course_search_tool(mock_vector_store):
    """Create a CourseSearchTool with mocked vector store"""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def tool_manager(course_search_tool):
    """Create a ToolManager with registered search tool"""
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    return manager


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing"""
    mock_client = Mock()

    # Create a mock response object
    mock_response = Mock()
    mock_response.content = [Mock(text="This is a test response")]
    mock_response.stop_reason = "end_turn"

    mock_client.messages.create.return_value = mock_response

    return mock_client


@pytest.fixture
def ai_generator_with_mock(mock_anthropic_client):
    """Create an AIGenerator with mocked Anthropic client"""
    generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")
    generator.client = mock_anthropic_client
    return generator


@pytest.fixture
def test_config():
    """Create a test configuration"""

    class TestConfig:
        ANTHROPIC_API_KEY = "test_key"
        ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        CHUNK_SIZE = 800
        CHUNK_OVERLAP = 100
        MAX_RESULTS = 5  # Note: Testing with proper value
        MAX_HISTORY = 2
        CHROMA_PATH = None  # Will be set per test

    return TestConfig()
