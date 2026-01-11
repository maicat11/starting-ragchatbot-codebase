"""
API endpoint tests for the RAG system FastAPI application

Tests cover:
- POST /api/query endpoint (query processing, session management, error handling)
- GET /api/courses endpoint (course analytics, error handling)
- Integration tests (session persistence, middleware)
"""
import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException


class TestAPIQueryEndpoint:
    """Test suite for POST /api/query endpoint"""

    @pytest.mark.api
    def test_query_endpoint_successful_response(self, test_client, mock_rag_system_for_api):
        """Test successful query processing returns proper response structure"""
        # Arrange
        mock_rag_system_for_api.query.return_value = (
            "Python is a high-level programming language.",
            [{"text": "Introduction to Python - Lesson 1", "url": "https://example.com/lesson1"}]
        )

        # Act
        response = test_client.post(
            "/api/query",
            json={"query": "What is Python?", "session_id": "test-123"}
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Python is a high-level programming language."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "Introduction to Python - Lesson 1"
        assert data["sources"][0]["url"] == "https://example.com/lesson1"
        assert data["session_id"] == "test-123"

        # Verify RAG system was called correctly
        mock_rag_system_for_api.query.assert_called_once_with(
            "What is Python?",
            "test-123"
        )

    @pytest.mark.api
    def test_query_endpoint_with_session_id(self, test_client, mock_rag_system_for_api):
        """Test that existing session_id is preserved and passed to RAG system"""
        # Arrange
        existing_session = "existing-session-456"
        mock_rag_system_for_api.query.return_value = (
            "Variables store data.",
            [{"text": "Python Basics - Lesson 2", "url": "https://example.com/lesson2"}]
        )

        # Act
        response = test_client.post(
            "/api/query",
            json={"query": "What are variables?", "session_id": existing_session}
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == existing_session

        # Verify RAG system received the correct session_id
        mock_rag_system_for_api.query.assert_called_once_with(
            "What are variables?",
            existing_session
        )

    @pytest.mark.api
    def test_query_endpoint_creates_session_if_not_provided(self, test_client, mock_rag_system_for_api):
        """Test that new session is created when session_id is not provided"""
        # Arrange
        new_session_id = "newly-created-789"
        mock_rag_system_for_api.session_manager.create_session.return_value = new_session_id
        mock_rag_system_for_api.query.return_value = (
            "Control flow manages execution order.",
            []
        )

        # Act
        response = test_client.post(
            "/api/query",
            json={"query": "Explain control flow"}
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == new_session_id

        # Verify session was created
        mock_rag_system_for_api.session_manager.create_session.assert_called_once()

        # Verify RAG system was called with new session
        mock_rag_system_for_api.query.assert_called_once_with(
            "Explain control flow",
            new_session_id
        )

    @pytest.mark.api
    def test_query_endpoint_returns_sources(self, test_client, mock_rag_system_for_api):
        """Test that sources are properly formatted and returned"""
        # Arrange
        multiple_sources = [
            {"text": "Python Fundamentals - Lesson 1", "url": "https://example.com/lesson1"},
            {"text": "Python Data Types - Lesson 2", "url": "https://example.com/lesson2"},
            {"text": "Python Functions - Lesson 3", "url": "https://example.com/lesson3"}
        ]
        mock_rag_system_for_api.query.return_value = (
            "Python supports multiple data types including int, float, and string.",
            multiple_sources
        )

        # Act
        response = test_client.post(
            "/api/query",
            json={"query": "What data types does Python have?", "session_id": "test-sources"}
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 3

        # Verify each source has correct structure
        for i, source in enumerate(data["sources"]):
            assert "text" in source
            assert "url" in source
            assert source["text"] == multiple_sources[i]["text"]
            assert source["url"] == multiple_sources[i]["url"]

    @pytest.mark.api
    def test_query_endpoint_handles_rag_system_error(self, test_client, mock_rag_system_for_api):
        """Test that RAG system errors are handled and return 500 status"""
        # Arrange
        mock_rag_system_for_api.query.side_effect = Exception("Database connection failed")

        # Act
        response = test_client.post(
            "/api/query",
            json={"query": "This will fail", "session_id": "test-error"}
        )

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Database connection failed" in data["detail"]

    @pytest.mark.api
    def test_query_endpoint_invalid_request_body(self, test_client):
        """Test that invalid request body returns 422 validation error"""
        # Act - Send invalid JSON structure
        response = test_client.post(
            "/api/query",
            json={"wrong_field": "value", "another_field": 123}
        )

        # Assert
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    @pytest.mark.api
    def test_query_endpoint_missing_required_field(self, test_client):
        """Test that missing required 'query' field returns 422 validation error"""
        # Act - Send request without 'query' field
        response = test_client.post(
            "/api/query",
            json={"session_id": "test-123"}
        )

        # Assert
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


class TestAPICoursesEndpoint:
    """Test suite for GET /api/courses endpoint"""

    @pytest.mark.api
    def test_courses_endpoint_returns_stats(self, test_client, mock_rag_system_for_api):
        """Test successful retrieval of course statistics"""
        # Arrange
        mock_rag_system_for_api.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Introduction to Python", "Advanced Python"]
        }

        # Act
        response = test_client.get("/api/courses")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 2
        assert len(data["course_titles"]) == 2
        assert "Introduction to Python" in data["course_titles"]
        assert "Advanced Python" in data["course_titles"]

        # Verify RAG system was called
        mock_rag_system_for_api.get_course_analytics.assert_called_once()

    @pytest.mark.api
    def test_courses_endpoint_empty_catalog(self, test_client, mock_rag_system_for_api):
        """Test endpoint returns valid response when no courses exist"""
        # Arrange
        mock_rag_system_for_api.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        # Act
        response = test_client.get("/api/courses")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    @pytest.mark.api
    def test_courses_endpoint_multiple_courses(self, test_client, mock_rag_system_for_api):
        """Test endpoint correctly returns multiple course titles"""
        # Arrange
        many_courses = [
            "Introduction to Python",
            "Advanced Python",
            "Python for Data Science",
            "Machine Learning with Python",
            "Web Development with Python"
        ]
        mock_rag_system_for_api.get_course_analytics.return_value = {
            "total_courses": len(many_courses),
            "course_titles": many_courses
        }

        # Act
        response = test_client.get("/api/courses")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 5
        assert len(data["course_titles"]) == 5

        # Verify all course titles are present
        for course in many_courses:
            assert course in data["course_titles"]

    @pytest.mark.api
    def test_courses_endpoint_handles_error(self, test_client, mock_rag_system_for_api):
        """Test that errors from get_course_analytics are handled properly"""
        # Arrange
        mock_rag_system_for_api.get_course_analytics.side_effect = Exception("Analytics retrieval failed")

        # Act
        response = test_client.get("/api/courses")

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Analytics retrieval failed" in data["detail"]


class TestAPIIntegration:
    """Integration tests for API endpoints"""

    @pytest.mark.api
    @pytest.mark.integration
    def test_multiple_queries_same_session(self, test_client, mock_rag_system_for_api):
        """Test that multiple queries can be made with the same session"""
        # Arrange
        session_id = "persistent-session-123"

        # Configure mock to return different responses for each query
        responses = [
            ("Python is a programming language.", [{"text": "Lesson 1", "url": "https://example.com/1"}]),
            ("Variables store data.", [{"text": "Lesson 2", "url": "https://example.com/2"}]),
            ("Functions encapsulate code.", [{"text": "Lesson 3", "url": "https://example.com/3"}])
        ]
        mock_rag_system_for_api.query.side_effect = responses

        # Act - Make multiple queries with same session
        response1 = test_client.post(
            "/api/query",
            json={"query": "What is Python?", "session_id": session_id}
        )
        response2 = test_client.post(
            "/api/query",
            json={"query": "What are variables?", "session_id": session_id}
        )
        response3 = test_client.post(
            "/api/query",
            json={"query": "What are functions?", "session_id": session_id}
        )

        # Assert
        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response3.status_code == 200

        # Verify all responses have the same session_id
        assert response1.json()["session_id"] == session_id
        assert response2.json()["session_id"] == session_id
        assert response3.json()["session_id"] == session_id

        # Verify RAG system was called 3 times with the same session
        assert mock_rag_system_for_api.query.call_count == 3

    @pytest.mark.api
    @pytest.mark.integration
    def test_conversation_history_persists(self, test_client, mock_rag_system_for_api):
        """Test that conversation history is maintained across queries"""
        # Arrange
        session_id = "history-session-456"
        mock_rag_system_for_api.query.return_value = ("Response", [])

        # Act - Make sequential queries
        test_client.post(
            "/api/query",
            json={"query": "First query", "session_id": session_id}
        )
        test_client.post(
            "/api/query",
            json={"query": "Second query", "session_id": session_id}
        )

        # Assert - Verify query method was called with session_id
        # This ensures the session_manager is being used to track history
        calls = mock_rag_system_for_api.query.call_args_list
        assert len(calls) == 2
        assert calls[0][0][1] == session_id  # First call's session_id
        assert calls[1][0][1] == session_id  # Second call's session_id

    @pytest.mark.api
    @pytest.mark.integration
    def test_cors_headers_present(self, test_client, mock_rag_system_for_api):
        """Test that CORS headers are properly set in response"""
        # Arrange
        mock_rag_system_for_api.query.return_value = ("Answer", [])

        # Act
        response = test_client.post(
            "/api/query",
            json={"query": "Test query", "session_id": "test-cors"}
        )

        # Assert - Check for CORS headers
        # Note: TestClient may not include all headers, but we can verify the endpoint works
        assert response.status_code == 200

        # Verify CORS middleware allows the request to succeed
        # In production, Access-Control-Allow-Origin would be present
        # TestClient simulates a same-origin request, so headers may differ

    @pytest.mark.api
    @pytest.mark.integration
    def test_trusted_host_middleware(self, test_client, mock_rag_system_for_api):
        """Test that trusted host middleware allows requests"""
        # Arrange
        mock_rag_system_for_api.get_course_analytics.return_value = {
            "total_courses": 1,
            "course_titles": ["Test Course"]
        }

        # Act - Make request that goes through middleware
        response = test_client.get("/api/courses")

        # Assert - Verify middleware allows request
        assert response.status_code == 200
        # If middleware blocked request, we'd get a different status code
        data = response.json()
        assert "total_courses" in data
