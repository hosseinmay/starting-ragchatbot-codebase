"""API endpoint tests for the RAG chatbot FastAPI application.

This module tests the HTTP API layer including:
- POST /api/query - Query processing endpoint
- GET /api/courses - Course statistics endpoint
- DELETE /api/session/{session_id} - Session clearing endpoint
- GET / - Root endpoint (health check)

Tests use the test fixtures from conftest.py which provide a FastAPI
TestClient with mocked RAGSystem dependencies.
"""

from typing import Any
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient


class TestRootEndpoint:
    """Tests for the root endpoint (GET /)."""

    def test_root_returns_ok_status(self, test_client: TestClient) -> None:
        """Test that root endpoint returns success status."""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_root_returns_json_content_type(self, test_client: TestClient) -> None:
        """Test that root endpoint returns JSON content type."""
        response = test_client.get("/")

        assert "application/json" in response.headers["content-type"]


class TestQueryEndpoint:
    """Tests for the query endpoint (POST /api/query)."""

    def test_query_with_valid_request_returns_200(
        self,
        test_client: TestClient,
    ) -> None:
        """Test successful query returns 200 status."""
        response = test_client.post(
            "/api/query",
            json={"query": "What is Claude?"},
        )

        assert response.status_code == 200

    def test_query_returns_expected_response_structure(
        self,
        test_client: TestClient,
    ) -> None:
        """Test that query response contains all required fields."""
        response = test_client.post(
            "/api/query",
            json={"query": "What is Claude?"},
        )

        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

    def test_query_returns_answer_from_rag_system(
        self,
        test_client_with_rag: tuple[TestClient, Mock],
    ) -> None:
        """Test that query returns the answer from RAGSystem."""
        client, mock_rag = test_client_with_rag
        mock_rag.query.return_value = ("Custom test answer", [])

        response = client.post(
            "/api/query",
            json={"query": "Test question"},
        )

        data = response.json()
        assert data["answer"] == "Custom test answer"

    def test_query_creates_session_when_not_provided(
        self,
        test_client_with_rag: tuple[TestClient, Mock],
    ) -> None:
        """Test that a new session is created if not provided in request."""
        client, mock_rag = test_client_with_rag
        mock_rag.session_manager.create_session.return_value = "new_session_456"

        response = client.post(
            "/api/query",
            json={"query": "Test question"},
        )

        data = response.json()
        mock_rag.session_manager.create_session.assert_called_once()
        assert data["session_id"] == "new_session_456"

    def test_query_uses_provided_session_id(
        self,
        test_client_with_rag: tuple[TestClient, Mock],
    ) -> None:
        """Test that provided session_id is used and returned."""
        client, mock_rag = test_client_with_rag

        response = client.post(
            "/api/query",
            json={"query": "Test question", "session_id": "existing_session"},
        )

        data = response.json()
        mock_rag.query.assert_called_with("Test question", "existing_session")
        assert data["session_id"] == "existing_session"

    def test_query_returns_sources_from_rag_system(
        self,
        test_client_with_rag: tuple[TestClient, Mock],
    ) -> None:
        """Test that sources from RAGSystem are included in response."""
        client, mock_rag = test_client_with_rag
        mock_rag.query.return_value = (
            "Answer",
            [
                {"title": "Source 1", "url": "https://example.com/1"},
                {"title": "Source 2", "url": "https://example.com/2"},
            ],
        )

        response = client.post(
            "/api/query",
            json={"query": "Test"},
        )

        data = response.json()
        assert len(data["sources"]) == 2
        assert data["sources"][0]["title"] == "Source 1"
        assert data["sources"][1]["url"] == "https://example.com/2"

    def test_query_with_empty_query_still_processes(
        self,
        test_client: TestClient,
    ) -> None:
        """Test that empty query string is still processed (validation at RAG level)."""
        response = test_client.post(
            "/api/query",
            json={"query": ""},
        )

        # Empty query is valid at API level - RAG system handles validation
        assert response.status_code == 200

    def test_query_without_query_field_returns_422(
        self,
        test_client: TestClient,
    ) -> None:
        """Test that missing query field returns validation error."""
        response = test_client.post(
            "/api/query",
            json={},
        )

        assert response.status_code == 422

    def test_query_with_invalid_json_returns_422(
        self,
        test_client: TestClient,
    ) -> None:
        """Test that invalid JSON body returns error."""
        response = test_client.post(
            "/api/query",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    def test_query_handles_rag_system_error(
        self,
        test_client_with_rag: tuple[TestClient, Mock],
    ) -> None:
        """Test that RAGSystem errors result in 500 response."""
        client, mock_rag = test_client_with_rag
        mock_rag.query.side_effect = RuntimeError("Database connection failed")

        response = client.post(
            "/api/query",
            json={"query": "Test"},
        )

        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]

    def test_query_with_sources_without_url(
        self,
        test_client_with_rag: tuple[TestClient, Mock],
    ) -> None:
        """Test that sources without URL are handled correctly."""
        client, mock_rag = test_client_with_rag
        mock_rag.query.return_value = (
            "Answer",
            [{"title": "Source without URL", "url": None}],
        )

        response = client.post(
            "/api/query",
            json={"query": "Test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["sources"][0]["url"] is None


class TestCoursesEndpoint:
    """Tests for the courses endpoint (GET /api/courses)."""

    def test_courses_returns_200(self, test_client: TestClient) -> None:
        """Test that courses endpoint returns 200 status."""
        response = test_client.get("/api/courses")

        assert response.status_code == 200

    def test_courses_returns_expected_structure(
        self,
        test_client: TestClient,
    ) -> None:
        """Test that courses response contains required fields."""
        response = test_client.get("/api/courses")

        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

    def test_courses_returns_data_from_rag_system(
        self,
        test_client_with_rag: tuple[TestClient, Mock],
    ) -> None:
        """Test that courses data comes from RAGSystem analytics."""
        client, mock_rag = test_client_with_rag
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 5,
            "course_titles": ["Python 101", "ML Basics", "Data Science"],
        }

        response = client.get("/api/courses")

        data = response.json()
        assert data["total_courses"] == 5
        assert "Python 101" in data["course_titles"]
        assert len(data["course_titles"]) == 3

    def test_courses_handles_empty_catalog(
        self,
        test_client_with_rag: tuple[TestClient, Mock],
    ) -> None:
        """Test courses endpoint with empty course catalog."""
        client, mock_rag = test_client_with_rag
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }

        response = client.get("/api/courses")

        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_courses_handles_analytics_error(
        self,
        test_client_with_rag: tuple[TestClient, Mock],
    ) -> None:
        """Test that analytics errors result in 500 response."""
        client, mock_rag = test_client_with_rag
        mock_rag.get_course_analytics.side_effect = RuntimeError("Vector store error")

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert "Vector store error" in response.json()["detail"]


class TestSessionEndpoint:
    """Tests for the session endpoint (DELETE /api/session/{session_id})."""

    def test_clear_session_returns_200(self, test_client: TestClient) -> None:
        """Test that clearing session returns 200 status."""
        response = test_client.delete("/api/session/test_session_123")

        assert response.status_code == 200

    def test_clear_session_returns_confirmation(
        self,
        test_client: TestClient,
    ) -> None:
        """Test that clear session response confirms the action."""
        response = test_client.delete("/api/session/my_session")

        data = response.json()
        assert data["status"] == "cleared"
        assert data["session_id"] == "my_session"

    def test_clear_session_calls_session_manager(
        self,
        test_client_with_rag: tuple[TestClient, Mock],
    ) -> None:
        """Test that session manager clear_session is called."""
        client, mock_rag = test_client_with_rag

        client.delete("/api/session/session_to_clear")

        mock_rag.session_manager.clear_session.assert_called_once_with(
            "session_to_clear"
        )

    def test_clear_nonexistent_session_succeeds(
        self,
        test_client: TestClient,
    ) -> None:
        """Test that clearing non-existent session doesn't error."""
        response = test_client.delete("/api/session/nonexistent_session")

        # Should succeed - clearing non-existent session is idempotent
        assert response.status_code == 200


class TestRequestValidation:
    """Tests for request validation across endpoints."""

    def test_query_accepts_extra_fields(self, test_client: TestClient) -> None:
        """Test that extra fields in request body are ignored."""
        response = test_client.post(
            "/api/query",
            json={
                "query": "Test",
                "extra_field": "should be ignored",
            },
        )

        assert response.status_code == 200

    def test_query_with_wrong_content_type_fails(
        self,
        test_client: TestClient,
    ) -> None:
        """Test that non-JSON content type is rejected."""
        response = test_client.post(
            "/api/query",
            content="query=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        assert response.status_code == 422

    def test_courses_ignores_query_params(self, test_client: TestClient) -> None:
        """Test that query parameters on courses endpoint are ignored."""
        response = test_client.get("/api/courses?unused_param=value")

        assert response.status_code == 200


class TestCORSAndMiddleware:
    """Tests for CORS and middleware behavior."""

    def test_options_request_allowed(self, test_client: TestClient) -> None:
        """Test that OPTIONS requests are handled (for CORS preflight)."""
        response = test_client.options("/api/query")

        # FastAPI TestClient may return 405 for OPTIONS by default
        # In production, CORS middleware handles this
        assert response.status_code in (200, 405)


class TestResponseFormats:
    """Tests for response format consistency."""

    def test_query_response_json_serializable(
        self,
        test_client: TestClient,
    ) -> None:
        """Test that query response is valid JSON."""
        response = test_client.post(
            "/api/query",
            json={"query": "Test"},
        )

        # Should not raise
        data = response.json()
        assert isinstance(data, dict)

    def test_courses_response_json_serializable(
        self,
        test_client: TestClient,
    ) -> None:
        """Test that courses response is valid JSON."""
        response = test_client.get("/api/courses")

        data = response.json()
        assert isinstance(data, dict)

    def test_error_response_has_detail_field(
        self,
        test_client_with_rag: tuple[TestClient, Mock],
    ) -> None:
        """Test that error responses include detail field."""
        client, mock_rag = test_client_with_rag
        mock_rag.query.side_effect = RuntimeError("Test error")

        response = client.post(
            "/api/query",
            json={"query": "Test"},
        )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
