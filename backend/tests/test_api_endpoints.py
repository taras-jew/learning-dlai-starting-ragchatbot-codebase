import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
import json

from vector_store import SearchResults


@pytest.mark.api
class TestAPIEndpoints:
    """Test FastAPI endpoints for the RAG system"""
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint returns basic info"""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "RAG System API"
        assert data["status"] == "running"
    
    def test_query_endpoint_success(self, test_client, mock_rag_system, sample_search_results):
        """Test successful query endpoint"""
        # Setup mock response
        mock_rag_system.query.return_value = ("This is the answer", ["Source 1"])
        mock_rag_system.session_manager.create_session.return_value = "test_session"
        
        # Make request
        response = test_client.post("/api/query", json={
            "query": "What is machine learning?"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "This is the answer"
        assert data["sources"] == ["Source 1"]
        assert "session_id" in data
        
        # Verify RAG system was called
        mock_rag_system.query.assert_called_once()
        mock_rag_system.session_manager.create_session.assert_called_once()
    
    def test_query_endpoint_with_session(self, test_client, mock_rag_system):
        """Test query endpoint with existing session ID"""
        # Setup mock response
        mock_rag_system.query.return_value = ("Follow-up answer", ["Source 2"])
        
        # Make request with session ID
        response = test_client.post("/api/query", json={
            "query": "Follow-up question",
            "session_id": "existing_session_123"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "Follow-up answer"
        assert data["session_id"] == "existing_session_123"
        
        # Verify query was called with session
        mock_rag_system.query.assert_called_once_with("Follow-up question", "existing_session_123")
        # Session should not be created when provided
        mock_rag_system.session_manager.create_session.assert_not_called()
    
    def test_query_endpoint_missing_query(self, test_client):
        """Test query endpoint with missing query field"""
        response = test_client.post("/api/query", json={})
        
        assert response.status_code == 422  # Validation error
        assert "query" in response.text.lower()
    
    def test_query_endpoint_empty_query(self, test_client):
        """Test query endpoint with empty query"""
        response = test_client.post("/api/query", json={"query": ""})
        
        # Should accept empty query (validation at business logic level)
        assert response.status_code == 200 or response.status_code == 500
    
    def test_query_endpoint_rag_system_error(self, test_client, mock_rag_system):
        """Test query endpoint when RAG system raises exception"""
        # Setup mock to raise exception
        mock_rag_system.query.side_effect = Exception("RAG system error")
        mock_rag_system.session_manager.create_session.return_value = "test_session"
        
        response = test_client.post("/api/query", json={
            "query": "Test query"
        })
        
        assert response.status_code == 500
        data = response.json()
        assert "RAG system error" in data["detail"]
    
    def test_courses_endpoint_success(self, test_client, mock_rag_system):
        """Test successful courses endpoint"""
        # Setup mock analytics
        mock_analytics = {
            "total_courses": 3,
            "course_titles": ["Course A", "Course B", "Course C"]
        }
        mock_rag_system.get_course_analytics.return_value = mock_analytics
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Course A" in data["course_titles"]
        
        mock_rag_system.get_course_analytics.assert_called_once()
    
    def test_courses_endpoint_empty_courses(self, test_client, mock_rag_system):
        """Test courses endpoint with no courses"""
        # Setup mock analytics for empty state
        mock_analytics = {
            "total_courses": 0,
            "course_titles": []
        }
        mock_rag_system.get_course_analytics.return_value = mock_analytics
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
    
    def test_courses_endpoint_error(self, test_client, mock_rag_system):
        """Test courses endpoint when analytics fails"""
        # Setup mock to raise exception
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 500
        data = response.json()
        assert "Analytics error" in data["detail"]
    
    def test_delete_session_success(self, test_client, mock_rag_system):
        """Test successful session deletion"""
        session_id = "test_session_to_delete"
        
        response = test_client.delete(f"/api/sessions/{session_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert session_id in data["message"]
        
        mock_rag_system.session_manager.clear_session.assert_called_once_with(session_id)
    
    def test_delete_session_error(self, test_client, mock_rag_system):
        """Test session deletion when clear_session fails"""
        # Setup mock to raise exception
        mock_rag_system.session_manager.clear_session.side_effect = Exception("Session not found")
        
        session_id = "nonexistent_session"
        response = test_client.delete(f"/api/sessions/{session_id}")
        
        assert response.status_code == 500
        data = response.json()
        assert "Session not found" in data["detail"]
    
    def test_invalid_endpoint(self, test_client):
        """Test request to non-existent endpoint"""
        response = test_client.get("/api/nonexistent")
        
        assert response.status_code == 404
    
    def test_invalid_http_method(self, test_client):
        """Test using wrong HTTP method on endpoint"""
        # GET on POST endpoint
        response = test_client.get("/api/query")
        assert response.status_code == 405  # Method not allowed
        
        # POST on GET endpoint
        response = test_client.post("/api/courses")
        assert response.status_code == 405  # Method not allowed


@pytest.mark.api
class TestAPIRequestValidation:
    """Test API request validation and edge cases"""
    
    def test_query_request_validation_extra_fields(self, test_client, mock_rag_system):
        """Test query request with extra unexpected fields"""
        mock_rag_system.query.return_value = ("Answer", [])
        mock_rag_system.session_manager.create_session.return_value = "test"
        
        response = test_client.post("/api/query", json={
            "query": "Valid query",
            "extra_field": "should be ignored",
            "another_field": 123
        })
        
        # FastAPI should ignore extra fields by default
        assert response.status_code == 200
    
    def test_query_request_wrong_types(self, test_client):
        """Test query request with wrong field types"""
        # Query as number instead of string
        response = test_client.post("/api/query", json={
            "query": 123
        })
        assert response.status_code == 422
        
        # Session ID as number instead of string
        response = test_client.post("/api/query", json={
            "query": "Valid query",
            "session_id": 123
        })
        assert response.status_code == 422
    
    def test_query_request_malformed_json(self, test_client):
        """Test query request with malformed JSON"""
        response = test_client.post("/api/query", 
                                  content='{"query": "test"',  # Missing closing brace
                                  headers={"content-type": "application/json"})
        
        assert response.status_code == 422
    
    def test_query_request_content_type(self, test_client):
        """Test query endpoint with different content types"""
        # Test with form data instead of JSON
        response = test_client.post("/api/query", data={"query": "test"})
        
        # Should handle form data or return appropriate error
        assert response.status_code in [200, 422]
    
    def test_large_query_payload(self, test_client, mock_rag_system):
        """Test query with very large payload"""
        large_query = "x" * 10000  # 10KB query
        
        mock_rag_system.query.return_value = ("Answer", [])
        mock_rag_system.session_manager.create_session.return_value = "test"
        
        response = test_client.post("/api/query", json={
            "query": large_query
        })
        
        # Should handle large queries (or reject with appropriate limit)
        assert response.status_code in [200, 413, 422]  # 413 = Payload Too Large


@pytest.mark.api 
@pytest.mark.slow
class TestAPIPerformance:
    """Test API performance and concurrent requests"""
    
    def test_concurrent_queries(self, test_client, mock_rag_system):
        """Test multiple concurrent queries"""
        import threading
        import time
        
        # Setup mock response
        mock_rag_system.query.return_value = ("Concurrent answer", [])
        mock_rag_system.session_manager.create_session.return_value = "test"
        
        results = []
        threads = []
        
        def make_request(query_num):
            response = test_client.post("/api/query", json={
                "query": f"Query {query_num}",
                "session_id": f"session_{query_num}"
            })
            results.append(response.status_code)
        
        # Create 5 concurrent requests
        for i in range(5):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all requests succeeded
        assert len(results) == 5
        assert all(status == 200 for status in results)
    
    def test_query_response_time(self, test_client, mock_rag_system):
        """Test query response time is reasonable"""
        import time
        
        mock_rag_system.query.return_value = ("Fast answer", [])
        mock_rag_system.session_manager.create_session.return_value = "test"
        
        start_time = time.time()
        response = test_client.post("/api/query", json={
            "query": "Performance test query"
        })
        end_time = time.time()
        
        assert response.status_code == 200
        # Response should be under 5 seconds for mocked system
        assert (end_time - start_time) < 5.0


@pytest.mark.api
class TestAPIErrorHandling:
    """Test API error handling and edge cases"""
    
    def test_session_manager_error_in_query(self, test_client, mock_rag_system):
        """Test query when session manager fails"""
        # Setup session creation to fail
        mock_rag_system.session_manager.create_session.side_effect = Exception("Session manager error")
        
        response = test_client.post("/api/query", json={
            "query": "Test query"
        })
        
        assert response.status_code == 500
        assert "Session manager error" in response.json()["detail"]
    
    def test_query_with_unicode_content(self, test_client, mock_rag_system):
        """Test query with unicode/international characters"""
        mock_rag_system.query.return_value = ("Unicode answer: é, ñ, 中文", [])
        mock_rag_system.session_manager.create_session.return_value = "test"
        
        response = test_client.post("/api/query", json={
            "query": "Query with unicode: é, ñ, 中文, 🚀"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "Unicode answer" in data["answer"]
    
    def test_special_characters_in_session_id(self, test_client, mock_rag_system):
        """Test session operations with special characters in session ID"""
        import urllib.parse
        
        special_session = "session-with-special@chars"  # Use URL-safe characters
        encoded_session = urllib.parse.quote(special_session, safe="")
        
        # Test delete with special characters
        response = test_client.delete(f"/api/sessions/{encoded_session}")
        
        # Should handle URL encoding properly
        assert response.status_code in [200, 500]  # 500 if session manager rejects format
        
        # Verify the session manager was called (it will decode the URL parameter)
        mock_rag_system.session_manager.clear_session.assert_called_once()