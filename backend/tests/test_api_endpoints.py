"""
API endpoint tests for the FastAPI RAG system.

Tests all FastAPI endpoints including request/response validation,
error handling, and proper integration with the RAG system components.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json


@pytest.mark.api
class TestQueryEndpoint:
    """Test suite for /api/query endpoint"""
    
    def test_query_endpoint_success_with_session(self, test_client, sample_query_request):
        """Test successful query with session ID"""
        response = test_client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        
        # Verify response content
        assert data["answer"] == "Test response"
        assert data["session_id"] == "test_session_123"
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) > 0
        
        # Verify source structure
        source = data["sources"][0]
        assert "text" in source
        assert "url" in source
        assert source["text"] == "Test source"
        assert source["url"] is None
    
    def test_query_endpoint_success_without_session(self, test_client, sample_query_request_no_session):
        """Test successful query without session ID (auto-creation)"""
        response = test_client.post("/api/query", json=sample_query_request_no_session)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify that session was created automatically
        assert data["session_id"] == "test_session_123"
        assert data["answer"] == "Test response"
    
    def test_query_endpoint_with_sources_url(self, test_client):
        """Test query response with sources that include URLs"""
        request_data = {"query": "Test query with URL sources"}
        
        # Mock rag_system to return sources with URLs
        with patch('app.rag_system') as mock_rag:
            mock_rag.query.return_value = (
                "Response with URL sources",
                [
                    {"text": "Source with URL", "url": "https://example.com/page1"},
                    {"text": "Source without URL", "url": None},
                    "String source"  # Old format
                ]
            )
            mock_rag.session_manager.create_session.return_value = "session_with_urls"
            
            response = test_client.post("/api/query", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert len(data["sources"]) == 3
            assert data["sources"][0]["text"] == "Source with URL"
            assert data["sources"][0]["url"] == "https://example.com/page1"
            assert data["sources"][1]["url"] is None
            assert data["sources"][2]["text"] == "String source"
            assert data["sources"][2]["url"] is None
    
    def test_query_endpoint_validation_error(self, test_client):
        """Test query endpoint with invalid request data"""
        # Missing required 'query' field
        invalid_request = {"session_id": "test_session"}
        
        response = test_client.post("/api/query", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    def test_query_endpoint_empty_query(self, test_client):
        """Test query endpoint with empty query string"""
        empty_query = {"query": ""}
        
        response = test_client.post("/api/query", json=empty_query)
        # Should still be valid (empty string is valid), but may return different response
        assert response.status_code == 200
    
    def test_query_endpoint_internal_error(self, test_client):
        """Test query endpoint when RAG system raises an exception"""
        request_data = {"query": "This will cause an error"}
        
        with patch('app.rag_system') as mock_rag:
            mock_rag.query.side_effect = Exception("Internal system error")
            
            response = test_client.post("/api/query", json=request_data)
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Internal system error" in data["detail"]
    
    def test_query_endpoint_session_manager_error(self, test_client):
        """Test query when session manager fails to create session"""
        request_data = {"query": "Test query"}
        
        with patch('app.rag_system') as mock_rag:
            mock_rag.session_manager.create_session.side_effect = Exception("Session creation failed")
            
            response = test_client.post("/api/query", json=request_data)
            
            assert response.status_code == 500
    
    def test_query_endpoint_malformed_json(self, test_client):
        """Test query endpoint with malformed JSON"""
        response = test_client.post(
            "/api/query",
            data="{invalid_json}",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


@pytest.mark.api
class TestCoursesEndpoint:
    """Test suite for /api/courses endpoint"""
    
    def test_courses_endpoint_success(self, test_client, expected_course_stats):
        """Test successful retrieval of course statistics"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "total_courses" in data
        assert "course_titles" in data
        
        # Verify response content
        assert data["total_courses"] == expected_course_stats["total_courses"]
        assert data["course_titles"] == expected_course_stats["course_titles"]
        assert isinstance(data["course_titles"], list)
    
    def test_courses_endpoint_no_courses(self, test_client):
        """Test courses endpoint when no courses are loaded"""
        with patch('app.rag_system') as mock_rag:
            mock_rag.get_course_analytics.return_value = {
                "total_courses": 0,
                "course_titles": []
            }
            
            response = test_client.get("/api/courses")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_courses"] == 0
            assert data["course_titles"] == []
    
    def test_courses_endpoint_internal_error(self, test_client):
        """Test courses endpoint when analytics system fails"""
        with patch('app.rag_system') as mock_rag:
            mock_rag.get_course_analytics.side_effect = Exception("Analytics system failure")
            
            response = test_client.get("/api/courses")
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Analytics system failure" in data["detail"]
    
    def test_courses_endpoint_large_dataset(self, test_client):
        """Test courses endpoint with large number of courses"""
        large_course_list = [f"Course {i}" for i in range(100)]
        
        with patch('app.rag_system') as mock_rag:
            mock_rag.get_course_analytics.return_value = {
                "total_courses": 100,
                "course_titles": large_course_list
            }
            
            response = test_client.get("/api/courses")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_courses"] == 100
            assert len(data["course_titles"]) == 100


@pytest.mark.api  
class TestRootEndpoint:
    """Test suite for root endpoint"""
    
    def test_root_endpoint_success(self, test_client):
        """Test successful access to root endpoint"""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "status" in data
        assert data["message"] == "RAG System API"
        assert data["status"] == "running"


@pytest.mark.api
class TestRequestResponseModels:
    """Test Pydantic model validation"""
    
    def test_query_request_model_validation(self, test_client):
        """Test QueryRequest model validation"""
        # Valid requests
        valid_requests = [
            {"query": "Test query"},
            {"query": "Test query", "session_id": "session123"},
            {"query": "Test query", "session_id": None}
        ]
        
        for request_data in valid_requests:
            response = test_client.post("/api/query", json=request_data)
            assert response.status_code == 200
    
    def test_query_request_model_invalid_types(self, test_client):
        """Test QueryRequest model with invalid data types"""
        invalid_requests = [
            {"query": 123},  # query should be string
            {"query": "Valid query", "session_id": 123},  # session_id should be string or None
            {"query": None},  # query cannot be None
        ]
        
        for request_data in invalid_requests:
            response = test_client.post("/api/query", json=request_data)
            assert response.status_code == 422
    
    def test_source_item_model_variations(self, test_client):
        """Test different source item formats are handled correctly"""
        with patch('app.rag_system') as mock_rag:
            # Test various source formats
            mock_rag.query.return_value = (
                "Mixed source formats response",
                [
                    {"text": "Dict source with URL", "url": "https://example.com"},
                    {"text": "Dict source without URL"},
                    {"text": "Dict source with None URL", "url": None},
                    "Plain string source"
                ]
            )
            mock_rag.session_manager.create_session.return_value = "test_session"
            
            response = test_client.post("/api/query", json={"query": "test"})
            
            assert response.status_code == 200
            data = response.json()
            sources = data["sources"]
            
            # All should be converted to SourceItem format
            for source in sources:
                assert "text" in source
                assert "url" in source


@pytest.mark.api
class TestEndpointIntegration:
    """Integration tests across multiple endpoints"""
    
    def test_query_then_courses_workflow(self, test_client):
        """Test typical workflow of querying then checking courses"""
        # First, make a query
        query_response = test_client.post("/api/query", json={"query": "What courses are available?"})
        assert query_response.status_code == 200
        
        # Then, check course statistics
        courses_response = test_client.get("/api/courses")
        assert courses_response.status_code == 200
        
        # Both should succeed independently
        query_data = query_response.json()
        courses_data = courses_response.json()
        
        assert "answer" in query_data
        assert "total_courses" in courses_data
    
    def test_multiple_queries_same_session(self, test_client):
        """Test multiple queries with the same session ID"""
        session_id = "persistent_session"
        
        queries = [
            "What is artificial intelligence?",
            "Tell me more about machine learning",
            "How does deep learning work?"
        ]
        
        for query in queries:
            response = test_client.post("/api/query", json={
                "query": query,
                "session_id": session_id
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == session_id
    
    def test_concurrent_requests_different_sessions(self, test_client):
        """Test handling of concurrent requests with different sessions"""
        import concurrent.futures
        
        def make_query(session_num):
            return test_client.post("/api/query", json={
                "query": f"Query from session {session_num}",
                "session_id": f"session_{session_num}"
            })
        
        # Simulate concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_query, i) for i in range(3)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data
            assert data["session_id"].startswith("session_") or data["session_id"] == "test_session_123"


@pytest.mark.api
@pytest.mark.slow
class TestEndpointPerformance:
    """Performance and stress tests for API endpoints"""
    
    def test_query_endpoint_response_time(self, test_client):
        """Test that query endpoint responds within reasonable time"""
        import time
        
        start_time = time.time()
        response = test_client.post("/api/query", json={"query": "Performance test query"})
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 5.0  # Should respond within 5 seconds
    
    def test_courses_endpoint_response_time(self, test_client):
        """Test that courses endpoint responds quickly"""
        import time
        
        start_time = time.time()
        response = test_client.get("/api/courses")
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should respond within 1 second
    
    def test_many_sequential_requests(self, test_client):
        """Test handling many sequential requests"""
        num_requests = 10
        
        for i in range(num_requests):
            response = test_client.post("/api/query", json={
                "query": f"Sequential request {i}"
            })
            assert response.status_code == 200
    
    def test_large_query_handling(self, test_client):
        """Test endpoint with very large query text"""
        large_query = "What is artificial intelligence? " * 100  # Repeat 100 times
        
        response = test_client.post("/api/query", json={"query": large_query})
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])