"""
Pytest configuration and shared fixtures for the RAG system tests.

This module provides common fixtures for mocking dependencies and setting up
test data across all test modules.
"""
import pytest
import tempfile
import shutil
import os
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from config import config
from rag_system import RAGSystem
from models import Course, Lesson


class MockConfig:
    """Mock configuration for testing that mirrors the real config structure"""
    def __init__(self):
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 100
        self.CHROMA_PATH = "test_chroma"
        self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.MAX_RESULTS = 5
        self.ANTHROPIC_API_KEY = "test_key"
        self.ANTHROPIC_MODEL = "claude-3-sonnet-20241022"
        self.MAX_HISTORY = 10


@pytest.fixture
def mock_config():
    """Provide a mock configuration for testing"""
    return MockConfig()


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test isolation"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def isolated_config(temp_directory):
    """Provide an isolated config that doesn't interfere with real data"""
    original_chroma_path = config.CHROMA_PATH
    original_api_key = config.ANTHROPIC_API_KEY
    
    # Temporarily modify config for testing
    config.CHROMA_PATH = os.path.join(temp_directory, "test_chroma")
    config.ANTHROPIC_API_KEY = "test-key-12345"
    
    yield config
    
    # Restore original values
    config.CHROMA_PATH = original_chroma_path
    config.ANTHROPIC_API_KEY = original_api_key


@pytest.fixture
def sample_course():
    """Create a sample course object for testing"""
    lessons = [
        Lesson(
            title="Introduction to AI",
            content="Artificial Intelligence is the simulation of human intelligence...",
            video_url="https://example.com/video1",
            duration="15:30"
        ),
        Lesson(
            title="Machine Learning Basics",
            content="Machine learning is a subset of AI that enables systems to learn...",
            video_url="https://example.com/video2",
            duration="22:15"
        )
    ]
    
    return Course(
        title="Fundamentals of AI",
        instructor="Dr. Jane Smith",
        course_link="https://example.com/course",
        lessons=lessons
    )


@pytest.fixture
def sample_courses_list(sample_course):
    """Create a list of sample courses for testing"""
    course2 = Course(
        title="Advanced Machine Learning",
        instructor="Prof. John Doe",
        course_link="https://example.com/course2",
        lessons=[
            Lesson(
                title="Neural Networks",
                content="Neural networks are computing systems inspired by biological neural networks...",
                video_url="https://example.com/video3",
                duration="30:45"
            )
        ]
    )
    
    return [sample_course, course2]


@pytest.fixture
def mock_document_processor():
    """Mock DocumentProcessor for testing"""
    with patch('rag_system.DocumentProcessor') as mock_processor:
        mock_instance = MagicMock()
        mock_processor.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore for testing"""
    with patch('rag_system.VectorStore') as mock_store:
        mock_instance = MagicMock()
        mock_store.return_value = mock_instance
        
        # Default mock behaviors
        mock_instance.get_existing_course_titles.return_value = []
        mock_instance.get_course_count.return_value = 0
        
        yield mock_instance


@pytest.fixture
def mock_ai_generator():
    """Mock AIGenerator for testing"""
    with patch('rag_system.AIGenerator') as mock_generator:
        mock_instance = MagicMock()
        mock_generator.return_value = mock_instance
        
        # Default mock behavior
        mock_instance.generate_response.return_value = "Mocked AI response"
        
        yield mock_instance


@pytest.fixture
def mock_session_manager():
    """Mock SessionManager for testing"""
    with patch('rag_system.SessionManager') as mock_manager:
        mock_instance = MagicMock()
        mock_manager.return_value = mock_instance
        
        # Default mock behaviors
        mock_instance.create_session.return_value = "test_session_123"
        mock_instance.get_conversation_history.return_value = None
        
        yield mock_instance


@pytest.fixture
def mock_tool_manager():
    """Mock ToolManager for testing"""
    with patch('rag_system.ToolManager') as mock_manager:
        mock_instance = MagicMock()
        mock_manager.return_value = mock_instance
        
        # Default mock behaviors
        mock_instance.get_tool_definitions.return_value = []
        mock_instance.get_last_sources.return_value = []
        
        yield mock_instance


@pytest.fixture
def mock_rag_system(
    mock_config,
    mock_document_processor,
    mock_vector_store,
    mock_ai_generator,
    mock_session_manager,
    mock_tool_manager
):
    """Create a fully mocked RAG system for testing"""
    with patch('rag_system.CourseSearchTool'), patch('rag_system.CourseOutlineTool'):
        rag_system = RAGSystem(mock_config)
        return rag_system


@pytest.fixture
def test_app():
    """Create a test FastAPI app with minimal dependencies"""
    app = FastAPI(title="Test RAG System")
    
    # Mock the RAG system to avoid initialization issues
    with patch('app.rag_system') as mock_rag:
        mock_instance = MagicMock()
        mock_rag.return_value = mock_instance
        
        # Default mock behaviors for API endpoints
        mock_instance.query.return_value = ("Test response", [{"text": "Test source", "url": None}])
        mock_instance.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Course 1", "Course 2"]
        }
        mock_instance.session_manager.create_session.return_value = "test_session_123"
        
        # Import and configure the endpoints
        from app import QueryRequest, QueryResponse, CourseStats, SourceItem
        
        @app.post("/api/query", response_model=QueryResponse)
        async def query_documents(request: QueryRequest):
            session_id = request.session_id or mock_instance.session_manager.create_session()
            answer, sources = mock_instance.query(request.query, session_id)
            
            formatted_sources = []
            for source in sources:
                if isinstance(source, dict) and 'text' in source:
                    formatted_sources.append(SourceItem(text=source['text'], url=source.get('url')))
                else:
                    formatted_sources.append(SourceItem(text=str(source), url=None))
            
            return QueryResponse(
                answer=answer,
                sources=formatted_sources,
                session_id=session_id
            )
        
        @app.get("/api/courses", response_model=CourseStats)
        async def get_course_stats():
            analytics = mock_instance.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        
        @app.get("/")
        async def root():
            return {"message": "RAG System API", "status": "running"}
        
        yield app


@pytest.fixture
def test_client(test_app):
    """Create a test client for the FastAPI application"""
    return TestClient(test_app)


@pytest.fixture
def sample_query_request():
    """Sample query request data for API testing"""
    return {
        "query": "What is artificial intelligence?",
        "session_id": "test_session_123"
    }


@pytest.fixture
def sample_query_request_no_session():
    """Sample query request without session ID"""
    return {
        "query": "What is machine learning?"
    }


@pytest.fixture
def expected_query_response():
    """Expected response format for query API"""
    return {
        "answer": "Test response",
        "sources": [{"text": "Test source", "url": None}],
        "session_id": "test_session_123"
    }


@pytest.fixture
def expected_course_stats():
    """Expected response format for course stats API"""
    return {
        "total_courses": 2,
        "course_titles": ["Course 1", "Course 2"]
    }


# Utility fixture for error simulation
@pytest.fixture
def simulate_error():
    """Utility fixture to help simulate various error conditions"""
    def _simulate(error_type, message="Test error"):
        if error_type == "api_error":
            return Exception(f"API Error: {message}")
        elif error_type == "database_error":
            return Exception(f"Database Error: {message}")
        elif error_type == "validation_error":
            return ValueError(f"Validation Error: {message}")
        else:
            return Exception(message)
    
    return _simulate


# Auto-use fixtures for test isolation
@pytest.fixture(autouse=True)
def isolate_warnings():
    """Automatically suppress warnings during tests"""
    import warnings
    warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")


# Markers for different test categories
pytest_plugins = []


def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests for system workflows") 
    config.addinivalue_line("markers", "api: API endpoint tests")
    config.addinivalue_line("markers", "slow: Tests that take longer to run")