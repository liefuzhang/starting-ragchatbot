import pytest
import unittest.mock as mock
import os
from rag_system import RAGSystem
from models import Course, Lesson


class MockConfig:
    """Mock configuration for testing"""
    def __init__(self):
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 100
        self.CHROMA_PATH = "test_chroma"
        self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.MAX_RESULTS = 5
        self.ANTHROPIC_API_KEY = "test_key"
        self.ANTHROPIC_MODEL = "claude-3-sonnet-20241022"
        self.MAX_HISTORY = 10


class TestRAGSystem:
    """Test suite for RAG System functionality"""
    
    def setup_method(self):
        """Setup test fixtures before each test method"""
        self.mock_config = MockConfig()
        
        # Create patches for all dependencies
        self.mock_doc_processor = mock.patch('rag_system.DocumentProcessor').start()
        self.mock_vector_store = mock.patch('rag_system.VectorStore').start()
        self.mock_ai_generator = mock.patch('rag_system.AIGenerator').start()
        self.mock_session_manager = mock.patch('rag_system.SessionManager').start()
        self.mock_tool_manager = mock.patch('rag_system.ToolManager').start()
        self.mock_search_tool = mock.patch('rag_system.CourseSearchTool').start()
        self.mock_outline_tool = mock.patch('rag_system.CourseOutlineTool').start()
        
        # Create the RAG system
        self.rag_system = RAGSystem(self.mock_config)
    
    def teardown_method(self):
        """Clean up after each test method"""
        mock.patch.stopall()
    
    def test_initialization(self):
        """Test RAG system initialization"""
        # Verify all components were initialized with correct parameters
        self.mock_doc_processor.assert_called_once_with(1000, 100)
        self.mock_vector_store.assert_called_once_with("test_chroma", "all-MiniLM-L6-v2", 5)
        self.mock_ai_generator.assert_called_once_with("test_key", "claude-3-sonnet-20241022")
        self.mock_session_manager.assert_called_once_with(10)
        
        # Verify tools were registered
        tool_manager_instance = self.mock_tool_manager.return_value
        assert tool_manager_instance.register_tool.call_count == 2
    
    def test_add_course_document_success(self):
        """Test successful addition of a course document"""
        # Mock successful document processing
        mock_course = Course(
            title="Test Course",
            instructor="Test Instructor",
            course_link="http://test.com",
            lessons=[]
        )
        mock_chunks = [mock.MagicMock(), mock.MagicMock()]
        
        doc_processor_instance = self.mock_doc_processor.return_value
        doc_processor_instance.process_course_document.return_value = (mock_course, mock_chunks)
        
        vector_store_instance = self.mock_vector_store.return_value
        
        # Call the method
        result_course, result_chunks = self.rag_system.add_course_document("test_file.txt")
        
        # Verify document processing was called
        doc_processor_instance.process_course_document.assert_called_once_with("test_file.txt")
        
        # Verify vector store operations
        vector_store_instance.add_course_metadata.assert_called_once_with(mock_course)
        vector_store_instance.add_course_content.assert_called_once_with(mock_chunks)
        
        # Verify return values
        assert result_course == mock_course
        assert result_chunks == 2
    
    def test_add_course_document_failure(self):
        """Test handling of document processing failure"""
        doc_processor_instance = self.mock_doc_processor.return_value
        doc_processor_instance.process_course_document.side_effect = Exception("Processing failed")
        
        result_course, result_chunks = self.rag_system.add_course_document("bad_file.txt")
        
        assert result_course is None
        assert result_chunks == 0
    
    @mock.patch('rag_system.os.path.exists')
    @mock.patch('rag_system.os.listdir')
    @mock.patch('rag_system.os.path.isfile')
    def test_add_course_folder_success(self, mock_isfile, mock_listdir, mock_exists):
        """Test successful addition of course folder"""
        # Mock file system
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt", "course2.pdf", "readme.md"]
        mock_isfile.return_value = True
        
        # Mock vector store
        vector_store_instance = self.mock_vector_store.return_value
        vector_store_instance.get_existing_course_titles.return_value = []
        
        # Mock document processor
        doc_processor_instance = self.mock_doc_processor.return_value
        mock_course1 = Course(title="Course 1", instructor="Prof A", course_link="", lessons=[])
        mock_course2 = Course(title="Course 2", instructor="Prof B", course_link="", lessons=[])
        mock_chunks1 = [mock.MagicMock()]
        mock_chunks2 = [mock.MagicMock(), mock.MagicMock()]
        
        doc_processor_instance.process_course_document.side_effect = [
            (mock_course1, mock_chunks1),
            (mock_course2, mock_chunks2)
        ]
        
        total_courses, total_chunks = self.rag_system.add_course_folder("test_folder")
        
        # Verify calls
        assert doc_processor_instance.process_course_document.call_count == 2
        assert vector_store_instance.add_course_metadata.call_count == 2
        assert vector_store_instance.add_course_content.call_count == 2
        
        # Verify results
        assert total_courses == 2
        assert total_chunks == 3
    
    @mock.patch('rag_system.os.path.exists')
    def test_add_course_folder_nonexistent(self, mock_exists):
        """Test handling of nonexistent folder"""
        mock_exists.return_value = False
        
        total_courses, total_chunks = self.rag_system.add_course_folder("nonexistent_folder")
        
        assert total_courses == 0
        assert total_chunks == 0
    
    @mock.patch('rag_system.os.path.exists')
    @mock.patch('rag_system.os.listdir')
    @mock.patch('rag_system.os.path.isfile')
    def test_add_course_folder_with_clear_existing(self, mock_isfile, mock_listdir, mock_exists):
        """Test folder addition with clear_existing=True"""
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt"]
        mock_isfile.return_value = True
        
        vector_store_instance = self.mock_vector_store.return_value
        vector_store_instance.get_existing_course_titles.return_value = []
        
        doc_processor_instance = self.mock_doc_processor.return_value
        mock_course = Course(title="Course 1", instructor="Prof", course_link="", lessons=[])
        doc_processor_instance.process_course_document.return_value = (mock_course, [])
        
        self.rag_system.add_course_folder("test_folder", clear_existing=True)
        
        # Verify clear_all_data was called
        vector_store_instance.clear_all_data.assert_called_once()
    
    @mock.patch('rag_system.os.path.exists')
    @mock.patch('rag_system.os.listdir')
    @mock.patch('rag_system.os.path.isfile')
    def test_add_course_folder_skip_existing(self, mock_isfile, mock_listdir, mock_exists):
        """Test that existing courses are skipped"""
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt"]
        mock_isfile.return_value = True
        
        vector_store_instance = self.mock_vector_store.return_value
        vector_store_instance.get_existing_course_titles.return_value = ["Course 1"]
        
        doc_processor_instance = self.mock_doc_processor.return_value
        mock_course = Course(title="Course 1", instructor="Prof", course_link="", lessons=[])
        doc_processor_instance.process_course_document.return_value = (mock_course, [])
        
        total_courses, total_chunks = self.rag_system.add_course_folder("test_folder")
        
        # Should not add to vector store since course already exists
        assert vector_store_instance.add_course_metadata.call_count == 0
        assert vector_store_instance.add_course_content.call_count == 0
        assert total_courses == 0
        assert total_chunks == 0
    
    def test_query_without_session(self):
        """Test query processing without session context"""
        # Mock AI generator response
        ai_generator_instance = self.mock_ai_generator.return_value
        ai_generator_instance.generate_response.return_value = "Test response"
        
        # Mock tool manager
        tool_manager_instance = self.mock_tool_manager.return_value
        tool_manager_instance.get_tool_definitions.return_value = [{"name": "test_tool"}]
        tool_manager_instance.get_last_sources.return_value = [{"text": "Source", "url": "http://test.com"}]
        
        response, sources = self.rag_system.query("What is AI?")
        
        # Verify AI generator was called correctly
        ai_generator_instance.generate_response.assert_called_once_with(
            query="Answer this question about course materials: What is AI?",
            conversation_history=None,
            tools=[{"name": "test_tool"}],
            tool_manager=tool_manager_instance
        )
        
        # Verify sources were retrieved and reset
        tool_manager_instance.get_last_sources.assert_called_once()
        tool_manager_instance.reset_sources.assert_called_once()
        
        assert response == "Test response"
        assert sources == [{"text": "Source", "url": "http://test.com"}]
    
    def test_query_with_session(self):
        """Test query processing with session context"""
        session_manager_instance = self.mock_session_manager.return_value
        session_manager_instance.get_conversation_history.return_value = "Previous conversation"
        
        ai_generator_instance = self.mock_ai_generator.return_value
        ai_generator_instance.generate_response.return_value = "Contextual response"
        
        tool_manager_instance = self.mock_tool_manager.return_value
        tool_manager_instance.get_tool_definitions.return_value = []
        tool_manager_instance.get_last_sources.return_value = []
        
        response, sources = self.rag_system.query("Follow up question", session_id="session_123")
        
        # Verify session manager interactions
        session_manager_instance.get_conversation_history.assert_called_once_with("session_123")
        session_manager_instance.add_exchange.assert_called_once_with(
            "session_123", 
            "Follow up question", 
            "Contextual response"
        )
        
        # Verify AI generator used conversation history
        ai_generator_instance.generate_response.assert_called_once_with(
            query="Answer this question about course materials: Follow up question",
            conversation_history="Previous conversation",
            tools=[],
            tool_manager=tool_manager_instance
        )
        
        assert response == "Contextual response"
    
    def test_query_error_handling(self):
        """Test query error handling when AI generator fails"""
        ai_generator_instance = self.mock_ai_generator.return_value
        ai_generator_instance.generate_response.side_effect = Exception("API failure")
        
        tool_manager_instance = self.mock_tool_manager.return_value
        tool_manager_instance.get_tool_definitions.return_value = []
        tool_manager_instance.get_last_sources.return_value = []
        
        # Should raise the exception since RAGSystem doesn't handle AI generator errors
        with pytest.raises(Exception, match="API failure"):
            self.rag_system.query("Test query")
    
    def test_get_course_analytics(self):
        """Test course analytics functionality"""
        vector_store_instance = self.mock_vector_store.return_value
        vector_store_instance.get_course_count.return_value = 5
        vector_store_instance.get_existing_course_titles.return_value = ["Course A", "Course B"]
        
        analytics = self.rag_system.get_course_analytics()
        
        vector_store_instance.get_course_count.assert_called_once()
        vector_store_instance.get_existing_course_titles.assert_called_once()
        
        assert analytics["total_courses"] == 5
        assert analytics["course_titles"] == ["Course A", "Course B"]
    
    def test_integration_query_flow(self):
        """Test the complete query flow integration"""
        # This test verifies the complete flow without mocking individual method calls
        session_manager_instance = self.mock_session_manager.return_value
        session_manager_instance.get_conversation_history.return_value = "Context"
        
        ai_generator_instance = self.mock_ai_generator.return_value
        ai_generator_instance.generate_response.return_value = "Generated answer"
        
        tool_manager_instance = self.mock_tool_manager.return_value
        tool_manager_instance.get_tool_definitions.return_value = [{"name": "search_tool"}]
        tool_manager_instance.get_last_sources.return_value = [{"text": "Test Source"}]
        
        # Execute query
        response, sources = self.rag_system.query("Complex question", session_id="test_session")
        
        # Verify the complete flow
        assert session_manager_instance.get_conversation_history.called
        assert ai_generator_instance.generate_response.called
        assert tool_manager_instance.get_tool_definitions.called
        assert tool_manager_instance.get_last_sources.called
        assert tool_manager_instance.reset_sources.called
        assert session_manager_instance.add_exchange.called
        
        assert response == "Generated answer"
        assert sources == [{"text": "Test Source"}]


if __name__ == "__main__":
    pytest.main([__file__])