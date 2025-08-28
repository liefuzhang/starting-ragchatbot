import pytest
import unittest.mock as mock
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults, VectorStore


class TestCourseSearchTool:
    """Test suite for CourseSearchTool functionality"""
    
    def setup_method(self):
        """Setup test fixtures before each test method"""
        self.mock_vector_store = mock.MagicMock(spec=VectorStore)
        self.search_tool = CourseSearchTool(self.mock_vector_store)
    
    def test_get_tool_definition(self):
        """Test that tool definition is correctly structured"""
        definition = self.search_tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["properties"]["query"]["type"] == "string"
        assert "query" in definition["input_schema"]["required"]
    
    def test_execute_successful_search(self):
        """Test successful search execution with results"""
        # Mock successful search results
        mock_results = SearchResults(
            documents=["Test content from course"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.8],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test query")
        
        # Verify vector store was called correctly
        self.mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=None
        )
        
        # Verify result formatting
        assert "[Test Course - Lesson 1]" in result
        assert "Test content from course" in result
    
    def test_execute_with_course_filter(self):
        """Test search execution with course name filter"""
        mock_results = SearchResults(
            documents=["Filtered content"],
            metadata=[{"course_title": "Filtered Course", "lesson_number": 2}],
            distances=[0.7],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test query", course_name="Filtered Course")
        
        self.mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Filtered Course",
            lesson_number=None
        )
        
        assert "[Filtered Course - Lesson 2]" in result
    
    def test_execute_with_lesson_filter(self):
        """Test search execution with lesson number filter"""
        mock_results = SearchResults(
            documents=["Lesson specific content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 3}],
            distances=[0.6],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test query", lesson_number=3)
        
        self.mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=3
        )
        
        assert "[Test Course - Lesson 3]" in result
    
    def test_execute_error_handling(self):
        """Test handling of search errors"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Database connection failed"
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test query")
        
        assert result == "Database connection failed"
    
    def test_execute_empty_results(self):
        """Test handling of empty search results"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("nonexistent query")
        
        assert "No relevant content found" in result
    
    def test_execute_empty_results_with_filters(self):
        """Test empty results message includes filter information"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("query", course_name="Test Course", lesson_number=5)
        
        assert "No relevant content found in course 'Test Course' in lesson 5" in result
    
    def test_format_results_with_sources(self):
        """Test that sources are properly tracked for UI"""
        self.mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson1"
        
        mock_results = SearchResults(
            documents=["Content with link"],
            metadata=[{"course_title": "Linked Course", "lesson_number": 1}],
            distances=[0.9],
            error=None
        )
        
        result = self.search_tool._format_results(mock_results)
        
        # Check that last_sources is populated
        assert len(self.search_tool.last_sources) == 1
        source = self.search_tool.last_sources[0]
        assert source["text"] == "Linked Course - Lesson 1"
        assert source["url"] == "http://example.com/lesson1"


class TestCourseOutlineTool:
    """Test suite for CourseOutlineTool functionality"""
    
    def setup_method(self):
        """Setup test fixtures before each test method"""
        self.mock_vector_store = mock.MagicMock(spec=VectorStore)
        self.outline_tool = CourseOutlineTool(self.mock_vector_store)
    
    def test_get_tool_definition(self):
        """Test that tool definition is correctly structured"""
        definition = self.outline_tool.get_tool_definition()
        
        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["properties"]["course_name"]["type"] == "string"
        assert "course_name" in definition["input_schema"]["required"]
    
    def test_execute_successful_outline(self):
        """Test successful course outline retrieval"""
        # Mock course resolution
        self.mock_vector_store._resolve_course_name.return_value = "Resolved Course Title"
        
        # Mock course metadata
        mock_course_metadata = {
            "title": "Resolved Course Title",
            "course_link": "http://example.com/course",
            "lessons": [
                {"lesson_number": 1, "lesson_title": "Introduction"},
                {"lesson_number": 2, "lesson_title": "Advanced Topics"}
            ]
        }
        self.mock_vector_store.get_all_courses_metadata.return_value = [mock_course_metadata]
        
        result = self.outline_tool.execute("Course")
        
        # Verify calls
        self.mock_vector_store._resolve_course_name.assert_called_once_with("Course")
        self.mock_vector_store.get_all_courses_metadata.assert_called_once()
        
        # Verify result format
        assert "Course: Resolved Course Title" in result
        assert "Course Link: http://example.com/course" in result
        assert "Lessons (2 total):" in result
        assert "1. Introduction" in result
        assert "2. Advanced Topics" in result
    
    def test_execute_course_not_found(self):
        """Test handling when course cannot be resolved"""
        self.mock_vector_store._resolve_course_name.return_value = None
        
        result = self.outline_tool.execute("Nonexistent Course")
        
        assert result == "No course found matching 'Nonexistent Course'"
    
    def test_execute_metadata_not_found(self):
        """Test handling when course metadata is missing"""
        self.mock_vector_store._resolve_course_name.return_value = "Resolved Course"
        self.mock_vector_store.get_all_courses_metadata.return_value = []
        
        result = self.outline_tool.execute("Course")
        
        assert result == "Course metadata not found for 'Resolved Course'"


class TestToolManager:
    """Test suite for ToolManager functionality"""
    
    def setup_method(self):
        """Setup test fixtures before each test method"""
        self.tool_manager = ToolManager()
        self.mock_tool = mock.MagicMock()
        self.mock_tool.get_tool_definition.return_value = {
            "name": "test_tool",
            "description": "Test tool"
        }
        self.mock_tool.execute.return_value = "Test result"
    
    def test_register_tool(self):
        """Test tool registration"""
        self.tool_manager.register_tool(self.mock_tool)
        
        assert "test_tool" in self.tool_manager.tools
        assert self.tool_manager.tools["test_tool"] == self.mock_tool
    
    def test_register_tool_without_name(self):
        """Test error handling for tool without name"""
        bad_tool = mock.MagicMock()
        bad_tool.get_tool_definition.return_value = {"description": "No name"}
        
        with pytest.raises(ValueError, match="Tool must have a 'name' in its definition"):
            self.tool_manager.register_tool(bad_tool)
    
    def test_get_tool_definitions(self):
        """Test getting all tool definitions"""
        self.tool_manager.register_tool(self.mock_tool)
        
        definitions = self.tool_manager.get_tool_definitions()
        
        assert len(definitions) == 1
        assert definitions[0]["name"] == "test_tool"
    
    def test_execute_tool(self):
        """Test tool execution"""
        self.tool_manager.register_tool(self.mock_tool)
        
        result = self.tool_manager.execute_tool("test_tool", param1="value1")
        
        self.mock_tool.execute.assert_called_once_with(param1="value1")
        assert result == "Test result"
    
    def test_execute_nonexistent_tool(self):
        """Test handling of nonexistent tool execution"""
        result = self.tool_manager.execute_tool("nonexistent_tool")
        
        assert result == "Tool 'nonexistent_tool' not found"
    
    def test_get_last_sources(self):
        """Test getting sources from tools"""
        # Create mock tool with sources
        tool_with_sources = mock.MagicMock()
        tool_with_sources.get_tool_definition.return_value = {"name": "source_tool"}
        tool_with_sources.last_sources = [{"text": "Test Source", "url": "http://test.com"}]
        
        self.tool_manager.register_tool(tool_with_sources)
        
        sources = self.tool_manager.get_last_sources()
        
        assert len(sources) == 1
        assert sources[0]["text"] == "Test Source"
    
    def test_reset_sources(self):
        """Test resetting sources from all tools"""
        tool_with_sources = mock.MagicMock()
        tool_with_sources.get_tool_definition.return_value = {"name": "source_tool"}
        tool_with_sources.last_sources = [{"text": "Test Source"}]
        
        self.tool_manager.register_tool(tool_with_sources)
        self.tool_manager.reset_sources()
        
        assert tool_with_sources.last_sources == []


if __name__ == "__main__":
    pytest.main([__file__])