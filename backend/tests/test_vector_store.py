import pytest
import unittest.mock as mock
from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestSearchResults:
    """Test suite for SearchResults class"""
    
    def test_from_chroma_with_results(self):
        """Test SearchResults creation from ChromaDB results"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'title': 'Course A'}, {'title': 'Course B'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == ['doc1', 'doc2']
        assert results.metadata == [{'title': 'Course A'}, {'title': 'Course B'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None
    
    def test_from_chroma_empty_results(self):
        """Test SearchResults creation with empty ChromaDB results"""
        chroma_results = {
            'documents': [],
            'metadatas': [],
            'distances': []
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error is None
    
    def test_empty_with_error(self):
        """Test SearchResults.empty() class method"""
        results = SearchResults.empty("Test error message")
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "Test error message"
    
    def test_is_empty_true(self):
        """Test is_empty() returns True for empty results"""
        results = SearchResults([], [], [])
        assert results.is_empty() is True
    
    def test_is_empty_false(self):
        """Test is_empty() returns False for non-empty results"""
        results = SearchResults(['doc'], [{}], [0.1])
        assert results.is_empty() is False


class TestVectorStore:
    """Test suite for VectorStore functionality"""
    
    def setup_method(self):
        """Setup test fixtures before each test method"""
        # Mock ChromaDB and related components
        self.mock_chromadb = mock.patch('vector_store.chromadb').start()
        self.mock_settings = mock.patch('vector_store.Settings').start()
        
        # Mock client and collections
        self.mock_client = mock.MagicMock()
        self.mock_chromadb.PersistentClient.return_value = self.mock_client
        
        self.mock_catalog_collection = mock.MagicMock()
        self.mock_content_collection = mock.MagicMock()
        
        # Configure get_or_create_collection to return different collections
        def side_effect(name, embedding_function):
            if name == "course_catalog":
                return self.mock_catalog_collection
            elif name == "course_content":
                return self.mock_content_collection
            return mock.MagicMock()
        
        self.mock_client.get_or_create_collection.side_effect = side_effect
        
        # Create VectorStore instance
        self.vector_store = VectorStore("test_path", "test_model", 5)
    
    def teardown_method(self):
        """Clean up after each test method"""
        mock.patch.stopall()
    
    def test_initialization(self):
        """Test VectorStore initialization"""
        # Verify ChromaDB client creation
        self.mock_chromadb.PersistentClient.assert_called_once_with(
            path="test_path",
            settings=self.mock_settings.return_value
        )
        
        # Verify collections were created
        assert self.mock_client.get_or_create_collection.call_count == 2
        
        # Verify instance attributes
        assert self.vector_store.max_results == 5
        assert self.vector_store.course_catalog == self.mock_catalog_collection
        assert self.vector_store.course_content == self.mock_content_collection
    
    def test_search_successful(self):
        """Test successful search operation"""
        # Mock successful search results
        mock_query_results = {
            'documents': [['Found content']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 1}]],
            'distances': [[0.1]]
        }
        self.mock_content_collection.query.return_value = mock_query_results
        
        results = self.vector_store.search("test query")
        
        # Verify query was called correctly
        self.mock_content_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where=None
        )
        
        # Verify results
        assert results.documents == ['Found content']
        assert results.metadata == [{'course_title': 'Test Course', 'lesson_number': 1}]
        assert results.distances == [0.1]
        assert results.error is None
    
    def test_search_with_course_filter(self):
        """Test search with course name filtering"""
        # Mock course name resolution
        self.mock_catalog_collection.query.return_value = {
            'documents': [['Course Title']],
            'metadatas': [[{'title': 'Resolved Course'}]]
        }
        
        # Mock content search
        mock_query_results = {
            'documents': [['Filtered content']],
            'metadatas': [[{'course_title': 'Resolved Course'}]],
            'distances': [[0.2]]
        }
        self.mock_content_collection.query.return_value = mock_query_results
        
        results = self.vector_store.search("test query", course_name="Course")
        
        # Verify course resolution was called
        self.mock_catalog_collection.query.assert_called_once_with(
            query_texts=["Course"],
            n_results=1
        )
        
        # Verify content search with filter
        self.mock_content_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where={"course_title": "Resolved Course"}
        )
    
    def test_search_course_not_found(self):
        """Test search when course cannot be resolved"""
        # Mock failed course resolution
        self.mock_catalog_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }
        
        results = self.vector_store.search("test query", course_name="Nonexistent Course")
        
        assert results.error == "No course found matching 'Nonexistent Course'"
        assert results.is_empty()
    
    def test_search_with_lesson_filter(self):
        """Test search with lesson number filtering"""
        mock_query_results = {
            'documents': [['Lesson content']],
            'metadatas': [[{'lesson_number': 3}]],
            'distances': [[0.3]]
        }
        self.mock_content_collection.query.return_value = mock_query_results
        
        results = self.vector_store.search("test query", lesson_number=3)
        
        self.mock_content_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where={"lesson_number": 3}
        )
    
    def test_search_with_combined_filters(self):
        """Test search with both course and lesson filters"""
        # Mock course resolution
        self.mock_catalog_collection.query.return_value = {
            'documents': [['Course Title']],
            'metadatas': [[{'title': 'Test Course'}]]
        }
        
        mock_query_results = {
            'documents': [['Combined filter content']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 2}]],
            'distances': [[0.15]]
        }
        self.mock_content_collection.query.return_value = mock_query_results
        
        results = self.vector_store.search("test query", course_name="Test", lesson_number=2)
        
        expected_filter = {"$and": [
            {"course_title": "Test Course"},
            {"lesson_number": 2}
        ]}
        
        self.mock_content_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where=expected_filter
        )
    
    def test_search_error_handling(self):
        """Test search error handling"""
        self.mock_content_collection.query.side_effect = Exception("Database error")
        
        results = self.vector_store.search("test query")
        
        assert results.error == "Search error: Database error"
        assert results.is_empty()
    
    def test_add_course_metadata(self):
        """Test adding course metadata"""
        lessons = [
            Lesson(lesson_number=1, title="Lesson 1", lesson_link="http://lesson1.com"),
            Lesson(lesson_number=2, title="Lesson 2", lesson_link="http://lesson2.com")
        ]
        course = Course(
            title="Test Course",
            instructor="Test Instructor",
            course_link="http://course.com",
            lessons=lessons
        )
        
        self.vector_store.add_course_metadata(course)
        
        # Verify the catalog collection was called correctly
        self.mock_catalog_collection.add.assert_called_once()
        call_args = self.mock_catalog_collection.add.call_args[1]
        
        assert call_args["documents"] == ["Test Course"]
        assert call_args["ids"] == ["Test Course"]
        
        metadata = call_args["metadatas"][0]
        assert metadata["title"] == "Test Course"
        assert metadata["instructor"] == "Test Instructor"
        assert metadata["course_link"] == "http://course.com"
        assert metadata["lesson_count"] == 2
        
        # Verify lessons JSON structure
        import json
        lessons_data = json.loads(metadata["lessons_json"])
        assert len(lessons_data) == 2
        assert lessons_data[0]["lesson_number"] == 1
        assert lessons_data[0]["lesson_title"] == "Lesson 1"
        assert lessons_data[0]["lesson_link"] == "http://lesson1.com"
    
    def test_add_course_content(self):
        """Test adding course content chunks"""
        chunks = [
            CourseChunk(
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0,
                content="Chunk content 1"
            ),
            CourseChunk(
                course_title="Test Course",
                lesson_number=1,
                chunk_index=1,
                content="Chunk content 2"
            )
        ]
        
        self.vector_store.add_course_content(chunks)
        
        # Verify the content collection was called correctly
        self.mock_content_collection.add.assert_called_once()
        call_args = self.mock_content_collection.add.call_args[1]
        
        assert call_args["documents"] == ["Chunk content 1", "Chunk content 2"]
        assert call_args["ids"] == ["Test_Course_0", "Test_Course_1"]
        
        metadata = call_args["metadatas"]
        assert len(metadata) == 2
        assert metadata[0]["course_title"] == "Test Course"
        assert metadata[0]["lesson_number"] == 1
        assert metadata[0]["chunk_index"] == 0
    
    def test_add_course_content_empty(self):
        """Test adding empty course content"""
        self.vector_store.add_course_content([])
        
        # Should not call add on empty chunks
        self.mock_content_collection.add.assert_not_called()
    
    def test_clear_all_data(self):
        """Test clearing all data"""
        self.vector_store.clear_all_data()
        
        # Verify collections were deleted
        self.mock_client.delete_collection.assert_any_call("course_catalog")
        self.mock_client.delete_collection.assert_any_call("course_content")
        
        # Verify collections were recreated
        assert self.mock_client.get_or_create_collection.call_count >= 4  # 2 initial + 2 after clear
    
    def test_get_existing_course_titles(self):
        """Test getting existing course titles"""
        self.mock_catalog_collection.get.return_value = {
            'ids': ['Course A', 'Course B', 'Course C']
        }
        
        titles = self.vector_store.get_existing_course_titles()
        
        self.mock_catalog_collection.get.assert_called_once()
        assert titles == ['Course A', 'Course B', 'Course C']
    
    def test_get_existing_course_titles_empty(self):
        """Test getting existing course titles when empty"""
        self.mock_catalog_collection.get.return_value = {'ids': []}
        
        titles = self.vector_store.get_existing_course_titles()
        
        assert titles == []
    
    def test_get_course_count(self):
        """Test getting course count"""
        self.mock_catalog_collection.get.return_value = {
            'ids': ['Course A', 'Course B']
        }
        
        count = self.vector_store.get_course_count()
        
        assert count == 2
    
    def test_get_all_courses_metadata(self):
        """Test getting all courses metadata with JSON parsing"""
        import json
        lessons_json = json.dumps([
            {"lesson_number": 1, "lesson_title": "Intro", "lesson_link": "http://lesson1.com"}
        ])
        
        self.mock_catalog_collection.get.return_value = {
            'metadatas': [{
                'title': 'Test Course',
                'instructor': 'Prof Test',
                'lessons_json': lessons_json,
                'lesson_count': 1
            }]
        }
        
        metadata = self.vector_store.get_all_courses_metadata()
        
        assert len(metadata) == 1
        course_meta = metadata[0]
        assert course_meta['title'] == 'Test Course'
        assert course_meta['instructor'] == 'Prof Test'
        assert 'lessons_json' not in course_meta  # Should be removed after parsing
        assert 'lessons' in course_meta  # Should be added after parsing
        assert len(course_meta['lessons']) == 1
        assert course_meta['lessons'][0]['lesson_title'] == 'Intro'
    
    def test_get_lesson_link(self):
        """Test getting lesson link"""
        import json
        lessons_json = json.dumps([
            {"lesson_number": 1, "lesson_title": "Intro", "lesson_link": "http://lesson1.com"},
            {"lesson_number": 2, "lesson_title": "Advanced", "lesson_link": "http://lesson2.com"}
        ])
        
        self.mock_catalog_collection.get.return_value = {
            'metadatas': [{
                'lessons_json': lessons_json
            }]
        }
        
        link = self.vector_store.get_lesson_link("Test Course", 2)
        
        self.mock_catalog_collection.get.assert_called_once_with(ids=["Test Course"])
        assert link == "http://lesson2.com"
    
    def test_get_lesson_link_not_found(self):
        """Test getting lesson link when lesson not found"""
        import json
        lessons_json = json.dumps([
            {"lesson_number": 1, "lesson_title": "Intro", "lesson_link": "http://lesson1.com"}
        ])
        
        self.mock_catalog_collection.get.return_value = {
            'metadatas': [{
                'lessons_json': lessons_json
            }]
        }
        
        link = self.vector_store.get_lesson_link("Test Course", 999)
        
        assert link is None


if __name__ == "__main__":
    pytest.main([__file__])