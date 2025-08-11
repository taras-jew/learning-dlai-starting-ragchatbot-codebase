import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile
import shutil

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestSearchResults:
    """Test suite for SearchResults dataclass"""
    
    def test_from_chroma_with_data(self):
        """Test creating SearchResults from ChromaDB data"""
        chroma_data = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'course': 'A'}, {'course': 'B'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_data)
        
        assert results.documents == ['doc1', 'doc2']
        assert results.metadata == [{'course': 'A'}, {'course': 'B'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None
    
    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB data"""
        chroma_data = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        results = SearchResults.from_chroma(chroma_data)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error is None
    
    def test_from_chroma_none_data(self):
        """Test creating SearchResults from None ChromaDB data"""
        chroma_data = {
            'documents': None,
            'metadatas': None,
            'distances': None
        }
        
        results = SearchResults.from_chroma(chroma_data)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
    
    def test_empty_with_error(self):
        """Test creating empty SearchResults with error"""
        results = SearchResults.empty("Test error message")
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "Test error message"
    
    def test_is_empty_true(self):
        """Test is_empty returns True for empty results"""
        results = SearchResults(documents=[], metadata=[], distances=[])
        assert results.is_empty() is True
    
    def test_is_empty_false(self):
        """Test is_empty returns False for non-empty results"""
        results = SearchResults(documents=['doc'], metadata=[{}], distances=[0.1])
        assert results.is_empty() is False


class TestVectorStore:
    """Test suite for VectorStore functionality"""
    
    @pytest.fixture(autouse=True)
    def setup_vector_store(self):
        """Set up test fixtures with mocked dependencies"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock sentence transformers to avoid model download
        with patch('sentence_transformers.SentenceTransformer') as mock_st, \
             patch('vector_store.chromadb.PersistentClient') as mock_client_class:
            
            # Mock sentence transformer
            mock_embedding_func = Mock()
            mock_st.return_value = mock_embedding_func
            
            # Mock ChromaDB client
            self.mock_client = Mock()
            mock_client_class.return_value = self.mock_client
            
            # Mock collections
            self.mock_catalog = Mock()
            self.mock_content = Mock()
            self.mock_client.get_or_create_collection.side_effect = [
                self.mock_catalog, self.mock_content
            ]
            
            self.vector_store = VectorStore(
                chroma_path=self.temp_dir,
                embedding_model="test-model",
                max_results=5
            )
            
            yield  # This allows the test to run
            
        # Cleanup
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test VectorStore initialization"""
        assert self.vector_store.max_results == 5
        assert self.mock_client.get_or_create_collection.call_count == 2
    
    def test_search_successful_no_filters(self):
        """Test successful search without filters"""
        mock_chroma_results = {
            'documents': [['Content about ML']],
            'metadatas': [[{'course_title': 'ML Course', 'lesson_number': 1}]],
            'distances': [[0.1]]
        }
        self.mock_content.query.return_value = mock_chroma_results
        
        results = self.vector_store.search("machine learning")
        
        assert not results.error
        assert results.documents == ['Content about ML']
        assert results.metadata == [{'course_title': 'ML Course', 'lesson_number': 1}]
        
        self.mock_content.query.assert_called_once_with(
            query_texts=["machine learning"],
            n_results=5,
            where=None
        )
    
    def test_search_with_course_name_resolved(self):
        """Test search with course name that gets resolved"""
        # Mock course name resolution
        self.mock_catalog.query.return_value = {
            'documents': [['Python Basics Course']],
            'metadatas': [[{'title': 'Python Fundamentals'}]]
        }
        
        # Mock content search
        mock_chroma_results = {
            'documents': [['Python variables info']],
            'metadatas': [[{'course_title': 'Python Fundamentals', 'lesson_number': 2}]],
            'distances': [[0.2]]
        }
        self.mock_content.query.return_value = mock_chroma_results
        
        results = self.vector_store.search("variables", course_name="Python")
        
        assert not results.error
        assert results.documents == ['Python variables info']
        
        # Verify course resolution was called
        self.mock_catalog.query.assert_called_once_with(
            query_texts=["Python"],
            n_results=1
        )
        
        # Verify content search used resolved course name
        self.mock_content.query.assert_called_once_with(
            query_texts=["variables"],
            n_results=5,
            where={"course_title": "Python Fundamentals"}
        )
    
    def test_search_course_name_not_found(self):
        """Test search when course name cannot be resolved"""
        # Mock failed course resolution
        self.mock_catalog.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }
        
        results = self.vector_store.search("variables", course_name="Nonexistent Course")
        
        assert results.error == "No course found matching 'Nonexistent Course'"
        assert results.is_empty()
    
    def test_search_with_lesson_number(self):
        """Test search with lesson number filter"""
        mock_chroma_results = {
            'documents': [['Lesson 3 content']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 3}]],
            'distances': [[0.15]]
        }
        self.mock_content.query.return_value = mock_chroma_results
        
        results = self.vector_store.search("concepts", lesson_number=3)
        
        assert not results.error
        assert results.documents == ['Lesson 3 content']
        
        self.mock_content.query.assert_called_once_with(
            query_texts=["concepts"],
            n_results=5,
            where={"lesson_number": 3}
        )
    
    def test_search_with_both_filters(self):
        """Test search with both course name and lesson number"""
        # Mock successful course resolution
        self.mock_catalog.query.return_value = {
            'documents': [['Course content']],
            'metadatas': [[{'title': 'Resolved Course'}]]
        }
        
        mock_chroma_results = {
            'documents': [['Specific lesson content']],
            'metadatas': [[{'course_title': 'Resolved Course', 'lesson_number': 2}]],
            'distances': [[0.1]]
        }
        self.mock_content.query.return_value = mock_chroma_results
        
        results = self.vector_store.search("topic", course_name="Test", lesson_number=2)
        
        assert not results.error
        
        self.mock_content.query.assert_called_once_with(
            query_texts=["topic"],
            n_results=5,
            where={"$and": [
                {"course_title": "Resolved Course"},
                {"lesson_number": 2}
            ]}
        )
    
    def test_search_chromadb_exception(self):
        """Test search when ChromaDB raises exception"""
        self.mock_content.query.side_effect = Exception("ChromaDB connection error")
        
        results = self.vector_store.search("test query")
        
        assert "Search error: ChromaDB connection error" in results.error
        assert results.is_empty()
    
    def test_search_custom_limit(self):
        """Test search with custom result limit"""
        mock_chroma_results = {
            'documents': [['Result 1', 'Result 2']],
            'metadatas': [[{}, {}]],
            'distances': [[0.1, 0.2]]
        }
        self.mock_content.query.return_value = mock_chroma_results
        
        results = self.vector_store.search("query", limit=2)
        
        self.mock_content.query.assert_called_once_with(
            query_texts=["query"],
            n_results=2,
            where=None
        )
    
    def test_resolve_course_name_successful(self):
        """Test successful course name resolution"""
        self.mock_catalog.query.return_value = {
            'documents': [['Course content']],
            'metadatas': [[{'title': 'Full Course Title'}]]
        }
        
        result = self.vector_store._resolve_course_name("partial")
        
        assert result == "Full Course Title"
        self.mock_catalog.query.assert_called_once_with(
            query_texts=["partial"],
            n_results=1
        )
    
    def test_resolve_course_name_no_match(self):
        """Test course name resolution with no matches"""
        self.mock_catalog.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }
        
        result = self.vector_store._resolve_course_name("nonexistent")
        
        assert result is None
    
    def test_resolve_course_name_exception(self):
        """Test course name resolution with exception"""
        self.mock_catalog.query.side_effect = Exception("Database error")
        
        with patch('builtins.print') as mock_print:
            result = self.vector_store._resolve_course_name("test")
            
            assert result is None
            mock_print.assert_called_once()
    
    def test_build_filter_no_filters(self):
        """Test filter building with no parameters"""
        filter_dict = self.vector_store._build_filter(None, None)
        assert filter_dict is None
    
    def test_build_filter_course_only(self):
        """Test filter building with course only"""
        filter_dict = self.vector_store._build_filter("Test Course", None)
        assert filter_dict == {"course_title": "Test Course"}
    
    def test_build_filter_lesson_only(self):
        """Test filter building with lesson only"""
        filter_dict = self.vector_store._build_filter(None, 5)
        assert filter_dict == {"lesson_number": 5}
    
    def test_build_filter_both(self):
        """Test filter building with both parameters"""
        filter_dict = self.vector_store._build_filter("Test Course", 3)
        expected = {
            "$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 3}
            ]
        }
        assert filter_dict == expected
    
    def test_add_course_metadata(self):
        """Test adding course metadata to catalog"""
        lessons = [
            Lesson(lesson_number=1, title="Intro", lesson_link="http://lesson1.com"),
            Lesson(lesson_number=2, title="Advanced", lesson_link="http://lesson2.com")
        ]
        course = Course(
            title="Test Course",
            course_link="http://course.com",
            instructor="John Doe",
            lessons=lessons
        )
        
        self.vector_store.add_course_metadata(course)
        
        self.mock_catalog.add.assert_called_once()
        call_args = self.mock_catalog.add.call_args
        
        assert call_args[1]["documents"] == ["Test Course"]
        assert call_args[1]["ids"] == ["Test Course"]
        
        metadata = call_args[1]["metadatas"][0]
        assert metadata["title"] == "Test Course"
        assert metadata["instructor"] == "John Doe"
        assert metadata["course_link"] == "http://course.com"
        assert metadata["lesson_count"] == 2
        assert "lessons_json" in metadata
    
    def test_add_course_content(self):
        """Test adding course content chunks"""
        chunks = [
            CourseChunk(
                content="Content 1",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="Content 2", 
                course_title="Test Course",
                lesson_number=2,
                chunk_index=1
            )
        ]
        
        self.vector_store.add_course_content(chunks)
        
        self.mock_content.add.assert_called_once()
        call_args = self.mock_content.add.call_args
        
        assert call_args[1]["documents"] == ["Content 1", "Content 2"]
        expected_metadata = [
            {"course_title": "Test Course", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "Test Course", "lesson_number": 2, "chunk_index": 1}
        ]
        assert call_args[1]["metadatas"] == expected_metadata
        assert call_args[1]["ids"] == ["Test_Course_0", "Test_Course_1"]
    
    def test_add_course_content_empty(self):
        """Test adding empty course content list"""
        self.vector_store.add_course_content([])
        
        self.mock_content.add.assert_not_called()
    
    def test_clear_all_data(self):
        """Test clearing all data from collections"""
        self.vector_store.clear_all_data()
        
        assert self.mock_client.delete_collection.call_count == 2
        self.mock_client.delete_collection.assert_any_call("course_catalog")
        self.mock_client.delete_collection.assert_any_call("course_content")
        
        # Should recreate collections
        assert self.mock_client.get_or_create_collection.call_count == 4  # 2 initial + 2 recreated
    
    def test_clear_all_data_exception(self):
        """Test clear_all_data handles exceptions"""
        self.mock_client.delete_collection.side_effect = Exception("Delete error")
        
        with patch('builtins.print') as mock_print:
            self.vector_store.clear_all_data()
            mock_print.assert_called_once()
    
    def test_get_existing_course_titles(self):
        """Test getting existing course titles"""
        self.mock_catalog.get.return_value = {
            'ids': ['Course A', 'Course B', 'Course C']
        }
        
        titles = self.vector_store.get_existing_course_titles()
        
        assert titles == ['Course A', 'Course B', 'Course C']
        self.mock_catalog.get.assert_called_once()
    
    def test_get_existing_course_titles_empty(self):
        """Test getting course titles when none exist"""
        self.mock_catalog.get.return_value = {'ids': []}
        
        titles = self.vector_store.get_existing_course_titles()
        
        assert titles == []
    
    def test_get_existing_course_titles_exception(self):
        """Test get_existing_course_titles handles exceptions"""
        self.mock_catalog.get.side_effect = Exception("Get error")
        
        with patch('builtins.print') as mock_print:
            titles = self.vector_store.get_existing_course_titles()
            
            assert titles == []
            mock_print.assert_called_once()
    
    def test_get_course_count(self):
        """Test getting course count"""
        self.mock_catalog.get.return_value = {'ids': ['Course A', 'Course B']}
        
        count = self.vector_store.get_course_count()
        
        assert count == 2
    
    def test_get_course_count_empty(self):
        """Test getting course count when no courses"""
        self.mock_catalog.get.return_value = {'ids': []}
        
        count = self.vector_store.get_course_count()
        
        assert count == 0
    
    def test_get_all_courses_metadata(self):
        """Test getting all courses metadata with JSON parsing"""
        mock_metadata = [
            {
                'title': 'Course A',
                'instructor': 'Teacher 1',
                'lessons_json': '[{"lesson_number": 1, "lesson_title": "Intro"}]'
            }
        ]
        self.mock_catalog.get.return_value = {'metadatas': mock_metadata}
        
        metadata = self.vector_store.get_all_courses_metadata()
        
        assert len(metadata) == 1
        assert metadata[0]['title'] == 'Course A'
        assert metadata[0]['lessons'] == [{"lesson_number": 1, "lesson_title": "Intro"}]
        assert 'lessons_json' not in metadata[0]  # Should be removed after parsing
    
    def test_get_course_link(self):
        """Test getting course link by title"""
        self.mock_catalog.get.return_value = {
            'metadatas': [{'course_link': 'http://test.com'}]
        }
        
        link = self.vector_store.get_course_link("Test Course")
        
        assert link == "http://test.com"
        self.mock_catalog.get.assert_called_once_with(ids=["Test Course"])
    
    def test_get_course_link_not_found(self):
        """Test getting course link when course not found"""
        self.mock_catalog.get.return_value = {'metadatas': []}
        
        link = self.vector_store.get_course_link("Nonexistent")
        
        assert link is None


if __name__ == "__main__":
    pytest.main([__file__])