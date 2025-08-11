import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test suite for CourseSearchTool.execute() method"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
    
    def test_execute_successful_search_no_filters(self):
        """Test successful search without course/lesson filters"""
        # Mock successful search results
        mock_results = SearchResults(
            documents=["Content about machine learning basics"],
            metadata=[{"course_title": "ML Course", "lesson_number": 1}],
            distances=[0.1]
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("machine learning")
        
        assert result == "[ML Course - Lesson 1]\nContent about machine learning basics"
        self.mock_vector_store.search.assert_called_once_with(
            query="machine learning",
            course_name=None,
            lesson_number=None
        )
    
    def test_execute_successful_search_with_course_filter(self):
        """Test successful search with course name filter"""
        mock_results = SearchResults(
            documents=["Python programming concepts"],
            metadata=[{"course_title": "Python Basics", "lesson_number": 2}],
            distances=[0.2]
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("variables", course_name="Python")
        
        assert result == "[Python Basics - Lesson 2]\nPython programming concepts"
        self.mock_vector_store.search.assert_called_once_with(
            query="variables",
            course_name="Python",
            lesson_number=None
        )
    
    def test_execute_successful_search_with_lesson_filter(self):
        """Test successful search with lesson number filter"""
        mock_results = SearchResults(
            documents=["Introduction to data structures"],
            metadata=[{"course_title": "Data Structures", "lesson_number": 1}],
            distances=[0.15]
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("arrays", lesson_number=1)
        
        assert result == "[Data Structures - Lesson 1]\nIntroduction to data structures"
        self.mock_vector_store.search.assert_called_once_with(
            query="arrays",
            course_name=None,
            lesson_number=1
        )
    
    def test_execute_multiple_results(self):
        """Test search returning multiple results"""
        mock_results = SearchResults(
            documents=[
                "First result about algorithms",
                "Second result about sorting"
            ],
            metadata=[
                {"course_title": "Algorithms", "lesson_number": 1},
                {"course_title": "Algorithms", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("sorting algorithms")
        
        expected = "[Algorithms - Lesson 1]\nFirst result about algorithms\n\n[Algorithms - Lesson 2]\nSecond result about sorting"
        assert result == expected
    
    def test_execute_empty_results_no_filters(self):
        """Test search returning no results without filters"""
        mock_results = SearchResults(documents=[], metadata=[], distances=[])
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("nonexistent topic")
        
        assert result == "No relevant content found."
    
    def test_execute_empty_results_with_course_filter(self):
        """Test search returning no results with course filter"""
        mock_results = SearchResults(documents=[], metadata=[], distances=[])
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("advanced topics", course_name="Basic Course")
        
        assert result == "No relevant content found in course 'Basic Course'."
    
    def test_execute_empty_results_with_lesson_filter(self):
        """Test search returning no results with lesson filter"""
        mock_results = SearchResults(documents=[], metadata=[], distances=[])
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("specific concept", lesson_number=5)
        
        assert result == "No relevant content found in lesson 5."
    
    def test_execute_empty_results_with_both_filters(self):
        """Test search returning no results with both filters"""
        mock_results = SearchResults(documents=[], metadata=[], distances=[])
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("topic", course_name="Math", lesson_number=3)
        
        assert result == "No relevant content found in course 'Math' in lesson 3."
    
    def test_execute_search_error(self):
        """Test search returning error from vector store"""
        mock_results = SearchResults(
            documents=[], 
            metadata=[], 
            distances=[], 
            error="ChromaDB connection failed"
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("any query")
        
        assert result == "ChromaDB connection failed"
    
    def test_execute_metadata_missing_fields(self):
        """Test handling of missing metadata fields"""
        mock_results = SearchResults(
            documents=["Some content"],
            metadata=[{}],  # Empty metadata
            distances=[0.1]
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test query")
        
        assert result == "[unknown]\nSome content"
    
    def test_execute_metadata_partial_fields(self):
        """Test handling of partial metadata fields"""
        mock_results = SearchResults(
            documents=["Content without lesson number"],
            metadata=[{"course_title": "Test Course"}],  # No lesson_number
            distances=[0.1]
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test query")
        
        assert result == "[Test Course]\nContent without lesson number"
    
    def test_sources_tracking(self):
        """Test that search results are tracked in last_sources"""
        mock_results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )
        self.mock_vector_store.search.return_value = mock_results
        
        self.search_tool.execute("test query")
        
        expected_sources = ["Course A - Lesson 1", "Course B - Lesson 2"]
        assert self.search_tool.last_sources == expected_sources
    
    def test_sources_tracking_no_lesson_number(self):
        """Test source tracking without lesson numbers"""
        mock_results = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Course A"}],
            distances=[0.1]
        )
        self.mock_vector_store.search.return_value = mock_results
        
        self.search_tool.execute("test query")
        
        assert self.search_tool.last_sources == ["Course A"]
    
    def test_get_tool_definition(self):
        """Test tool definition structure"""
        tool_def = self.search_tool.get_tool_definition()
        
        assert tool_def["name"] == "search_course_content"
        assert "description" in tool_def
        assert "input_schema" in tool_def
        assert tool_def["input_schema"]["required"] == ["query"]
        assert "query" in tool_def["input_schema"]["properties"]
        assert "course_name" in tool_def["input_schema"]["properties"]
        assert "lesson_number" in tool_def["input_schema"]["properties"]


class TestToolManager:
    """Test suite for ToolManager functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.tool_manager = ToolManager()
        self.mock_vector_store = Mock()
    
    def test_register_and_execute_tool(self):
        """Test registering and executing a tool"""
        search_tool = CourseSearchTool(self.mock_vector_store)
        self.tool_manager.register_tool(search_tool)
        
        # Mock successful search
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.tool_manager.execute_tool("search_course_content", query="test")
        
        assert "[Test Course - Lesson 1]\nTest content" in result
    
    def test_execute_nonexistent_tool(self):
        """Test executing a tool that doesn't exist"""
        result = self.tool_manager.execute_tool("nonexistent_tool", query="test")
        
        assert result == "Tool 'nonexistent_tool' not found"
    
    def test_get_tool_definitions(self):
        """Test getting all tool definitions"""
        search_tool = CourseSearchTool(self.mock_vector_store)
        self.tool_manager.register_tool(search_tool)
        
        definitions = self.tool_manager.get_tool_definitions()
        
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"
    
    def test_get_last_sources_empty(self):
        """Test getting sources when no searches have been performed"""
        sources = self.tool_manager.get_last_sources()
        assert sources == []
    
    def test_get_last_sources_with_data(self):
        """Test getting sources after a search"""
        search_tool = CourseSearchTool(self.mock_vector_store)
        search_tool.last_sources = ["Course A - Lesson 1"]
        self.tool_manager.register_tool(search_tool)
        
        sources = self.tool_manager.get_last_sources()
        assert sources == ["Course A - Lesson 1"]
    
    def test_reset_sources(self):
        """Test resetting sources from all tools"""
        search_tool = CourseSearchTool(self.mock_vector_store)
        search_tool.last_sources = ["Course A - Lesson 1"]
        self.tool_manager.register_tool(search_tool)
        
        self.tool_manager.reset_sources()
        
        assert search_tool.last_sources == []
        assert self.tool_manager.get_last_sources() == []


if __name__ == "__main__":
    pytest.main([__file__])