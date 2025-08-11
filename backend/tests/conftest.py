import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


@pytest.fixture
def test_config():
    """Create test configuration with temporary paths"""
    config = Config()
    config.CHROMA_PATH = tempfile.mkdtemp()
    config.ANTHROPIC_API_KEY = "test-api-key"
    yield config
    # Cleanup
    shutil.rmtree(config.CHROMA_PATH, ignore_errors=True)


@pytest.fixture
def mock_rag_system(test_config):
    """Create a mocked RAG system for testing"""
    with patch('vector_store.chromadb.PersistentClient'), \
         patch('ai_generator.anthropic.Anthropic'):
        rag_system = RAGSystem(test_config)
        
        # Mock the main methods using patch to ensure proper mocking
        rag_system.query = Mock(return_value=("Default response", []))
        rag_system.get_course_analytics = Mock(return_value={
            "total_courses": 0, 
            "course_titles": []
        })
        
        # Mock vector store methods
        rag_system.vector_store.search = Mock()
        rag_system.vector_store.get_course_count = Mock(return_value=0)
        rag_system.vector_store.get_existing_course_titles = Mock(return_value=[])
        
        # Mock session manager methods
        rag_system.session_manager.create_session = Mock(return_value="default_session")
        rag_system.session_manager.clear_session = Mock()
        
        # Mock AI generator with default response
        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Test AI response")]
        mock_response.stop_reason = "end_turn"
        rag_system.ai_generator.client.messages.create.return_value = mock_response
        
        yield rag_system


@pytest.fixture
def test_app(mock_rag_system):
    """Create test FastAPI app with mocked dependencies"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Create a test app without static file mounting
    app = FastAPI(title="Test RAG System")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Define request/response models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None
    
    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str
    
    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # Define test endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        from fastapi import HTTPException
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            answer, sources = mock_rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        from fastapi import HTTPException
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        from fastapi import HTTPException
        try:
            mock_rag_system.session_manager.clear_session(session_id)
            return {"status": "success", "message": f"Session {session_id} cleared"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {"message": "RAG System API", "status": "running"}
    
    yield app


@pytest.fixture
def test_client(test_app):
    """Create test client for API testing"""
    return TestClient(test_app)


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return SearchResults(
        documents=["Machine learning is a subset of artificial intelligence"],
        metadata=[{"course_title": "AI Fundamentals", "lesson_number": 1}],
        distances=[0.1]
    )


@pytest.fixture
def sample_course():
    """Sample course data for testing"""
    return Course(
        title="Test Course",
        instructor="Test Instructor",
        lessons=[
            Lesson(lesson_number=1, title="Introduction"),
            Lesson(lesson_number=2, title="Advanced Topics")
        ]
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="Introduction to the course",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Advanced topic content",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=0
        )
    ]


@pytest.fixture
def mock_ai_response():
    """Mock AI response for testing"""
    response = Mock()
    response.content = [Mock(type="text", text="This is a test AI response")]
    response.stop_reason = "end_turn"
    return response


@pytest.fixture
def mock_tool_response():
    """Mock AI response with tool use"""
    response = Mock()
    response.content = [Mock(
        type="tool_use",
        name="search_course_content",
        input={"query": "test query"},
        id="tool_123"
    )]
    response.stop_reason = "tool_use"
    return response


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress common warnings during testing"""
    import warnings
    warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture
def temp_docs_folder():
    """Create temporary documents folder for testing"""
    temp_dir = tempfile.mkdtemp()
    
    # Create sample PDF files
    sample_files = ["course1.pdf", "course2.pdf", "README.txt"]
    for filename in sample_files:
        with open(os.path.join(temp_dir, filename), 'w') as f:
            f.write(f"Sample content for {filename}")
    
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_session():
    """Mock session data for testing"""
    return {
        "session_id": "test_session_123",
        "history": [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
    }