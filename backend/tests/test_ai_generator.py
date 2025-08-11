import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from dataclasses import dataclass

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator


# Mock Anthropic response structures
@dataclass
class MockContentBlock:
    type: str
    text: str = None
    name: str = None
    input: dict = None
    id: str = None


@dataclass 
class MockResponse:
    content: list
    stop_reason: str


class TestAIGenerator:
    """Test suite for AIGenerator functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        with patch('ai_generator.anthropic.Anthropic'):
            self.ai_gen = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
    
    def test_init(self):
        """Test AIGenerator initialization"""
        assert self.ai_gen.model == "claude-sonnet-4-20250514"
        assert self.ai_gen.base_params["temperature"] == 0
        assert self.ai_gen.base_params["max_tokens"] == 800
    
    def test_generate_response_simple_text(self):
        """Test simple text response without tools"""
        mock_response = MockResponse(
            content=[MockContentBlock(type="text", text="This is a simple answer")],
            stop_reason="end_turn"
        )
        
        self.ai_gen.client.messages.create.return_value = mock_response
        
        result = self.ai_gen.generate_response("What is 2+2?")
        
        assert result == "This is a simple answer"
        self.ai_gen.client.messages.create.assert_called_once()
        
        # Check API call parameters
        call_args = self.ai_gen.client.messages.create.call_args[1]
        assert call_args["model"] == "claude-sonnet-4-20250514"
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert call_args["messages"] == [{"role": "user", "content": "What is 2+2?"}]
        assert "tools" not in call_args
    
    def test_generate_response_with_conversation_history(self):
        """Test response generation with conversation history"""
        mock_response = MockResponse(
            content=[MockContentBlock(type="text", text="Based on our previous discussion...")],
            stop_reason="end_turn"
        )
        
        self.ai_gen.client.messages.create.return_value = mock_response
        
        history = "User: Hello\nAssistant: Hi there!"
        result = self.ai_gen.generate_response("Continue", conversation_history=history)
        
        assert result == "Based on our previous discussion..."
        
        # Check that history was included in system prompt
        call_args = self.ai_gen.client.messages.create.call_args[1]
        assert history in call_args["system"]
    
    def test_generate_response_with_tools_no_tool_use(self):
        """Test response with tools available but not used"""
        mock_response = MockResponse(
            content=[MockContentBlock(type="text", text="I can answer without using tools")],
            stop_reason="end_turn"
        )
        
        self.ai_gen.client.messages.create.return_value = mock_response
        
        tools = [{"name": "search_tool", "description": "Search content"}]
        result = self.ai_gen.generate_response("General question", tools=tools)
        
        assert result == "I can answer without using tools"
        
        # Check that tools were provided in API call
        call_args = self.ai_gen.client.messages.create.call_args[1]
        assert call_args["tools"] == tools
        assert call_args["tool_choice"] == {"type": "auto"}
    
    def test_generate_response_with_tool_use(self):
        """Test response that uses a tool"""
        # Mock initial response with tool use
        initial_response = MockResponse(
            content=[MockContentBlock(
                type="tool_use",
                name="search_course_content",
                input={"query": "machine learning"},
                id="tool_call_123"
            )],
            stop_reason="tool_use"
        )
        
        # Mock final response after tool execution
        final_response = MockResponse(
            content=[MockContentBlock(type="text", text="Based on the search results, machine learning is...")],
            stop_reason="end_turn"
        )
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results: ML is a subset of AI"
        
        # Set up mock to return different responses for different calls
        self.ai_gen.client.messages.create.side_effect = [initial_response, final_response]
        
        tools = [{"name": "search_course_content", "description": "Search content"}]
        result = self.ai_gen.generate_response(
            "What is machine learning?", 
            tools=tools, 
            tool_manager=mock_tool_manager
        )
        
        assert result == "Based on the search results, machine learning is..."
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", 
            query="machine learning"
        )
        
        # Verify two API calls were made
        assert self.ai_gen.client.messages.create.call_count == 2
    
    def test_handle_tool_execution_single_tool(self):
        """Test _handle_tool_execution with single tool call"""
        initial_response = MockResponse(
            content=[MockContentBlock(
                type="tool_use",
                name="search_tool",
                input={"query": "test"},
                id="tool_123"
            )],
            stop_reason="tool_use"
        )
        
        final_response = MockResponse(
            content=[MockContentBlock(type="text", text="Tool result processed")],
            stop_reason="end_turn"
        )
        
        self.ai_gen.client.messages.create.return_value = final_response
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool executed successfully"
        
        base_params = {
            "messages": [{"role": "user", "content": "test query"}],
            "system": "System prompt"
        }
        
        result = self.ai_gen._handle_tool_execution(initial_response, base_params, mock_tool_manager)
        
        assert result == "Tool result processed"
        mock_tool_manager.execute_tool.assert_called_once_with("search_tool", query="test")
        
        # Check final API call structure
        call_args = self.ai_gen.client.messages.create.call_args[1]
        assert len(call_args["messages"]) == 3  # Original + assistant + tool result
        assert call_args["messages"][1]["role"] == "assistant"
        assert call_args["messages"][2]["role"] == "user"
    
    def test_handle_tool_execution_multiple_tools(self):
        """Test _handle_tool_execution with multiple tool calls"""
        initial_response = MockResponse(
            content=[
                MockContentBlock(
                    type="tool_use",
                    name="search_tool",
                    input={"query": "test1"},
                    id="tool_1"
                ),
                MockContentBlock(
                    type="tool_use", 
                    name="outline_tool",
                    input={"course_title": "test2"},
                    id="tool_2"
                )
            ],
            stop_reason="tool_use"
        )
        
        final_response = MockResponse(
            content=[MockContentBlock(type="text", text="Multiple tools processed")],
            stop_reason="end_turn"
        )
        
        self.ai_gen.client.messages.create.return_value = final_response
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        
        base_params = {
            "messages": [{"role": "user", "content": "test query"}],
            "system": "System prompt"
        }
        
        result = self.ai_gen._handle_tool_execution(initial_response, base_params, mock_tool_manager)
        
        assert result == "Multiple tools processed"
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Verify both tools were called with correct parameters
        calls = mock_tool_manager.execute_tool.call_args_list
        assert calls[0][0] == ("search_tool",)
        assert calls[0][1] == {"query": "test1"}
        assert calls[1][0] == ("outline_tool",)
        assert calls[1][1] == {"course_title": "test2"}
    
    def test_tool_execution_error_handling(self):
        """Test handling of tool execution errors"""
        initial_response = MockResponse(
            content=[MockContentBlock(
                type="tool_use",
                name="failing_tool",
                input={"query": "test"},
                id="tool_fail"
            )],
            stop_reason="tool_use"
        )
        
        final_response = MockResponse(
            content=[MockContentBlock(type="text", text="Error handled gracefully")],
            stop_reason="end_turn"
        )
        
        self.ai_gen.client.messages.create.return_value = final_response
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution failed: Database unavailable"
        
        base_params = {
            "messages": [{"role": "user", "content": "test query"}],
            "system": "System prompt"
        }
        
        result = self.ai_gen._handle_tool_execution(initial_response, base_params, mock_tool_manager)
        
        assert result == "Error handled gracefully"
        
        # Check that error message was passed to final API call
        call_args = self.ai_gen.client.messages.create.call_args[1]
        tool_result_message = call_args["messages"][2]["content"][0]
        assert "Tool execution failed: Database unavailable" in tool_result_message["content"]
    
    def test_anthropic_api_error(self):
        """Test handling of Anthropic API errors"""
        self.ai_gen.client.messages.create.side_effect = Exception("API rate limit exceeded")
        
        with pytest.raises(Exception) as exc_info:
            self.ai_gen.generate_response("test query")
        
        assert "API rate limit exceeded" in str(exc_info.value)
    
    def test_system_prompt_structure(self):
        """Test that system prompt is properly structured"""
        mock_response = MockResponse(
            content=[MockContentBlock(type="text", text="Response")],
            stop_reason="end_turn"
        )
        
        self.ai_gen.client.messages.create.return_value = mock_response
        
        self.ai_gen.generate_response("test query")
        
        call_args = self.ai_gen.client.messages.create.call_args[1]
        system_prompt = call_args["system"]
        
        # Check key elements of system prompt
        assert "search_course_content" in system_prompt
        assert "get_course_outline" in system_prompt
        assert "One tool call per query maximum" in system_prompt
        assert "Course-specific content questions" in system_prompt
    
    def test_api_parameters_consistency(self):
        """Test that API parameters are consistent across calls"""
        mock_response = MockResponse(
            content=[MockContentBlock(type="text", text="Response")],
            stop_reason="end_turn"
        )
        
        self.ai_gen.client.messages.create.return_value = mock_response
        
        # Test multiple calls
        self.ai_gen.generate_response("query1")
        self.ai_gen.generate_response("query2", conversation_history="history")
        
        # Check all calls used consistent base parameters
        calls = self.ai_gen.client.messages.create.call_args_list
        
        for call in calls:
            params = call[1]
            assert params["model"] == "claude-sonnet-4-20250514"
            assert params["temperature"] == 0
            assert params["max_tokens"] == 800
    
    def test_empty_tool_list(self):
        """Test behavior with empty tools list"""
        mock_response = MockResponse(
            content=[MockContentBlock(type="text", text="No tools available")],
            stop_reason="end_turn"
        )
        
        self.ai_gen.client.messages.create.return_value = mock_response
        
        result = self.ai_gen.generate_response("test query", tools=[])
        
        assert result == "No tools available"
        
        # Empty tools list should still be passed to API
        call_args = self.ai_gen.client.messages.create.call_args[1]
        assert call_args["tools"] == []


if __name__ == "__main__":
    pytest.main([__file__])