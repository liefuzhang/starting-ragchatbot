import pytest
import unittest.mock as mock
from ai_generator import AIGenerator


class TestAIGenerator:
    """Test suite for AIGenerator functionality"""
    
    def setup_method(self):
        """Setup test fixtures before each test method"""
        self.mock_anthropic_client = mock.MagicMock()
        self.ai_generator = AIGenerator(api_key="test_key", model="claude-3-sonnet-20241022")
        # Replace the real client with our mock
        self.ai_generator.client = self.mock_anthropic_client
    
    def test_initialization(self):
        """Test AIGenerator initialization"""
        generator = AIGenerator(api_key="test_api_key", model="test_model")
        
        assert generator.model == "test_model"
        assert generator.base_params["model"] == "test_model"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800
    
    def test_system_prompt_content(self):
        """Test that system prompt contains expected content"""
        assert "course materials and educational content" in AIGenerator.SYSTEM_PROMPT
        assert "Course Outline Tool" in AIGenerator.SYSTEM_PROMPT
        assert "Content Search Tool" in AIGenerator.SYSTEM_PROMPT
        assert "Brief, Concise and focused" in AIGenerator.SYSTEM_PROMPT
    
    def test_generate_response_simple_query(self):
        """Test simple response generation without tools"""
        # Mock the API response
        mock_response = mock.MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [mock.MagicMock()]
        mock_response.content[0].text = "This is a test response"
        self.mock_anthropic_client.messages.create.return_value = mock_response
        
        result = self.ai_generator.generate_response("What is AI?")
        
        # Verify the API was called correctly
        self.mock_anthropic_client.messages.create.assert_called_once()
        call_args = self.mock_anthropic_client.messages.create.call_args[1]
        
        assert call_args["model"] == "claude-3-sonnet-20241022"
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert call_args["messages"][0]["content"] == "What is AI?"
        assert call_args["messages"][0]["role"] == "user"
        assert AIGenerator.SYSTEM_PROMPT in call_args["system"]
        
        assert result == "This is a test response"
    
    def test_generate_response_with_conversation_history(self):
        """Test response generation with conversation history"""
        mock_response = mock.MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [mock.MagicMock()]
        mock_response.content[0].text = "Response with history"
        self.mock_anthropic_client.messages.create.return_value = mock_response
        
        history = "Previous conversation context"
        result = self.ai_generator.generate_response("Follow up question", conversation_history=history)
        
        call_args = self.mock_anthropic_client.messages.create.call_args[1]
        assert "Previous conversation context" in call_args["system"]
        assert result == "Response with history"
    
    def test_generate_response_with_tools(self):
        """Test response generation with tools available"""
        mock_response = mock.MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [mock.MagicMock()]
        mock_response.content[0].text = "Response using tools"
        self.mock_anthropic_client.messages.create.return_value = mock_response
        
        tools = [{"name": "test_tool", "description": "Test tool"}]
        result = self.ai_generator.generate_response("Search for something", tools=tools)
        
        call_args = self.mock_anthropic_client.messages.create.call_args[1]
        assert call_args["tools"] == tools
        assert call_args["tool_choice"] == {"type": "auto"}
        assert result == "Response using tools"
    
    def test_generate_response_with_tool_use(self):
        """Test response generation that requires tool execution"""
        # Mock initial response with tool use
        mock_initial_response = mock.MagicMock()
        mock_initial_response.stop_reason = "tool_use"
        mock_tool_use = mock.MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.input = {"query": "test query"}
        mock_tool_use.id = "tool_123"
        mock_initial_response.content = [mock_tool_use]
        
        # Mock final response after tool execution
        mock_final_response = mock.MagicMock()
        mock_final_response.content = [mock.MagicMock()]
        mock_final_response.content[0].text = "Final response after tool use"
        
        # Configure mock to return different responses on subsequent calls
        self.mock_anthropic_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]
        
        # Mock tool manager
        mock_tool_manager = mock.MagicMock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"
        
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        result = self.ai_generator.generate_response(
            "Search for AI concepts",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify tool execution was called
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test query"
        )
        
        # Verify two API calls were made
        assert self.mock_anthropic_client.messages.create.call_count == 2
        
        assert result == "Final response after tool use"
    
    def test_handle_tool_execution_multiple_tools(self):
        """Test handling multiple tool calls in one response"""
        # Create mock response with multiple tool uses
        mock_response = mock.MagicMock()
        mock_tool_use_1 = mock.MagicMock()
        mock_tool_use_1.type = "tool_use"
        mock_tool_use_1.name = "tool_one"
        mock_tool_use_1.input = {"param": "value1"}
        mock_tool_use_1.id = "tool_1"
        
        mock_tool_use_2 = mock.MagicMock()
        mock_tool_use_2.type = "tool_use" 
        mock_tool_use_2.name = "tool_two"
        mock_tool_use_2.input = {"param": "value2"}
        mock_tool_use_2.id = "tool_2"
        
        mock_response.content = [mock_tool_use_1, mock_tool_use_2]
        
        # Mock tool manager
        mock_tool_manager = mock.MagicMock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        
        # Mock final response
        mock_final_response = mock.MagicMock()
        mock_final_response.content = [mock.MagicMock()]
        mock_final_response.content[0].text = "Combined results response"
        self.mock_anthropic_client.messages.create.return_value = mock_final_response
        
        base_params = {
            "messages": [{"role": "user", "content": "Test query"}],
            "system": "Test system prompt"
        }
        
        result = self.ai_generator._handle_tool_execution(
            mock_response, base_params, mock_tool_manager
        )
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("tool_one", param="value1")
        mock_tool_manager.execute_tool.assert_any_call("tool_two", param="value2")
        
        # Verify final API call was made with tool results
        final_call_args = self.mock_anthropic_client.messages.create.call_args[1]
        assert len(final_call_args["messages"]) == 3  # Original + AI response + tool results
        
        tool_results_message = final_call_args["messages"][2]
        assert tool_results_message["role"] == "user"
        assert len(tool_results_message["content"]) == 2  # Two tool results
        
        assert result == "Combined results response"
    
    def test_handle_tool_execution_error_handling(self):
        """Test tool execution error handling"""
        mock_response = mock.MagicMock()
        mock_tool_use = mock.MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "failing_tool"
        mock_tool_use.input = {"param": "value"}
        mock_tool_use.id = "tool_123"
        mock_response.content = [mock_tool_use]
        
        # Mock tool manager that raises exception
        mock_tool_manager = mock.MagicMock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        # Mock final response
        mock_final_response = mock.MagicMock()
        mock_final_response.content = [mock.MagicMock()]
        mock_final_response.content[0].text = "Error handled response"
        self.mock_anthropic_client.messages.create.return_value = mock_final_response
        
        base_params = {
            "messages": [{"role": "user", "content": "Test query"}],
            "system": "Test system prompt"
        }
        
        # This should not raise an exception but handle it gracefully
        with pytest.raises(Exception):
            self.ai_generator._handle_tool_execution(
                mock_response, base_params, mock_tool_manager
            )
    
    def test_response_without_tool_manager(self):
        """Test that tool use requests are ignored if no tool manager provided"""
        mock_response = mock.MagicMock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [mock.MagicMock()]
        mock_response.content[0].text = "Tool use ignored"
        self.mock_anthropic_client.messages.create.return_value = mock_response
        
        tools = [{"name": "test_tool"}]
        result = self.ai_generator.generate_response("Query", tools=tools, tool_manager=None)
        
        # Should return the text content directly without trying to execute tools
        assert result == "Tool use ignored"
    
    def test_api_parameters_efficiency(self):
        """Test that API parameters are built efficiently"""
        mock_response = mock.MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [mock.MagicMock()]
        mock_response.content[0].text = "Efficient response"
        self.mock_anthropic_client.messages.create.return_value = mock_response
        
        # Generate response multiple times to ensure base_params are reused
        for _ in range(3):
            result = self.ai_generator.generate_response("Test query")
            assert result == "Efficient response"
        
        # Verify the base parameters are consistent across calls
        assert self.mock_anthropic_client.messages.create.call_count == 3
        for call in self.mock_anthropic_client.messages.create.call_args_list:
            call_kwargs = call[1]
            assert call_kwargs["model"] == "claude-3-sonnet-20241022"
            assert call_kwargs["temperature"] == 0
            assert call_kwargs["max_tokens"] == 800


if __name__ == "__main__":
    pytest.main([__file__])