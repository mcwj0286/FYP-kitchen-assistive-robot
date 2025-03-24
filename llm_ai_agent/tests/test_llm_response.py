#!/usr/bin/env python3
"""
Tests for LLM response handling to diagnose the 'NoneType object is not iterable' error.
This script provides tests that reproduce and diagnose the error.
"""

import os
import sys
import unittest
import logging
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path to allow imports
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Import the necessary modules
from base_agent import BaseAgent

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestLLMResponseHandling(unittest.TestCase):
    """Test cases for LLM response handling to diagnose the NoneType error."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a base agent with mocked LLM and hardware disabled
        self.agent = BaseAgent(verbose=True, use_hardware=False)
        # Store reference to the original _parse_response method
        self.original_parse_response = self.agent._parse_response
        
    def test_parse_response_edge_cases(self):
        """Test _parse_response method with various edge cases."""
        test_cases = [
            # None response
            (None, {"response": "No response received from language model."}),
            
            # Empty response
            ("", {"response": ""}),
            
            # Regular text without JSON
            ("This is a regular text response without JSON.", {"response": "This is a regular text response without JSON."}),
            
            # Valid JSON with code block
            ("""```json
{
  "thought": "This is a thought",
  "response": "This is a response"
}
```""", {"thought": "This is a thought", "response": "This is a response"}),
            
            # Valid JSON without code block
            ('{"thought": "This is a thought", "response": "This is a response"}', 
             {"thought": "This is a thought", "response": "This is a response"}),
            
            # Invalid JSON
            ("```json\n{thought: This is a thought\n}\n```", {"response": "```json\n{thought: This is a thought\n}\n```"}),
            
            # Multiple JSON blocks (should get the first one)
            ("""```json
{
  "thought": "First thought",
  "response": "First response"
}
```
And some text in between.
```json
{
  "thought": "Second thought",
  "response": "Second response"
}
```""", {"thought": "First thought", "response": "First response"})
        ]
        
        for input_response, expected_output in test_cases:
            try:
                result = self.agent._parse_response(input_response)
                
                logger.info(f"Input: {input_response}")
                logger.info(f"Result: {result}")
                
                # For dict inputs, we check if all expected keys are present
                if isinstance(expected_output, dict):
                    for key, value in expected_output.items():
                        self.assertIn(key, result)
                        if key in result:
                            self.assertEqual(result[key], value)
                else:
                    self.assertEqual(result, expected_output)
                    
            except Exception as e:
                self.fail(f"Error processing case: {input_response}. Error: {e}")
                
    def test_process_method_with_none_response(self):
        """Test the process method when the LLM returns None."""
        # Mock the LLM to return None content
        mock_response = MagicMock()
        mock_response.content = None
        
        # Patch the llm.invoke method to return our mock response
        with patch.object(self.agent.llm, 'invoke', return_value=mock_response):
            result = self.agent.process("Test input")
            
            # Verify that we get an output, not an error
            self.assertIn("output", result)
            logger.info(f"Output with None response: {result['output']}")
            
    def test_full_process_flow(self):
        """Test the full process flow with mocked responses."""
        # Test case 1: LLM returns a response with a thought and action
        mock_tool_response = MagicMock()
        mock_tool_response.content = """```json
{
  "thought": "I should calculate this",
  "action": "calculator",
  "action_input": "5 + 5"
}
```"""
        
        # Test case 2: LLM returns a final response
        mock_final_response = MagicMock()
        mock_final_response.content = """```json
{
  "thought": "I've calculated the answer",
  "response": "The result is 10"
}
```"""
        
        # Patch the methods
        with patch.object(self.agent.llm, 'invoke', side_effect=[mock_tool_response, mock_final_response]), \
             patch.object(self.agent, '_execute_tool', return_value="10"):
            
            result = self.agent.process("What is 5 + 5?")
            
            # Verify result
            self.assertIn("output", result)
            self.assertEqual(result["output"], "The result is 10")
            logger.info(f"Full process result: {result}")
    
    def test_troubleshoot_iterable_error(self):
        """Specific test to troubleshoot the 'NoneType' object is not iterable error."""
        # Check if the error is in chat_history handling
        # This simulates the case where the error occurs
        with patch.object(self.agent.llm, 'invoke', side_effect=Exception("'NoneType' object is not iterable")):
            result = self.agent.process("Test input")
            
            # We should still get an output with the error
            self.assertIn("output", result)
            self.assertIn("Error", result["output"])
            self.assertIn("NoneType", result["output"])
            logger.info(f"Error handling result: {result['output']}")
            
    def test_handle_empty_messages(self):
        """Test handling of empty or invalid message structures."""
        # Simulate adding an empty or invalid message to chat history
        empty_history = [{}]  # Missing 'role' and 'content'
        invalid_history = [{"role": "user"}]  # Missing 'content'
        
        # Test with empty history
        try:
            result = self.agent.process("Test input", empty_history)
            logger.info(f"Result with empty history: {result}")
        except Exception as e:
            logger.error(f"Error with empty history: {e}")
            
        # Test with invalid history
        try:
            result = self.agent.process("Test input", invalid_history)
            logger.info(f"Result with invalid history: {result}")
        except Exception as e:
            logger.error(f"Error with invalid history: {e}")

if __name__ == "__main__":
    unittest.main() 