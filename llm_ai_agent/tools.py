#!/usr/bin/env python3
"""
Tools for the Kitchen Assistive Robot AI.
This module provides a collection of tools that can be used by agents.
"""

import re
import logging
from typing import Dict, Any, Optional, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculator_tool(input_str: str) -> str:
    """
    Calculate the result of a mathematical expression.
    
    Args:
        input_str: A mathematical expression as a string (e.g., "2 + 2", "3 * 4")
        
    Returns:
        The result of the calculation as a string
    """
    try:
        # Using Python's eval but only for mathematical operations
        # This is safe as long as we're careful about what we allow
        # Convert ^ to ** for exponentiation
        input_str = input_str.replace("^", "**")
        
        # Clean up the input by removing any non-mathematical characters
        allowed_chars = set("0123456789+-*/() .**")
        cleaned_input = ''.join(c for c in input_str if c in allowed_chars)
        
        # Evaluate the expression
        result = eval(cleaned_input)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

def text_processor(text: str) -> str:
    """
    Process text by counting words, characters, and performing other text operations.
    
    Args:
        text: The text to process
        
    Returns:
        Information about the text
    """
    try:
        # Count words
        words = text.split()
        word_count = len(words)
        
        # Count characters
        char_count = len(text)
        char_count_no_spaces = len(text.replace(" ", ""))
        
        # Count sentences (simple approximation)
        sentence_count = len(re.findall(r'[.!?]+', text)) or 1
        
        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / max(1, word_count)
        
        # Calculate average sentence length
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # Generate result
        result = (
            f"Text Analysis:\n"
            f"- Words: {word_count}\n"
            f"- Characters (with spaces): {char_count}\n"
            f"- Characters (without spaces): {char_count_no_spaces}\n"
            f"- Sentences: {sentence_count}\n"
            f"- Average word length: {avg_word_length:.2f} characters\n"
            f"- Average sentence length: {avg_sentence_length:.2f} words"
        )
        
        return result
    except Exception as e:
        return f"Error processing text: {str(e)}"

def echo_tool(text: str) -> str:
    """
    Simply echoes back the input text.
    
    Args:
        text: The text to echo
        
    Returns:
        The input text prefixed with "Echo: "
    """
    return f"Echo: {text}"

# Add more tools as needed

# Dictionary of all available tools
TOOLS = {
    "calculator": calculator_tool,
    "text_processor": text_processor,
    "echo": echo_tool,
}

def get_tool(name: str):
    """
    Get a tool by name.
    
    Args:
        name: Name of the tool
        
    Returns:
        Tool function or None if not found
    """
    return TOOLS.get(name)

def get_all_tools() -> Dict[str, callable]:
    """
    Get all available tools.
    
    Returns:
        Dictionary of all tools
    """
    return TOOLS.copy()

def get_tool_names() -> List[str]:
    """
    Get the names of all available tools.
    
    Returns:
        List of tool names
    """
    return list(TOOLS.keys())

# Example usage
if __name__ == "__main__":
    # Test the tools
    print("Calculator: ", calculator_tool("2 + 2 * 3"))
    print("Text Processor: ", text_processor("This is a test. This is only a test."))
    print("Echo: ", echo_tool("Hello, world!")) 