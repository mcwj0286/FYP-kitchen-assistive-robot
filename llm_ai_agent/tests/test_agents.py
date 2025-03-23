#!/usr/bin/env python3
"""
Test script for the BaseAgent and specialized agents.
This demonstrates the basic functionality of the OOP architecture.
"""

import os
import sys
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseAgent, KitchenAssistantAgent
from langchain.agents.tools import Tool

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_base_agent():
    """Test the basic functionality of BaseAgent."""
    print("\n=== Testing BaseAgent ===")
    
    # Create a base agent instance
    agent = BaseAgent(verbose=True)
    
    # Add a custom tool
    def hello_world(name):
        return f"Hello, {name}! Welcome to the agent test."
    
    agent.add_tool(
        Tool(
            name="hello",
            func=hello_world,
            description="Greet someone by name."
        )
    )
    
    # Test basic calculator functionality
    print("\nTesting calculator...")
    response = agent.process_to_string("What is 123 * 456?")
    print(f"Response: {response}")
    
    # Test the custom tool
    print("\nTesting custom hello tool...")
    response = agent.process_to_string("Say hello to John")
    print(f"Response: {response}")
    
    return agent

def test_kitchen_assistant():
    """Test the specialized KitchenAssistantAgent."""
    print("\n=== Testing KitchenAssistantAgent ===")
    
    # Create a kitchen assistant agent
    agent = KitchenAssistantAgent(verbose=True)
    
    # Test with a recipe-related query
    print("\nTesting kitchen-specific functionality...")
    response = agent.process_to_string(
        "I have 2 cups of flour, 1 cup sugar, 3 eggs, and 1/2 cup butter. What can I make?"
    )
    print(f"Response: {response}")
    
    # Test the echo tool (placeholder for hardware)
    print("\nTesting echo tool...")
    response = agent.process_to_string(
        "Can you echo 'Test kitchen hardware'"
    )
    print(f"Response: {response}")
    
    return agent

def main():
    """Run all tests."""
    print("Starting agent tests...")
    
    # Test BaseAgent
    base_agent = test_base_agent()
    
    # Test KitchenAssistantAgent
    kitchen_agent = test_kitchen_assistant()
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main() 