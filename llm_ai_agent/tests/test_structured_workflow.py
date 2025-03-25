#!/usr/bin/env python3
"""
Test script for the structured workflow implementation.
This shows the desired workflow: User → Agent → Tool → System → Agent → User
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, Any

# Add the parent directory to the path to import the agents module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the agent creation function
from llm_ai_agent.agents import create_agent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def pretty_print_json(data: Dict[str, Any]) -> None:
    """
    Pretty print a JSON object.
    
    Args:
        data: The data to print
    """
    if not data:
        print("  [None]")
        return
        
    try:
        print(json.dumps(data, indent=2))
    except (TypeError, ValueError):
        print(f"  {data}")

def test_structured_workflow():
    """Test the structured workflow implementation."""
    print("\n" + "=" * 80)
    print("Testing Structured Workflow")
    print("=" * 80)
    
    # Create the structured agent
    agent = create_agent(
        agent_type="structured_agent", 
        verbose=True,
        use_hardware=False
    )
    agent.print_system_prompt()
    if not agent:
        logger.error("Failed to create agent")
        return
    
    print("\nAgent created successfully")
    
    # Example 1: Calculator tool
    print("\n" + "-" * 80)
    print("Example 1: Calculator Tool")
    print("-" * 80)
    
    user_input = "What's the result of 125 * 37?"
    print(f"\n👤 User: {user_input}")
    
    response = agent.process(user_input)
    
    # Print thought process
    print("\n💭 Agent Thought Process:")
    print(response.get("thought", "No thought process provided"))
    
    # Print initial tool call decision
    print("\n🔍 Agent Tool Decision:")
    tool_calls = response.get("tool_calls", [])
    if tool_calls:
        for i, tool_call in enumerate(tool_calls):
            print(f"\n  Tool Call #{i+1}:")
            tool_name = tool_call.get("tool_name", "Unknown")
            params = tool_call.get("parameters", {})
            print(f"  - Name: {tool_name}")
            print(f"  - Parameters:")
            for k, v in params.items():
                print(f"    - {k}: {v}")
    else:
        print("  No tools were called")
    
    # Print tool execution results
    print("\n🛠️ Tool Execution Results:")
    tool_results = response.get("tool_results", [])
    if tool_results:
        for i, result in enumerate(tool_results):
            print(f"\n  Result #{i+1}:")
            tool_name = result.get("tool_name", "Unknown")
            call_id = result.get("call_id", "")
            data = result.get("data", {})
            print(f"  - Tool: {tool_name}")
            print(f"  - Call ID: {call_id}")
            print(f"  - Data: {data}")
    else:
        print("  No tool results available")
    
    # Print final response
    print("\n🤖 Agent Final Response:")
    print(response.get("output", "No response available"))
    
    # Example 2: Text processor tool
    print("\n" + "-" * 80)
    print("Example 2: Text Processor Tool")
    print("-" * 80)
    
    user_input = "Can you analyze this text: 'The quick brown fox jumps over the lazy dog. This sentence contains all the letters in the English alphabet.'"
    print(f"\n👤 User: {user_input}")
    
    response = agent.process(user_input)
    
    # Print thought process
    print("\n💭 Agent Thought Process:")
    print(response.get("thought", "No thought process provided"))
    
    # Print initial tool call decision
    print("\n🔍 Agent Tool Decision:")
    tool_calls = response.get("tool_calls", [])
    if tool_calls:
        for i, tool_call in enumerate(tool_calls):
            print(f"\n  Tool Call #{i+1}:")
            tool_name = tool_call.get("tool_name", "Unknown")
            params = tool_call.get("parameters", {})
            print(f"  - Name: {tool_name}")
            print(f"  - Parameters:")
            for k, v in params.items():
                print(f"    - {k}: {v}")
    else:
        print("  No tools were called")
    
    # Print tool execution results
    print("\n🛠️ Tool Execution Results:")
    tool_results = response.get("tool_results", [])
    if tool_results:
        for i, result in enumerate(tool_results):
            print(f"\n  Result #{i+1}:")
            tool_name = result.get("tool_name", "Unknown")
            call_id = result.get("call_id", "")
            data = result.get("data", {})
            print(f"  - Tool: {tool_name}")
            print(f"  - Call ID: {call_id}")
            print(f"  - Data: {data}")
    else:
        print("  No tool results available")
    
    # Print final response
    print("\n🤖 Agent Final Response:")
    print(response.get("output", "No response available"))
    
    # Example 3: No tools needed
    print("\n" + "-" * 80)
    print("Example 3: No Tools Needed")
    print("-" * 80)
    
    user_input = "What's the capital of France?"
    print(f"\n👤 User: {user_input}")
    
    response = agent.process(user_input)
    
    # Print thought process
    print("\n💭 Agent Thought Process:")
    print(response.get("thought", "No thought process provided"))
    
    # Print final response
    print("\n🤖 Agent Final Response:")
    print(response.get("output", "No response available"))
    
    print("\n" + "=" * 80)
    print("Structured Workflow Test Complete")
    print("=" * 80)

def test_hardware_tools():
    """Test the hardware tools structured workflow."""
    print("\n" + "=" * 80)
    print("Testing Hardware Tools Workflow")
    print("=" * 80)
    
    # Create the structured agent with hardware
    agent = create_agent(
        agent_type="structured_agent", 
        verbose=True,
        use_hardware=True  # Enable hardware (will use mock implementations if real hardware is not available)
    )
    
    if not agent:
        logger.error("Failed to create agent")
        return
    
    print("\nAgent with hardware support created successfully")
    
    # Example 1: Speak Tool
    print("\n" + "-" * 80)
    print("Example 1: Speak Tool")
    print("-" * 80)
    
    user_input = "Can you say hello to the user?"
    print(f"\n👤 User: {user_input}")
    
    response = agent.process(user_input)
    
    # Print thought process
    print("\n💭 Agent Thought Process:")
    print(response.get("thought", "No thought process provided"))
    
    # Print initial tool call decision
    print("\n🔍 Agent Tool Decision:")
    tool_calls = response.get("tool_calls", [])
    if tool_calls:
        for i, tool_call in enumerate(tool_calls):
            print(f"\n  Tool Call #{i+1}:")
            tool_name = tool_call.get("tool_name", "Unknown")
            params = tool_call.get("parameters", {})
            print(f"  - Name: {tool_name}")
            print(f"  - Parameters:")
            for k, v in params.items():
                print(f"    - {k}: {v}")
    else:
        print("  No tools were called")
    
    # Print tool execution results
    print("\n🛠️ Tool Execution Results:")
    tool_results = response.get("tool_results", [])
    if tool_results:
        for i, result in enumerate(tool_results):
            print(f"\n  Result #{i+1}:")
            tool_name = result.get("tool_name", "Unknown")
            call_id = result.get("call_id", "")
            data = result.get("data", {})
            print(f"  - Tool: {tool_name}")
            print(f"  - Call ID: {call_id}")
            print(f"  - Data: {data}")
    else:
        print("  No tool results available")
    
    # Print final response
    print("\n🤖 Agent Final Response:")
    print(response.get("output", "No response available"))
    
    # Example 2: Camera Tool
    print("\n" + "-" * 80)
    print("Example 2: Camera Tool")
    print("-" * 80)
    
    user_input = "Take a picture with both cameras and describe what you see."
    print(f"\n👤 User: {user_input}")
    
    response = agent.process(user_input)
    
    # Print thought process
    print("\n💭 Agent Thought Process:")
    print(response.get("thought", "No thought process provided"))
    
    # Print initial tool call decision
    print("\n🔍 Agent Tool Decision:")
    tool_calls = response.get("tool_calls", [])
    if tool_calls:
        for i, tool_call in enumerate(tool_calls):
            print(f"\n  Tool Call #{i+1}:")
            tool_name = tool_call.get("tool_name", "Unknown")
            params = tool_call.get("parameters", {})
            print(f"  - Name: {tool_name}")
            print(f"  - Parameters:")
            for k, v in params.items():
                print(f"    - {k}: {v}")
    else:
        print("  No tools were called")
    
    # Print tool execution results
    print("\n🛠️ Tool Execution Results:")
    tool_results = response.get("tool_results", [])
    if tool_results:
        for i, result in enumerate(tool_results):
            print(f"\n  Result #{i+1}:")
            tool_name = result.get("tool_name", "Unknown")
            call_id = result.get("call_id", "")
            data = result.get("data", {})
            print(f"  - Tool: {tool_name}")
            print(f"  - Call ID: {call_id}")
            print(f"  - Data: {data}")
    else:
        print("  No tool results available")
    
    # Print final response
    print("\n🤖 Agent Final Response:")
    print(response.get("output", "No response available"))
    
    # Example 3: Robot Arm Tool
    print("\n" + "-" * 80)
    print("Example 3: Robot Arm Tool")
    print("-" * 80)
    
    user_input = "Move the robotic arm to grab an object at position x=250, y=300, z=200."
    print(f"\n👤 User: {user_input}")
    
    response = agent.process(user_input)
    
    # Print thought process
    print("\n💭 Agent Thought Process:")
    print(response.get("thought", "No thought process provided"))
    
    # Print initial tool call decision
    print("\n🔍 Agent Tool Decision:")
    tool_calls = response.get("tool_calls", [])
    if tool_calls:
        for i, tool_call in enumerate(tool_calls):
            print(f"\n  Tool Call #{i+1}:")
            tool_name = tool_call.get("tool_name", "Unknown")
            params = tool_call.get("parameters", {})
            print(f"  - Name: {tool_name}")
            print(f"  - Parameters:")
            for k, v in params.items():
                print(f"    - {k}: {v}")
    else:
        print("  No tools were called")
    
    # Print tool execution results
    print("\n🛠️ Tool Execution Results:")
    tool_results = response.get("tool_results", [])
    if tool_results:
        for i, result in enumerate(tool_results):
            print(f"\n  Result #{i+1}:")
            tool_name = result.get("tool_name", "Unknown")
            call_id = result.get("call_id", "")
            data = result.get("data", {})
            print(f"  - Tool: {tool_name}")
            print(f"  - Call ID: {call_id}")
            print(f"  - Data: {data}")
    else:
        print("  No tool results available")
    
    # Print final response
    print("\n🤖 Agent Final Response:")
    print(response.get("output", "No response available"))
    
    print("\n" + "=" * 80)
    print("Hardware Tools Workflow Test Complete")
    print("=" * 80)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the structured workflow implementation")
    parser.add_argument("--hardware", action="store_true", help="Enable hardware tool testing")
    parser.add_argument("--basic", action="store_true", help="Enable basic tool testing")
    args = parser.parse_args()
    
    # Run the requested tests or all tests if none specified
    if args.basic or not (args.hardware):
        test_structured_workflow()
    
    if args.hardware:
        test_hardware_tools() 