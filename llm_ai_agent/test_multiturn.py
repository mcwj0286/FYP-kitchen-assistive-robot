#!/usr/bin/env python3
"""
Test script for multi-turn tool calling in the ConfigurableAgent.
"""

import os
import sys
import logging
import json
from pathlib import Path

# Set up logging to see detailed information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path to import the module
sys.path.append(str(Path(__file__).parent.parent))

# Import the ConfigurableAgent
from llm_ai_agent.configurable_agent import ConfigurableAgent

def run_test(test_query, output_file="multiturn_test_results.txt"):
    """
    Run a test with the specified query and write results to the given output file.
    
    Args:
        test_query: The query to test
        output_file: File to write results to
    """
    # Create a ConfigurableAgent with a higher max_tool_iterations value for testing
    agent = ConfigurableAgent(
        agent_type="kitchen_assistant",
        verbose=True,
        max_tool_iterations=5  # Allow up to 5 iterations of tool calling
    )
    
    print(f"\n=================== TESTING MULTI-TURN TOOL CALLING ===================\n")
    print(f"Test Query: {test_query}\n")
    
    # Process the query with the agent
    response = agent.process(test_query)
    
    # Create output string
    output = []
    output.append("\n=================== RESPONSE DETAILS ===================\n")
    output.append(f"Iterations performed: {response.get('iterations', 0)}")
    output.append(f"Total tool calls: {len(response.get('tool_results', []))}")
    output.append(f"Is complete: {response.get('is_complete', False)}")
    
    output.append("\n=================== THOUGHT PROCESS ===================\n")
    output.append(response.get('thought', ''))
    
    output.append("\n=================== TOOL CALLS SEQUENCE ===================\n")
    for i, result in enumerate(response.get('tool_results', [])):
        output.append(f"Step {i+1}: {result.get('tool_name')}({json.dumps(result.get('parameters', {}), indent=2)})")
        
        # Show the full data
        data = result.get('data', '')
        if isinstance(data, dict):
            data = json.dumps(data, indent=2)
        
        output.append(f"Result: {data}\n")
    
    output.append("\n=================== FINAL RESPONSE ===================\n")
    output.append(response.get('output', ''))
    
    # Print the output
    output_text = "\n".join(output)
    print(output_text)
    
    # Also write to a file
    with open(output_file, "w") as f:
        f.write(f"Test Query: {test_query}\n\n")
        f.write(output_text)
    
    print(f"\nResults also written to {output_file}")
    
    return response

def main():
    """
    Test the multi-turn tool calling functionality.
    """
    # Test 1: Simple jar opening
    test1_query = "Can you help me open a jar? I want to know if you have a plan for this and where I should position the jar."
    run_test(test1_query, "multiturn_simple_test.txt")
    
    # Test 2: More complex query that might require more tool calls
    test2_query = "I need to prepare coffee and then open a jar of pickles. Can you help me with both tasks? What items do I need, where are they located, and what is your plan for assisting me?"
    run_test(test2_query, "multiturn_complex_test.txt")

if __name__ == "__main__":
    main() 