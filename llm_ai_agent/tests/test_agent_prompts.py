#!/usr/bin/env python3
"""
Test script to initialize all available agent types and print their system prompts.
This is useful for examining and comparing the prompts used by different agent configurations.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, Any, List

# Add the parent directory to the path to import the agents module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the agent module functions
from llm_ai_agent.agents import create_agent, get_available_agent_types, get_agent_info

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_separator(agent_type: str, with_line: bool = True) -> None:
    """
    Print a separator line with the agent type name.
    
    Args:
        agent_type: The name of the agent type
        with_line: Whether to include a line of equal signs
    """
    if with_line:
        print("\n" + "=" * 80)
    
    print(f"AGENT TYPE: {agent_type}")
    
    if with_line:
        print("=" * 80)

def test_all_agent_prompts(use_hardware: bool = False) -> None:
    """
    Initialize all available agent types and print their system prompts.
    
    Args:
        use_hardware: Whether to initialize agents with hardware support
    """
    print("\n" + "=" * 80)
    print("TESTING ALL AGENT PROMPTS")
    print("=" * 80)
    
    # Get all available agent types
    agent_types = get_available_agent_types()
    
    print(f"\nFound {len(agent_types)} agent types: {', '.join(agent_types)}")
    
    # Initialize each agent type and print its system prompt
    for i, agent_type in enumerate(agent_types):
        try:
            print_separator(agent_type)
            
            # Print agent info
            info = get_agent_info(agent_type)
            print(f"\nDescription: {info.get('description', 'No description available')}")
            print(f"Version: {info.get('version', '1.0.0')}")
            print(f"Hardware Required: {info.get('hardware_required', False)}")
            tools = info.get('tools', [])
            print(f"Tools: {', '.join(tools) if tools else 'None'}")
            
            # Create the agent
            agent = create_agent(
                agent_type=agent_type,
                verbose=False,
                use_hardware=use_hardware
            )
            
            if not agent:
                logger.error(f"Failed to create agent of type '{agent_type}'")
                continue
                
            # Print the system prompt
            print("\nSYSTEM PROMPT:")
            print("-" * 40)
            agent.print_system_prompt()
            print("-" * 40)
            
            print(f"\nAgent '{agent_type}' initialized successfully.")
            
            # Print a separator if not the last agent
            if i < len(agent_types) - 1:
                print("\n" + "-" * 80)
            
        except Exception as e:
            logger.error(f"Error initializing agent of type '{agent_type}': {e}")
            print(f"\nFailed to initialize agent '{agent_type}': {str(e)}")
    
    print("\n" + "=" * 80)
    print("AGENT PROMPT TESTING COMPLETE")
    print("=" * 80)

def test_single_agent_prompt(agent_type: str, use_hardware: bool = False) -> None:
    """
    Initialize a single agent type and print its system prompt.
    
    Args:
        agent_type: The name of the agent type to test
        use_hardware: Whether to initialize the agent with hardware support
    """
    print_separator(agent_type)
    
    try:
        # Print agent info
        info = get_agent_info(agent_type)
        
        if "error" in info:
            print(f"\nError: {info['error']}")
            print(f"\nAvailable agent types: {', '.join(get_available_agent_types())}")
            return
            
        print(f"\nDescription: {info.get('description', 'No description available')}")
        print(f"Version: {info.get('version', '1.0.0')}")
        print(f"Hardware Required: {info.get('hardware_required', False)}")
        tools = info.get('tools', [])
        print(f"Tools: {', '.join(tools) if tools else 'None'}")
        
        # Create the agent
        agent = create_agent(
            agent_type=agent_type,
            verbose=False,
            use_hardware=use_hardware
        )
        
        if not agent:
            logger.error(f"Failed to create agent of type '{agent_type}'")
            return
            
        # Print the system prompt
        print("\nSYSTEM PROMPT:")
        print("-" * 40)
        agent.print_system_prompt()
        print("-" * 40)
        
        print(f"\nAgent '{agent_type}' initialized successfully.")
        
    except Exception as e:
        logger.error(f"Error initializing agent of type '{agent_type}': {e}")
        print(f"\nFailed to initialize agent '{agent_type}': {str(e)}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test agent system prompts")
    parser.add_argument("--agent", type=str, help="Test a specific agent type")
    parser.add_argument("--hardware", action="store_true", help="Initialize agents with hardware support")
    parser.add_argument("--list", action="store_true", help="List available agent types")
    args = parser.parse_args()
    
    # If --list flag is provided, just list the available agent types
    if args.list:
        agent_types = get_available_agent_types()
        print(f"Available agent types: {', '.join(agent_types)}")
        for agent_type in agent_types:
            info = get_agent_info(agent_type)
            print(f"\n- {agent_type}: {info.get('description', 'No description')}")
        sys.exit(0)
    
    # If an agent type is specified, test just that one
    if args.agent:
        test_single_agent_prompt(args.agent, args.hardware)
    else:
        # Otherwise test all available agents
        test_all_agent_prompts(args.hardware) 