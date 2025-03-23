#!/usr/bin/env python3
"""
Interactive testing script for Kitchen Assistive Robot AI agents.
This script allows you to have a conversation with different agent types.
"""

import os
import sys
import argparse
import logging
from typing import Optional, List, Dict

from base_agent import BaseAgent, KitchenAssistantAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_screen():
    """Clear the terminal screen based on the OS."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(agent_type: str):
    """Print a header for the interactive session."""
    clear_screen()
    print("=" * 80)
    print(f"Kitchen Assistive Robot - Interactive {agent_type} Session")
    print("=" * 80)
    print("Type 'exit', 'quit', or 'q' to end the conversation.")
    print("Type 'clear' to clear the conversation history.")
    print("Type 'verbose on/off' to toggle verbose mode.")
    print("Type 'help' to see these commands again.")
    print("=" * 80)
    print()

def create_agent(agent_type: str, verbose: bool = True, model: Optional[str] = None) -> BaseAgent:
    """
    Create an agent of the specified type.
    
    Args:
        agent_type: Type of agent to create ('base' or 'kitchen')
        verbose: Whether to enable verbose logging
        model: Optional model name to use
        
    Returns:
        An instance of the specified agent type
    """
    if agent_type.lower() == 'kitchen':
        return KitchenAssistantAgent(model_name=model, verbose=verbose)
    else:  # Default to base agent
        return BaseAgent(model_name=model, verbose=verbose)

def interactive_session(agent_type: str = 'base', verbose: bool = True, model: Optional[str] = None):
    """
    Start an interactive session with the specified agent type.
    
    Args:
        agent_type: Type of agent to use ('base' or 'kitchen')
        verbose: Whether to enable verbose logging
        model: Optional model name to use
    """
    # Create the agent
    agent = create_agent(agent_type, verbose, model)
    
    # Print welcome header
    print_header(agent_type)
    
    # Initialize chat history
    chat_history = []
    
    # Start the conversation loop
    while True:
        # Get user input
        user_input = input("\n👤 You: ")
        
        # Check for exit commands
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\nThank you for using the Kitchen Assistive Robot AI. Goodbye!")
            break
            
        # Check for clear command
        elif user_input.lower() == 'clear':
            chat_history = []
            print_header(agent_type)
            continue
            
        # Check for help command
        elif user_input.lower() == 'help':
            print("\nCommands:")
            print("  exit, quit, q - End the conversation")
            print("  clear - Clear conversation history")
            print("  verbose on/off - Toggle verbose logging")
            print("  help - Show this help message")
            continue
            
        # Check for verbose toggle
        elif user_input.lower() in ['verbose on', 'verbose off']:
            agent.verbose = user_input.lower() == 'verbose on'
            print(f"\nVerbose mode {'enabled' if agent.verbose else 'disabled'}.")
            continue
            
        # Process the user input
        print("\n🤖 Agent: ", end="", flush=True)
        
        # Get agent response
        response = agent.process(user_input, chat_history)
        
        # Print the response
        output = response.get("output", "Sorry, I couldn't generate a response.")
        print(output)
        
        # Update chat history
        chat_history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": output}
        ])

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description='Interactive Kitchen Assistive Robot AI')
    parser.add_argument('--agent', '-a', type=str, default='base', choices=['base', 'kitchen'],
                        help='Agent type to use (base or kitchen)')
    parser.add_argument('--model', '-m', type=str, 
                        help='Model name to use (defaults to setting in .env file)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Disable verbose logging')
    
    args = parser.parse_args()
    
    try:
        interactive_session(
            agent_type=args.agent,
            verbose=not args.quiet,
            model=args.model
        )
    except KeyboardInterrupt:
        print("\nInteractive session terminated by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in interactive session: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 