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

# Import from agents module instead of directly from configurable_agent
from llm_ai_agent.agents import create_agent, get_available_agent_types
from llm_ai_agent.configurable_agent import ConfigurableAgent

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

def create_session_agent(agent_type: str, verbose: bool = True, model: Optional[str] = None, 
                config_path: Optional[str] = None, use_hardware: bool = True) -> ConfigurableAgent:
    """
    Create an agent for the interactive session.
    
    Args:
        agent_type: Type of agent to use (specified in YAML configs)
        verbose: Whether to enable verbose logging
        model: Optional model name to use
        config_path: Optional path to a YAML configuration file
        use_hardware: Whether to use hardware or mock implementations
        
    Returns:
        An instance of the ConfigurableAgent
    """
    try:
        # Use the create_agent function from agents.py
        return create_agent(
            agent_type=agent_type,
            config_path=config_path,
            verbose=verbose,
            model_name=model,
            use_hardware=use_hardware
        )
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        # Check available agent types
        available_agents = get_available_agent_types()
        
        if available_agents:
            logger.info(f"Available agent types: {', '.join(available_agents)}")
        raise ValueError(f"Failed to create agent of type '{agent_type}': {str(e)}")

def interactive_session(agent_type: str = 'base_agent', verbose: bool = True, model: Optional[str] = None,
                        config_path: Optional[str] = None, use_hardware: bool = True):
    """
    Start an interactive session with the specified agent type.
    
    Args:
        agent_type: Type of agent to use (specified in YAML configs)
        verbose: Whether to enable verbose logging
        model: Optional model name to use
        config_path: Optional path to a YAML configuration file
        use_hardware: Whether to use hardware or mock implementations
    """
    # Create the agent
    agent = create_session_agent(agent_type, verbose, model, config_path, use_hardware)
    
    # Print welcome header
    display_type = agent_type
    if hasattr(agent, 'agent_type'):
        display_type = agent.agent_type
    print_header(display_type)
    
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
            print_header(display_type)
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
        
        try:
            # Get agent response with extra error handling
            response = agent.process(user_input, chat_history)
            
            # If response is None, handle gracefully
            if response is None:
                logger.error("Received None response from agent.process()")
                output = "I apologize, but I encountered an error processing your request. Please try again with a different question."
            else:
                # Extract output or use a default message
                output = response.get("output", "I apologize, but I couldn't generate a proper response.")
                
                # Basic validation on output
                if output is None or not isinstance(output, str) or len(output.strip()) == 0:
                    logger.warning("Received invalid output from agent")
                    output = "I generated an empty or invalid response. Please try again with a different question."
            
            # Print the response
            print(output)
            
            # Update chat history with valid messages only
            chat_history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": output}
            ])
            
        except Exception as e:
            # Handle any errors during processing
            logger.error(f"Error processing user input: {e}")
            print(f"Sorry, I encountered an error: {str(e)}")
            print("Please try again with a different question.")

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description='Interactive Kitchen Assistive Robot AI')
    parser.add_argument('--agent', '-a', type=str, default='base_agent',
                        help='Agent type to use from YAML configs')
    parser.add_argument('--model', '-m', type=str, 
                        help='Model name to use (overrides setting in config)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Disable verbose logging')
    parser.add_argument('--config', '-c', type=str,
                        help='Path to a YAML configuration file')
    parser.add_argument('--list-configs', '-l', action='store_true',
                        help='List available agent configurations')
    parser.add_argument('--no-hardware', action='store_true',
                        help='Disable hardware use (use mock implementations)')
    
    args = parser.parse_args()
    
    # If --list-configs is specified, show available configurations and exit
    if args.list_configs:
        available_agents = get_available_agent_types()
        print("\nAvailable agent configurations:")
        for agent_type in available_agents:
            print(f"  - {agent_type}")
        print("\nTo use a specific configuration, run with --agent <agent_type>")
        return
    
    try:
        interactive_session(
            agent_type=args.agent,
            verbose=not args.quiet,
            model=args.model,
            config_path=args.config,
            use_hardware=not args.no_hardware  # Pass hardware flag to the session
        )
    except KeyboardInterrupt:
        print("\nInteractive session terminated by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in interactive session: {e}")
        # Suggest using a different agent type or providing a config path
        print("\nEncountered an error. You might want to try:")
        print(f"1. A different agent type: python interactive.py --agent base_agent")
        print(f"2. List available agents: python interactive.py --list-configs")
        print(f"3. Running without hardware: python interactive.py --no-hardware")
        sys.exit(1)

if __name__ == "__main__":
    main() 