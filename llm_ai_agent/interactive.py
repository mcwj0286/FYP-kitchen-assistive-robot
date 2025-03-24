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
from configurable_agent import ConfigurableAgent

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

def create_agent(agent_type: str, verbose: bool = True, model: Optional[str] = None, 
                config_path: Optional[str] = None, use_hardware: bool = True) -> BaseAgent:
    """
    Create an agent of the specified type.
    
    Args:
        agent_type: Type of agent to create ('base', 'kitchen', or from YAML config)
        verbose: Whether to enable verbose logging
        model: Optional model name to use
        config_path: Optional path to a YAML configuration file
        use_hardware: Whether to use hardware or mock implementations
        
    Returns:
        An instance of the specified agent type
    """
    # If a config path is provided, use the ConfigurableAgent
    if config_path:
        logger.info(f"Creating agent from configuration: {config_path}")
        return ConfigurableAgent(
            agent_type=agent_type,
            config_path=config_path,
            verbose=verbose,
            model_name=model,
            use_hardware=use_hardware
        )
    
    # If agent_type is in the YAML format (not 'base' or 'kitchen'),
    # use the ConfigurableAgent with the standard configs
    if agent_type not in ['base', 'kitchen']:
        logger.info(f"Creating agent from predefined configuration: {agent_type}")
        return ConfigurableAgent(
            agent_type=agent_type, 
            verbose=verbose,
            model_name=model,
            use_hardware=use_hardware
        )
    
    # Otherwise, use the legacy agent creation
    if agent_type.lower() == 'kitchen':
        return KitchenAssistantAgent(model_name=model, verbose=verbose, use_hardware=use_hardware)
    else:  # Default to base agent
        return BaseAgent(model_name=model, verbose=verbose, use_hardware=use_hardware)

def interactive_session(agent_type: str = 'base', verbose: bool = True, model: Optional[str] = None,
                        config_path: Optional[str] = None, use_hardware: bool = True):
    """
    Start an interactive session with the specified agent type.
    
    Args:
        agent_type: Type of agent to use ('base', 'kitchen', or from YAML config)
        verbose: Whether to enable verbose logging
        model: Optional model name to use
        config_path: Optional path to a YAML configuration file
        use_hardware: Whether to use hardware or mock implementations
    """
    # Create the agent
    agent = create_agent(agent_type, verbose, model, config_path, use_hardware)
    
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
    parser.add_argument('--agent', '-a', type=str, default='base',
                        help='Agent type to use (base, kitchen, or a custom type from configs)')
    parser.add_argument('--model', '-m', type=str, 
                        help='Model name to use (overrides setting in config)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Disable verbose logging')
    parser.add_argument('--config', '-c', type=str,
                        help='Path to a YAML configuration file')
    parser.add_argument('--list-configs', '-l', action='store_true',
                        help='List available agent configurations')
    parser.add_argument('--no-hardware', action='store_true',
                        help='Disable hardware use (for testing)')
    
    args = parser.parse_args()
    
    # If --list-configs is specified, show available configurations and exit
    if args.list_configs:
        from config_loader import AgentConfigLoader
        config_loader = AgentConfigLoader()
        available_agents = config_loader.get_available_agent_types()
        print("\nAvailable agent configurations:")
        for agent_type in available_agents:
            print(f"  - {agent_type}")
        print("\nTo use a specific configuration, run with --agent <agent_type>")
        return
    
    try:
        # Set hardware usage based on the --no-hardware flag
        use_hardware = not args.no_hardware
        
        # Print hardware status
        if args.no_hardware:
            logger.info("Hardware disabled (using mock implementations)")
        else:
            logger.info("Hardware enabled (using real hardware if available)")
            
            # If using a kitchen agent or kitchen_assistant configuration without explicit model
            # and not in quiet mode, suggest using --no-hardware flag if harware might not be available
            if not args.quiet and (args.agent.lower() == 'kitchen' or args.agent == 'kitchen_assistant'):
                print("\nNOTE: Using kitchen agent with hardware. If you encounter hardware errors,")
                print("      try running with the --no-hardware flag for testing purposes.")
                print("      Example: python interactive.py --agent kitchen_assistant --no-hardware\n")
        
        interactive_session(
            agent_type=args.agent,
            verbose=not args.quiet,
            model=args.model,
            config_path=args.config,
            use_hardware=use_hardware
        )
    except KeyboardInterrupt:
        print("\nInteractive session terminated by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in interactive session: {e}")
        # Suggest the no-hardware flag if there's an error and hardware is enabled
        if not args.no_hardware:
            print("\nEncountered an error. If this is related to hardware initialization,")
            print("try running with the --no-hardware flag:")
            print(f"python interactive.py --agent {args.agent} --no-hardware")
        sys.exit(1)

if __name__ == "__main__":
    main() 