#!/usr/bin/env python3
"""
Interactive testing script for Kitchen Assistive Robot AI agents.
This script allows you to have a conversation with different agent types.
"""

import os
import sys
import argparse
import logging
import yaml
import time
from typing import Optional, List, Dict, Any

# Add parent directory to sys.path to allow direct script execution
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

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
                config_path: Optional[str] = None, use_hardware: bool = True,
                enable_conversation_logging: bool = False) -> ConfigurableAgent:
    """
    Create an agent for the interactive session.
    
    Args:
        agent_type: Type of agent to use (specified in YAML configs)
        verbose: Whether to enable verbose logging
        model: Optional model name to use
        config_path: Optional path to a YAML configuration file
        use_hardware: Whether to use hardware or mock implementations
        enable_conversation_logging: Whether to log the full conversation to a file
        
    Returns:
        An instance of the ConfigurableAgent
    """
    try:
        # Use the create_agent function from agents.py with conversation logging
        return create_agent(
            agent_type=agent_type,
            config_path=config_path,
            verbose=verbose,
            model_name=model,
            use_hardware=use_hardware,
            enable_conversation_logging=enable_conversation_logging
        )
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        # Check available agent types
        available_agents = get_available_agent_types()
        
        if available_agents:
            logger.info(f"Available agent types: {', '.join(available_agents)}")
        raise ValueError(f"Failed to create agent of type '{agent_type}': {str(e)}")

def interactive_session(agent_type: str = 'base_agent', verbose: bool = True, model: Optional[str] = None,
                        config_path: Optional[str] = None, use_hardware: bool = True,
                        enable_conversation_logging: bool = False):
    """
    Start an interactive session with the specified agent type.
    
    Args:
        agent_type: Type of agent to use (specified in YAML configs)
        verbose: Whether to enable verbose logging
        model: Optional model name to use
        config_path: Optional path to a YAML configuration file
        use_hardware: Whether to use hardware or mock implementations
        enable_conversation_logging: Whether to log the full conversation to a file
    """
    # Create the agent with conversation logging
    agent = create_session_agent(
        agent_type, 
        verbose, 
        model, 
        config_path, 
        use_hardware,
        enable_conversation_logging
    )
    
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
            response = agent.process(user_input)
            
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

class ActionPlanExecutor:
    """Execute action plans from YAML files using the agent."""
    
    def __init__(self, agent: ConfigurableAgent):
        """
        Initialize the action plan executor.
        
        Args:
            agent: The agent to use for executing actions
        """
        self.agent = agent
        self.memory_dir = self._get_memory_path()
        self.action_plans_file = os.path.join(self.memory_dir, "action_plan.yaml")
    
    def _get_memory_path(self) -> str:
        """Get the path to the memory directory."""
        # Get the current file's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # The memory directory is a subdirectory of llm_ai_agent
        memory_dir = os.path.join(current_dir, "memory")
        return memory_dir
    
    def list_available_plans(self) -> List[str]:
        """
        List all available action plans.
        
        Returns:
            A list of action plan names
        """
        if not os.path.exists(self.action_plans_file):
            logger.warning(f"Action plans file not found: {self.action_plans_file}")
            return []
        
        try:
            with open(self.action_plans_file, 'r') as file:
                action_plans = yaml.safe_load(file)
            
            if not action_plans:
                logger.warning("No action plans found in the action plans file")
                return []
            
            return list(action_plans.keys())
        except Exception as e:
            logger.error(f"Error loading action plans: {e}")
            return []
    
    def get_plan_details(self, plan_name: str) -> Dict[str, Any]:
        """
        Get the details of a specific action plan.
        
        Args:
            plan_name: The name of the action plan
            
        Returns:
            A dictionary containing the plan details
        """
        if not os.path.exists(self.action_plans_file):
            logger.warning(f"Action plans file not found: {self.action_plans_file}")
            return {}
        
        try:
            with open(self.action_plans_file, 'r') as file:
                action_plans = yaml.safe_load(file)
            
            if not action_plans or plan_name not in action_plans:
                logger.warning(f"Action plan '{plan_name}' not found")
                return {}
            
            return action_plans[plan_name]
        except Exception as e:
            logger.error(f"Error loading action plan '{plan_name}': {e}")
            return {}
    
    def execute_plan(self, plan_name: str) -> List[Dict[str, Any]]:
        """
        Execute an action plan step by step.
        
        Args:
            plan_name: The name of the action plan to execute
            
        Returns:
            A list of execution results for each step
        """
        plan_details = self.get_plan_details(plan_name)
        if not plan_details:
            return []
        
        goal = plan_details.get('goal', 'No goal specified')
        steps = plan_details.get('steps', [])
        
        if not steps:
            logger.warning(f"No steps found in action plan '{plan_name}'")
            return []
        
        print(f"\n📋 Executing action plan: {plan_name}")
        print(f"🎯 Goal: {goal}")
        print(f"🔢 Total steps: {len(steps)}")
        print("=" * 50)
        
        execution_history = []
        
        for i, step in enumerate(steps, 1):
            step_num = step.get('step_num', i)
            description = step.get('description', 'No description')
            
            print(f"\n▶️ Step {step_num}/{len(steps)}: {description}")
            
            # Ask the agent to execute this step
            prompt = f"Execute:{description}"
            
            try:
                # Process the step
                print("🤖 Agent: ", end="", flush=True)
                response = self.agent.process(prompt)
                
                # Extract the output
                if response is None:
                    output = "Error executing step"
                    success = False
                else:
                    output = response.get("output", "No output")
                    # Consider the step successful if there's a valid output
                    success = isinstance(output, str) and len(output.strip()) > 0
                
                # Print the response
                print(output)
                
                # Record step execution
                execution_result = {
                    "step_num": step_num,
                    "description": description,
                    "prompt": prompt,
                    "response": output,
                    "success": success
                }
                
                execution_history.append(execution_result)
                
                # Wait a moment before moving to the next step
                # if i < len(steps):
                    # user_input = input("\nPress Enter to continue to the next step, or type 'skip' to skip this plan: ")
                    # if user_input.lower() == 'skip':
                    #     print("Skipping remaining steps...")
                    #     break
                
            except Exception as e:
                logger.error(f"Error executing step {step_num}: {e}")
                execution_result = {
                    "step_num": step_num,
                    "description": description,
                    "prompt": prompt,
                    "response": f"Error: {str(e)}",
                    "success": False
                }
                execution_history.append(execution_result)
                
                # Ask whether to continue or abort
                user_input = input("\nError executing step. Press Enter to continue to the next step, or type 'abort' to stop: ")
                if user_input.lower() == 'abort':
                    print("Aborting plan execution...")
                    break
        
        print("\n✅ Action plan execution completed")
        print("=" * 50)
        
        return execution_history

def action_plan_mode(agent_type: str = 'base_agent', verbose: bool = True, model: Optional[str] = None,
                    config_path: Optional[str] = None, use_hardware: bool = True,
                    enable_conversation_logging: bool = False):
    """
    Run the agent in action plan execution mode.
    
    Args:
        agent_type: Type of agent to use (specified in YAML configs)
        verbose: Whether to enable verbose logging
        model: Optional model name to use
        config_path: Optional path to a YAML configuration file
        use_hardware: Whether to use hardware or mock implementations
        enable_conversation_logging: Whether to log the full conversation to a file
    """
    # Create the agent
    try:
        agent = create_session_agent(
            agent_type, 
            verbose, 
            model, 
            config_path, 
            use_hardware,
            enable_conversation_logging
        )
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        print(f"Failed to create agent: {e}")
        return
    
    # Create the action plan executor
    executor = ActionPlanExecutor(agent)
    
    # Clear screen and print header
    clear_screen()
    print("=" * 80)
    print(f"Kitchen Assistive Robot - Action Plan Execution Mode ({agent.agent_type})")
    print("=" * 80)
    
    # List available plans
    plans = executor.list_available_plans()
    if not plans:
        print("No action plans found. Please check the action_plan.yaml file.")
        return
    
    print("Available action plans:")
    for i, plan in enumerate(plans, 1):
        plan_details = executor.get_plan_details(plan)
        goal = plan_details.get('goal', 'No goal specified')
        steps_count = len(plan_details.get('steps', []))
        print(f"{i}. {plan} - {goal} ({steps_count} steps)")
    
    # Action plan selection and execution loop
    while True:
        try:
            choice = input("\nSelect a plan to execute (number) or 'q' to quit: ")
            
            if choice.lower() in ['q', 'quit', 'exit']:
                print("\nExiting action plan mode. Goodbye!")
                break
            
            try:
                plan_index = int(choice) - 1
                if plan_index < 0 or plan_index >= len(plans):
                    print(f"Invalid choice. Please enter a number between 1 and {len(plans)}.")
                    continue
                
                selected_plan = plans[plan_index]
                
                # Execute the selected plan
                execution_history = executor.execute_plan(selected_plan)
                
                # Show execution summary
                success_count = sum(1 for step in execution_history if step['success'])
                total_steps = len(execution_history)
                
                print("\n📊 Execution Summary:")
                print(f"Plan: {selected_plan}")
                print(f"Steps completed: {len(execution_history)}/{total_steps}")
                print(f"Successful steps: {success_count}/{total_steps}")
                print(f"Status: {'✅ Completed' if success_count == total_steps else '❌ Incomplete'}")
                
                # Ask if user wants to execute another plan
                another = input("\nExecute another plan? (y/n): ")
                if another.lower() != 'y':
                    print("\nExiting action plan mode. Goodbye!")
                    break
            except ValueError:
                print("Please enter a valid number.")
                
        except KeyboardInterrupt:
            print("\nAction plan execution interrupted by user.")
            break
        except Exception as e:
            logger.error(f"Error in action plan mode: {e}")
            print(f"\nError: {e}")
            print("Please try again or select a different plan.")

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
    parser.add_argument('--log-conversation', action='store_true',
                        help='Enable detailed conversation logging to file')
    parser.add_argument('--action-plan', action='store_true',
                        help='Run in action plan execution mode')
    
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
        # Select mode based on arguments
        if args.action_plan:
            action_plan_mode(
                agent_type=args.agent,
                verbose=not args.quiet,
                model=args.model,
                config_path=args.config,
                use_hardware=not args.no_hardware,
                enable_conversation_logging=args.log_conversation
            )
        else:
            interactive_session(
                agent_type=args.agent,
                verbose=not args.quiet,
                model=args.model,
                config_path=args.config,
                use_hardware=not args.no_hardware,
                enable_conversation_logging=args.log_conversation
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