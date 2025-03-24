#!/usr/bin/env python3
"""
Configurable Agent implementation for Kitchen Assistant Robot.
This module provides an agent class that can be configured via YAML files.
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional, Callable, Union

from base_agent import BaseAgent, KitchenAssistantAgent
from config_loader import AgentConfigLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigurableAgent:
    """
    Agent that can be configured via YAML files.
    This provides a unified interface to different agent types.
    """
    
    def __init__(
        self, 
        agent_type: str = "base_agent", 
        config_path: Optional[str] = None,
        verbose: bool = True,
        use_hardware: Optional[bool] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize a configurable agent.
        
        Args:
            agent_type: Type of agent to create (must match a configuration file)
            config_path: Optional path to a custom configuration file
            verbose: Whether to enable verbose logging
            use_hardware: Whether to use hardware (overrides config)
            model_name: Model name to use (overrides config)
        """
        self.agent_type = agent_type
        self.verbose = verbose
        
        # Initialize configuration loader
        if config_path:
            config_dir = os.path.dirname(os.path.abspath(config_path))
            self.config_loader = AgentConfigLoader(config_dir=config_dir)
            
            # If a specific file is provided, load it directly
            try:
                with open(config_path, 'r') as f:
                    import yaml
                    config = yaml.safe_load(f)
                    
                if 'agent_type' in config:
                    self.agent_type = config['agent_type']
                    self.config_loader.agent_configs[self.agent_type] = config
                    # Process inheritance for this config
                    self.config_loader._process_inheritance()
                else:
                    logger.warning(f"No agent_type found in {config_path}")
            except Exception as e:
                logger.error(f"Error loading custom config {config_path}: {e}")
        else:
            self.config_loader = AgentConfigLoader()
        
        # Load agent configuration
        try:
            self.config = self.config_loader.get_agent_config(self.agent_type)
            logger.info(f"Loaded configuration for {self.agent_type}")
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            available_agents = self.config_loader.get_available_agent_types()
            logger.info(f"Available agent types: {available_agents}")
            if available_agents:
                self.agent_type = available_agents[0]
                self.config = self.config_loader.get_agent_config(self.agent_type)
                logger.info(f"Falling back to {self.agent_type}")
            else:
                raise ValueError("No agent configurations available")
        
        # Create the base agent instance based on config
        self._create_agent(use_hardware, model_name)
    
    def _create_agent(self, use_hardware: Optional[bool], model_name: Optional[str]):
        """
        Create the underlying agent based on the configuration.
        
        Args:
            use_hardware: Whether to use hardware (overrides config)
            model_name: Model name to use (overrides config)
        """
        # Get system prompt from configuration
        system_prompt = self.config.get('system_prompt', '')
        
        # Get hardware requirements from configuration
        hardware_required = self.config.get('hardware_required', False)
        if use_hardware is not None:
            # Override config if explicitly specified
            hardware_required = use_hardware
        
        # Log hardware configuration
        if hardware_required:
            logger.info("Hardware enabled (using real hardware if available)")
        else:
            logger.info("Hardware disabled (using mock implementations)")
        
        # Get model parameters from configuration
        model_defaults = self.config.get('model_defaults', {})
        if model_name:
            # Override model name if explicitly specified
            model = model_name
        else:
            model = model_defaults.get('model_name')
        
        # Get temperature and max_tokens
        temperature = model_defaults.get('temperature', 0.7)
        max_tokens = model_defaults.get('max_tokens', 1024)
        
        # Check if this is a kitchen assistant or base agent
        is_kitchen = self.agent_type == 'kitchen_assistant' or self.config.get('inherits_from') == 'kitchen_assistant'
        
        # Create appropriate agent type
        try:
            if is_kitchen:
                logger.info(f"Creating KitchenAssistantAgent with hardware={hardware_required}")
                self.agent = KitchenAssistantAgent(
                    model_name=model,
                    verbose=self.verbose,
                    use_hardware=hardware_required
                )
            else:
                logger.info(f"Creating BaseAgent with custom system prompt")
                self.agent = BaseAgent(
                    model_name=model,
                    system_prompt=system_prompt,
                    verbose=self.verbose,
                    use_hardware=hardware_required
                )
            
            # Apply any additional configuration
            if hasattr(self.agent, 'llm'):
                # Update model parameters
                if temperature is not None:
                    self.agent.llm.temperature = temperature
                if max_tokens is not None:
                    self.agent.llm.max_tokens = max_tokens
                
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            raise
    
    def process(self, user_input: str, chat_history: Optional[List[Dict[str, str]]] = None, max_iterations: int = 5) -> Dict[str, Any]:
        """
        Process a user input and return the agent's response.
        
        Args:
            user_input: The user's question or command
            chat_history: Optional list of previous conversation messages
            max_iterations: Maximum number of tool execution iterations
            
        Returns:
            The agent's response as a dictionary
        """
        return self.agent.process(user_input, chat_history, max_iterations)
    
    def process_to_string(self, user_input: str, chat_history: Optional[List[Dict[str, str]]] = None, max_iterations: int = 5) -> str:
        """
        Process a user input and return the agent's response as a string.
        
        Args:
            user_input: The user's question or command
            chat_history: Optional list of previous conversation messages
            max_iterations: Maximum number of tool execution iterations
            
        Returns:
            The agent's response as a string
        """
        return self.agent.process_to_string(user_input, chat_history, max_iterations)
    
    @property
    def verbose(self) -> bool:
        """Get the verbose setting."""
        return self._verbose
    
    @verbose.setter
    def verbose(self, value: bool) -> None:
        """Set the verbose setting."""
        self._verbose = value
        if hasattr(self, 'agent'):
            self.agent.verbose = value
    
    def __del__(self):
        """Clean up resources when the agent is deleted."""
        if hasattr(self, 'agent') and hasattr(self.agent, '__del__'):
            self.agent.__del__()


# Example usage
if __name__ == "__main__":
    # Create a configurable agent with the base configuration
    agent = ConfigurableAgent(agent_type="base_agent", verbose=True)
    
    # Test a simple query
    response = agent.process_to_string("What is 15 multiplied by 32?")
    print(f"Base agent response: {response}\n")
    
    # Create a kitchen assistant agent
    kitchen_agent = ConfigurableAgent(agent_type="kitchen_assistant", verbose=True)
    
    # Test a kitchen-related query
    response = kitchen_agent.process_to_string("Can you analyze this recipe: 2 cups flour, 1 cup sugar, 3 eggs?")
    print(f"Kitchen agent response: {response}") 