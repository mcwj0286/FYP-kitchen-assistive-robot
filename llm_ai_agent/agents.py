#!/usr/bin/env python3
"""
Central module for Kitchen Assistive Robot agents.
This module exports the ConfigurableAgent class as the main interface
for creating and using agents with different configurations.
"""

import logging
from typing import Dict, Any, List, Optional

from .configurable_agent import ConfigurableAgent
from .config_loader import AgentConfigLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_agent(
    agent_type: str = "base_agent", 
    config_path: Optional[str] = None,
    verbose: bool = True,
    model_name: Optional[str] = None,
    use_hardware: bool = True,
    capture_image: str = "",
    enable_conversation_logging: bool = False
) -> ConfigurableAgent:
    """
    Create an agent with the specified configuration.
    
    Args:
        agent_type: Type of agent to create (must match a configuration file)
        config_path: Optional path to a custom configuration file
        verbose: Whether to enable verbose logging
        model_name: Model name to use (overrides config)
        use_hardware: Whether to use real hardware (False uses mock implementations)
        capture_image: Which camera view to capture with each prompt ("environment", "wrist", "both", or "" to disable)
                      If empty string, will use the setting from the agent's YAML config file
                      (hardware.capture_image)
        
    Returns:
        An instance of ConfigurableAgent
    """
    try:
        return ConfigurableAgent(
            agent_type=agent_type,
            config_path=config_path,
            verbose=verbose,
            model_name=model_name,
            use_hardware=use_hardware,
            capture_image=capture_image,
            enable_conversation_logging=enable_conversation_logging
        )
    except Exception as e:
        logger.error(f"Error creating agent of type '{agent_type}': {e}")
        raise

def get_available_agent_types() -> List[str]:
    """
    Get a list of all available agent types.
    
    Returns:
        List of agent type names
    """
    config_loader = AgentConfigLoader()
    return config_loader.get_available_agent_types()

def get_agent_info(agent_type: str) -> Dict[str, Any]:
    """
    Get information about a specific agent type.
    
    Args:
        agent_type: The type of agent to get information for
        
    Returns:
        Dictionary with agent information
    """
    config_loader = AgentConfigLoader()
    
    try:
        config = config_loader.get_agent_config(agent_type)
        tools = config_loader.get_agent_tools(agent_type)
        
        return {
            "agent_type": agent_type,
            "description": config.get("description", "No description available"),
            "version": config.get("version", "1.0.0"),
            "tools": tools,
            "hardware_required": config.get("hardware_required", False)
        }
    except ValueError:
        logger.error(f"Agent type not found: {agent_type}")
        return {"error": f"Agent type '{agent_type}' not found"}


# Example usage
if __name__ == "__main__":
    import sys
    
    # List available agent types
    agent_types = get_available_agent_types()
    print(f"Available agent types: {', '.join(agent_types)}")
    
    # If an agent type is specified, show information about it
    if len(sys.argv) > 1:
        agent_type = sys.argv[1]
        info = get_agent_info(agent_type)
        
        if "error" in info:
            print(info["error"])
        else:
            print("\nAgent Information:")
            print(f"  Type: {info['agent_type']}")
            print(f"  Description: {info['description']}")
            print(f"  Version: {info['version']}")
            print(f"  Hardware Required: {info['hardware_required']}")
            print(f"  Tools: {', '.join(info['tools'])}")
    else:
        print("\nTo see information about a specific agent type, run:")
        print(f"  python -m llm_ai_agent.agents <agent_type>") 