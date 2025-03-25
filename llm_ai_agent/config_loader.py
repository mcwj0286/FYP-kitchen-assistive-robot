#!/usr/bin/env python3
"""
Configuration loading utilities for the Kitchen Assistant Agent.
This module provides tools for loading and processing YAML-based agent configurations.
"""

import os
import sys
import yaml
import logging
from typing import Dict, Any, List, Optional, Set, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentConfigLoader:
    """Loader for agent configurations from YAML files."""
    
    def __init__(self, config_dir: str = None):
        """
        Initialize the config loader.
        
        Args:
            config_dir: Path to the configuration directory (defaults to 'configs' in the current directory)
        """
        # Default configuration directory
        if config_dir is None:
            # Use the directory where this file is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_dir = os.path.join(current_dir, 'configs')
        
        self.config_dir = config_dir
        self.agents_dir = os.path.join(config_dir, 'agents')
        self.tools_dir = os.path.join(config_dir, 'tools')
        self.prompts_dir = os.path.join(config_dir, 'prompts')
        
        # Check if directories exist
        self._check_directories()
        
        # Load configurations
        self.agent_configs = self._load_agent_configs()
        self.tool_configs = self._load_tool_configs()
        
        # Process inheritance in agent configs
        self._process_inheritance()
        
        logger.info(f"Loaded configuration for {len(self.agent_configs)} agent types")
    
    def _check_directories(self) -> None:
        """Check if required configuration directories exist."""
        required_dirs = [self.config_dir, self.agents_dir, self.tools_dir]
        
        for directory in required_dirs:
            if not os.path.exists(directory):
                logger.warning(f"Configuration directory not found: {directory}")
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
    
    def _load_agent_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load agent configurations from YAML files.
        
        Returns:
            Dictionary mapping agent types to their configurations
        """
        agent_configs = {}
        
        if not os.path.exists(self.agents_dir):
            logger.warning(f"Agents directory not found: {self.agents_dir}")
            return agent_configs
        
        # Find all YAML files in the agents directory
        for filename in os.listdir(self.agents_dir):
            if filename.endswith(('.yaml', '.yml')):
                file_path = os.path.join(self.agents_dir, filename)
                
                try:
                    with open(file_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Extract agent type from the file
                    agent_type = config.get('agent_type')
                    if agent_type:
                        agent_configs[agent_type] = config
                        logger.debug(f"Loaded agent configuration: {agent_type}")
                    else:
                        logger.warning(f"Missing agent_type in configuration: {file_path}")
                
                except Exception as e:
                    logger.error(f"Error loading agent configuration from {file_path}: {e}")
        
        return agent_configs
    
    def _load_tool_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load tool category configurations from YAML files.
        
        Returns:
            Dictionary mapping tool category names to their configurations
        """
        tool_configs = {}
        
        if not os.path.exists(self.tools_dir):
            logger.warning(f"Tools directory not found: {self.tools_dir}")
            return tool_configs
        
        # Find all YAML files in the tools directory
        for filename in os.listdir(self.tools_dir):
            if filename.endswith(('.yaml', '.yml')):
                file_path = os.path.join(self.tools_dir, filename)
                
                try:
                    with open(file_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Extract categories
                    categories = config.get('categories', {})
                    if categories:
                        # Add each category to the tool configs
                        for category_name, category_config in categories.items():
                            # Create a qualified category name (file.category)
                            file_prefix = os.path.splitext(filename)[0]
                            qualified_name = f"{file_prefix}.{category_name}"
                            tool_configs[qualified_name] = category_config
                        
                        logger.debug(f"Loaded tool categories from: {filename}")
                    else:
                        logger.warning(f"No categories found in tool configuration: {file_path}")
                
                except Exception as e:
                    logger.error(f"Error loading tool configuration from {file_path}: {e}")
        
        return tool_configs
    
    def _process_inheritance(self) -> None:
        """Process inheritance between agent configurations."""
        # Keep track of processed agents to avoid circular inheritance
        processed = set()
        
        def process_agent(agent_type: str) -> Dict[str, Any]:
            """Recursively process an agent and its parents."""
            if agent_type in processed:
                logger.warning(f"Circular inheritance detected for agent: {agent_type}")
                return self.agent_configs[agent_type]
            
            if agent_type not in self.agent_configs:
                logger.warning(f"Agent type not found: {agent_type}")
                return {}
            
            config = self.agent_configs[agent_type]
            parent_type = config.get('inherits_from')
            
            if not parent_type:
                # No inheritance, return as is
                return config
            
            # Process the parent first
            parent_config = process_agent(parent_type)
            
            # Mark this agent as processed
            processed.add(agent_type)
            
            # Create a new configuration by merging parent and child
            merged_config = parent_config.copy()
            
            # Special handling for tools (merge rather than override)
            if 'tools' in config and 'tools' in parent_config:
                merged_tools = self._merge_tools(parent_config['tools'], config['tools'])
                merged_config['tools'] = merged_tools
                
                # Remove tools from child config to avoid double-merging
                child_config = config.copy()
                if 'tools' in child_config:
                    del child_config['tools']
                
                # Update merged config with remaining child properties
                merged_config.update(child_config)
            else:
                # Standard update for other properties
                merged_config.update(config)
            
            # Remove inherits_from to avoid confusion
            if 'inherits_from' in merged_config:
                del merged_config['inherits_from']
            
            # Update the agent_configs dictionary with the merged config
            self.agent_configs[agent_type] = merged_config
            
            return merged_config
        
        # Process each agent
        for agent_type in list(self.agent_configs.keys()):
            process_agent(agent_type)
    
    def _merge_tools(self, parent_tools: Dict[str, Any], child_tools: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge tool configurations from parent and child agents.
        
        Args:
            parent_tools: Tool configuration from parent
            child_tools: Tool configuration from child
            
        Returns:
            Merged tool configuration
        """
        # Create a copy of parent_tools or initialize to empty dict if None
        merged = parent_tools.copy() if parent_tools else {}
        
        # Handle None for child_tools
        if not child_tools:
            return merged
        
        # Merge categories
        if 'categories' in child_tools:
            if 'categories' not in merged:
                merged['categories'] = []
            
            # Add child categories
            parent_categories = set(merged['categories'] or [])
            child_categories = set(child_tools['categories'] or [])
            merged['categories'] = list(parent_categories.union(child_categories))
        
        # Merge includes
        if 'include' in child_tools:
            if 'include' not in merged:
                merged['include'] = []
            
            # Add child includes
            parent_includes = set(merged['include'] or [])
            child_includes = set(child_tools['include'] or [])
            merged['include'] = list(parent_includes.union(child_includes))
        
        # Merge excludes
        if 'exclude' in child_tools:
            if 'exclude' not in merged:
                merged['exclude'] = []
            
            # Add child excludes
            parent_excludes = set(merged['exclude'] or [])
            child_excludes = set(child_tools['exclude'] or [])
            merged['exclude'] = list(parent_excludes.union(child_excludes))
        
        return merged
    
    def get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific agent type.
        
        Args:
            agent_type: The type of agent to get configuration for
            
        Returns:
            Dictionary with agent configuration
        """
        if agent_type not in self.agent_configs:
            logger.warning(f"Agent type not found: {agent_type}")
            raise ValueError(f"No configuration found for agent type: {agent_type}")
        
        return self.agent_configs[agent_type].copy()
    
    def get_agent_tools(self, agent_type: str) -> List[str]:
        """
        Get the list of tools for a specific agent type.
        
        Args:
            agent_type: The type of agent to get tools for
            
        Returns:
            List of tool names assigned to the agent
        """
        config = self.get_agent_config(agent_type)
        tools_config = config.get('tools', {})
        
        # Tools explicitly included
        included_tools = set(tools_config.get('include', []))
        
        # Tools from categories
        category_tools = set()
        for category in tools_config.get('categories', []):
            if category in self.tool_configs:
                tools = self.tool_configs[category].get('tools', [])
                for tool in tools:
                    tool_name = tool.get('name')
                    if tool_name:
                        category_tools.add(tool_name)
            else:
                logger.warning(f"Tool category not found: {category}")
        
        # Combine included and category tools
        all_tools = included_tools.union(category_tools)
        
        # Remove excluded tools
        excluded_tools = set(tools_config.get('exclude', []))
        final_tools = all_tools - excluded_tools
        
        return sorted(list(final_tools))
    
    def get_agent_system_prompt(self, agent_type: str) -> str:
        """
        Get the system prompt for a specific agent type.
        
        Args:
            agent_type: The type of agent to get the system prompt for
            
        Returns:
            System prompt as a string
        """
        config = self.get_agent_config(agent_type)
        return config.get('system_prompt', '')
    
    def get_available_agent_types(self) -> List[str]:
        """
        Get a list of all available agent types.
        
        Returns:
            List of agent type names
        """
        return sorted(list(self.agent_configs.keys()))
    
    def get_tool_details(self, tool_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific tool.
        
        Args:
            tool_name: Name of the tool to get details for
            
        Returns:
            Dictionary with tool details or empty dict if tool not found
        """
        # Search for the tool in all categories
        for category_name, category_config in self.tool_configs.items():
            tools = category_config.get('tools', [])
            for tool in tools:
                if tool.get('name') == tool_name:
                    # Add category information
                    tool_copy = tool.copy()
                    tool_copy['category'] = category_name
                    return tool_copy
        
        logger.warning(f"Tool not found: {tool_name}")
        return {}


# Example usage
if __name__ == "__main__":
    # Create a config loader
    config_loader = AgentConfigLoader()
    
    # Show available agent types
    agent_types = config_loader.get_available_agent_types()
    print(f"Available agent types: {agent_types}")
    
    # Show tools for each agent type
    for agent_type in agent_types:
        tools = config_loader.get_agent_tools(agent_type)
        print(f"\nTools for {agent_type}: {tools}")
        
        # Show the first few lines of the system prompt
        system_prompt = config_loader.get_agent_system_prompt(agent_type)
        prompt_preview = system_prompt.split('\n')[:3]
        print(f"System prompt preview: {prompt_preview}") 