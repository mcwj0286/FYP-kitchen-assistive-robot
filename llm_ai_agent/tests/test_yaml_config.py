#!/usr/bin/env python3
"""
Tests for the YAML configuration system for agents.
This script validates that agent configurations can be loaded from YAML files
and that agents can be created using these configurations.
"""

import os
import sys
import unittest
import logging
from pathlib import Path

# Add parent directory to path to allow imports
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Import the necessary modules
from config_loader import AgentConfigLoader
from configurable_agent import ConfigurableAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestYAMLConfig(unittest.TestCase):
    """Test cases for YAML configuration system."""
    
    def setUp(self):
        """Set up the test environment."""
        self.config_loader = AgentConfigLoader()
        
    def test_available_agent_types(self):
        """Test that agent types can be listed."""
        agent_types = self.config_loader.get_available_agent_types()
        self.assertIsInstance(agent_types, list)
        # We should have at least the base_agent and kitchen_assistant
        self.assertIn("base_agent", agent_types)
        self.assertIn("kitchen_assistant", agent_types)
        logger.info(f"Available agent types: {agent_types}")
        
    def test_load_base_agent_config(self):
        """Test loading the base agent configuration."""
        config = self.config_loader.get_agent_config("base_agent")
        self.assertIsNotNone(config)
        self.assertEqual(config.get("agent_type"), "base_agent")
        self.assertIn("system_prompt", config)
        self.assertIn("tools", config)
        logger.info("Base agent configuration loaded successfully")
        
    def test_load_kitchen_assistant_config(self):
        """Test loading the kitchen assistant configuration."""
        config = self.config_loader.get_agent_config("kitchen_assistant")
        self.assertIsNotNone(config)
        self.assertEqual(config.get("agent_type"), "kitchen_assistant")
        self.assertIn("system_prompt", config)
        self.assertIn("tools", config)
        self.assertTrue(config.get("hardware_required"))
        logger.info("Kitchen assistant configuration loaded successfully")
        
    def test_load_tool_configs(self):
        """Test loading tool configurations."""
        # Get tool details for calculator and text_processor
        calculator_tools = self.config_loader.get_tool_details("calculator")
        text_processor_tools = self.config_loader.get_tool_details("text_processor")
        
        self.assertIsNotNone(calculator_tools)
        self.assertIsNotNone(text_processor_tools)
        
        # At minimum, these should have a name field
        self.assertIn("name", calculator_tools)
        self.assertIn("name", text_processor_tools)
        
        logger.info("Tool configurations loaded successfully")
        
    def test_create_base_agent(self):
        """Test creating a base agent from configuration."""
        try:
            agent = ConfigurableAgent(
                agent_type="base_agent",
                verbose=True,
                use_hardware=False
            )
            self.assertIsNotNone(agent)
            self.assertEqual(agent.agent_type, "base_agent")
            logger.info("Base agent created successfully from configuration")
        except Exception as e:
            self.fail(f"Failed to create base agent: {e}")
            
    def test_create_kitchen_assistant(self):
        """Test creating a kitchen assistant agent from configuration."""
        try:
            agent = ConfigurableAgent(
                agent_type="kitchen_assistant",
                verbose=True,
                use_hardware=False  # Use mock hardware for testing
            )
            self.assertIsNotNone(agent)
            self.assertEqual(agent.agent_type, "kitchen_assistant")
            logger.info("Kitchen assistant agent created successfully from configuration")
        except Exception as e:
            self.fail(f"Failed to create kitchen assistant agent: {e}")
            
    def test_agent_has_tools(self):
        """Test that the agent has the expected tools."""
        agent = ConfigurableAgent(
            agent_type="base_agent",
            verbose=False,
            use_hardware=False
        )
        # Check if the agent has the expected tools
        self.assertIn("calculator", agent.agent._available_tools)
        self.assertIn("text_processor", agent.agent._available_tools)
        logger.info("Agent has the expected tools")
        
    def test_agent_response(self):
        """Test that the agent can generate a response."""
        agent = ConfigurableAgent(
            agent_type="base_agent",
            verbose=False,
            use_hardware=False
        )
        
        # Simple query that should use the calculator tool
        response = agent.process_to_string("What is 5 + 10?")
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)
        # The response should contain the calculation result
        self.assertGreater(len(response), 0)
        logger.info(f"Agent response: {response}")

if __name__ == "__main__":
    unittest.main() 