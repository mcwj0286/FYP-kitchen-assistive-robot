#!/usr/bin/env python3
"""
Configurable Agent implementation for Kitchen Assistant Robot.
This module provides a single agent class that can be configured via YAML files.
"""

import os
import sys
import re
import json
import logging
import importlib
from typing import Dict, Any, List, Optional, Callable, Union

# Import necessary modules
from llm_ai_agent.config_loader import AgentConfigLoader

# Try to import LangChain classes (these are essential)
try:
    from langchain.chat_models import ChatOpenAI
except ImportError:
    logging.error("LangChain modules not found. Please install with: pip install langchain langchain-openai")
    sys.exit(1)

# Try to import hardware tools (these are optional)
try:
    from llm_ai_agent.hardware_tools import HardwareTools
    HARDWARE_AVAILABLE = True
except ImportError:
    logging.warning("Hardware tools module not found. Hardware capabilities will not be available.")
    HARDWARE_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigurableAgent:
    """
    A fully configurable agent that loads all properties from YAML configuration.
    This is the only agent class needed, with behavior determined by configuration.
    """
    
    def __init__(
        self, 
        agent_type: str = "base_agent", 
        config_path: Optional[str] = None,
        verbose: bool = True,
        model_name: Optional[str] = None,
        use_hardware: bool = True
    ):
        """
        Initialize a configurable agent.
        
        Args:
            agent_type: Type of agent to create (must match a configuration file)
            config_path: Optional path to a custom configuration file
            verbose: Whether to enable verbose logging
            model_name: Model name to use (overrides config)
            use_hardware: Whether to use real hardware (False uses mock implementations)
        """
        self.agent_type = agent_type
        self._verbose = verbose
        self._use_hardware = use_hardware
        
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
        
        # Initialize properties that will be set during setup
        self.llm = None
        self.hardware = None
        self._available_tools = {}
        self._system_prompt = ""
        
        # Set up the agent based on configuration
        self._setup_agent(model_name)
    
    def _setup_agent(self, model_name: Optional[str]):
        """
        Set up the agent based on the configuration.
        
        Args:
            model_name: Model name to use (overrides config)
        """
        # Get system prompt from configuration
        self._system_prompt = self.config.get('system_prompt', '')
        
        # Get hardware requirements from configuration
        hardware_required = self.config.get('hardware_required', False)
        
        # Set up hardware if required and available
        if hardware_required and HARDWARE_AVAILABLE and self._use_hardware:
            try:
                self.hardware = HardwareTools(use_mock=not self._use_hardware)
                logger.info("Hardware tools initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing hardware tools: {e}")
                self.hardware = None
                hardware_required = False
        
        # Log hardware configuration
        if hardware_required and HARDWARE_AVAILABLE and self._use_hardware:
            logger.info("Hardware enabled (using real hardware)")
        else:
            logger.info("Hardware disabled (using mock implementations or unavailable)")
        
        # Get model parameters from configuration
        model_defaults = self.config.get('model_defaults', {})
        if model_name:
            # Override model name if explicitly specified
            model = model_name
        else:
            model = model_defaults.get('model_name')
        
        # Initialize the language model
        try:
            self.llm = ChatOpenAI(
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                model_name=model or os.getenv("MODEL_NAME", "anthropic/claude-3-opus-20240229"),
                default_headers={
                    "HTTP-Referer": os.getenv("YOUR_SITE_URL", "https://example.com"),
                    "X-Title": os.getenv("YOUR_SITE_NAME", "AI Assistant"),
                }
            )
            
            # Apply model parameters
            temperature = model_defaults.get('temperature', 0.7)
            max_tokens = model_defaults.get('max_tokens', 1024)
            
            if temperature is not None:
                self.llm.temperature = temperature
                logger.info(f"Set LLM temperature to {temperature}")
            if max_tokens is not None:
                self.llm.max_tokens = max_tokens
                logger.info(f"Set LLM max_tokens to {max_tokens}")
                
            logger.info(f"Initialized language model: {model or os.getenv('MODEL_NAME')}")
        except Exception as e:
            logger.error(f"Error initializing language model: {e}")
            raise
        
        # Initialize available tools
        self._load_tools()
    
    def _load_tools(self):
        """Load tools based on configuration."""
        self._available_tools = {}
        
        # Import standard tools
        try:
            from llm_ai_agent.tools import get_all_tools
            
            # Add basic tools that should always be available
            self._available_tools.update(get_all_tools())
            
            logger.info("Loaded basic tools")
        except ImportError as e:
            logger.warning(f"Could not import basic tools: {e}")
        
        # Load hardware tools if hardware is available
        if self.hardware:
            try:
                # Add camera tools
                self._available_tools["capture_environment"] = self.hardware.camera_tools.capture_environment
                self._available_tools["capture_wrist"] = self.hardware.camera_tools.capture_wrist
                self._available_tools["analyze_image"] = self.hardware.camera_tools.analyze_image
                
                # Add speaker tools
                self._available_tools["speak"] = self.hardware.speaker_tools.speak
                self._available_tools["is_speaking"] = self.hardware.speaker_tools.is_speaking
                self._available_tools["stop_speaking"] = self.hardware.speaker_tools.stop_speaking
                
                # Add robotic arm tools
                self._available_tools["move_home"] = self.hardware.arm_tools.move_home
                self._available_tools["move_position"] = self.hardware.arm_tools.move_position
                self._available_tools["grasp"] = self.hardware.arm_tools.grasp
                self._available_tools["release"] = self.hardware.arm_tools.release
                self._available_tools["get_position"] = self.hardware.arm_tools.get_position
                self._available_tools["move_default"] = self.hardware.arm_tools.move_default
                
                logger.info("Loaded hardware tools")
            except Exception as e:
                logger.error(f"Error loading hardware tools: {e}")
        
        # Load tools from configuration
        tool_config = self.config.get('tools', {})
        if tool_config:
            logger.info("Processing tool configuration")
            
            # Process tool inclusions/exclusions based on configuration
            categories = tool_config.get('categories', [])
            includes = tool_config.get('include', [])
            excludes = tool_config.get('exclude', [])
            
            # Here we would process categories to include/exclude tools
            # For now, we'll just log the configuration
            if categories:
                logger.info(f"Tool categories: {categories}")
            if includes:
                logger.info(f"Tools to include: {includes}")
            if excludes:
                logger.info(f"Tools to exclude: {excludes}")
                # Remove excluded tools
                for tool_name in excludes:
                    if tool_name in self._available_tools:
                        del self._available_tools[tool_name]
                        logger.info(f"Excluded tool: {tool_name}")
            
        logger.info(f"Loaded {len(self._available_tools)} tools: {', '.join(self._available_tools.keys())}")
    
    def _format_tools_for_prompt(self) -> str:
        """
        Format the available tools information for inclusion in the system prompt.
        
        Returns:
            String with formatted tool information
        """
        tools_info = ""
        for name, func in self._available_tools.items():
            doc = func.__doc__ or "No description available."
            # Extract first line without using split with backslash
            first_line = doc.strip()
            if "\n" in first_line:
                first_line = first_line[:first_line.find("\n")]
            tools_info += f"- {name}: {first_line}" + "\n"
        return tools_info
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract structured data.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            Parsed response as a dictionary
        """
        try:
            # Guard against None responses
            if response is None:
                logger.warning("Received None response from LLM")
                return {"response": "No response received from language model."}
                
            # If the response is empty
            if not response or response.strip() == "":
                logger.warning("Received empty response from LLM")
                return {"response": ""}
                
            # Extract JSON from the response using different approaches
            
            # Approach 1: Look for JSON in a code block (```json ... ```)
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    parsed = json.loads(json_str)
                    return parsed
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON from code block: {json_str}")
                    # Continue to other approaches
            
            # Approach 2: Look for JSON with curly braces (maybe multiple objects)
            json_matches = re.findall(r'({.*?})', response, re.DOTALL)
            for potential_json in json_matches:
                try:
                    parsed = json.loads(potential_json)
                    if isinstance(parsed, dict) and ('response' in parsed or 'action' in parsed):
                        return parsed
                except json.JSONDecodeError:
                    continue  # Try the next match
            
            # Approach 3: Look for tool calls in specific formats like: use_tool(parameters)
            tool_match = re.search(r'use_(\w+)\(([^)]*)\)', response)
            if tool_match:
                tool_name = tool_match.group(1)
                tool_input = tool_match.group(2).strip()
                return {
                    "action": tool_name,
                    "action_input": tool_input,
                    "response": f"Using {tool_name} tool with input: {tool_input}"
                }
            
            # Approach 4: Look for tool names mentioned with parameters
            tool_pattern = r'(?:use|using|call|execute|run|invoke)(?:\s+the)?\s+(\w+)(?:\s+tool)?(?:\s+with)?[:\s]+["\'`]?(.*?)["\'`]?(?:\.|$)'
            tool_match = re.search(tool_pattern, response, re.IGNORECASE)
            if tool_match:
                tool_name = tool_match.group(1).lower()
                tool_input = tool_match.group(2).strip()
                
                # Check if this is a known tool
                if tool_name in self._available_tools:
                    return {
                        "action": tool_name,
                        "action_input": tool_input,
                        "response": f"Using {tool_name} tool with input: {tool_input}"
                    }
            
            # If all approaches fail, return the original response
            logger.warning("Could not extract JSON from response")
            return {"response": response}
                
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {"response": response}
    
    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """
        Execute a tool with the given input.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Input for the tool
            
        Returns:
            Result of the tool execution
        """
        if tool_name not in self._available_tools:
            return f"Error: Tool '{tool_name}' not found. Available tools: {', '.join(self._available_tools.keys())}"
        
        try:
            tool_func = self._available_tools[tool_name]
            result = tool_func(tool_input)
            return result
        except Exception as e:
            return f"Error executing tool {tool_name}: {str(e)}"
    
    def process(
        self, 
        user_input: str, 
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Process a user input and return the agent's response.
        
        Args:
            user_input: The user's question or command
            chat_history: Optional list of previous conversation messages
            max_iterations: Maximum number of tool execution iterations
            
        Returns:
            The agent's response as a dictionary
        """
        try:
            if chat_history is None:
                chat_history = []
            
            # Initialize messages list for the conversation
            messages = [
                {"role": "system", "content": self._system_prompt}
            ]
            
            # Add chat history
            for message in chat_history:
                messages.append(message)
            
            # Add the user's input
            messages.append({"role": "user", "content": user_input})
            
            # Initialize variables for the tool execution loop
            iterations = 0
            final_response = None
            
            # Execute tools and continue the conversation until we get a final response
            while iterations < max_iterations and final_response is None:
                # Get response from LLM
                if self.verbose:
                    logger.info(f"Iteration {iterations+1}: Requesting response from LLM")
                
                llm_response = self.llm.invoke(messages)
                llm_content = llm_response.content
                
                if self.verbose:
                    logger.info(f"LLM response: {llm_content}")
                
                # Parse the response
                parsed_response = self._parse_response(llm_content)
                
                # Add the assistant's message to the conversation
                messages.append({"role": "assistant", "content": llm_content})
                
                # Check if we need to execute a tool
                if "action" in parsed_response and "action_input" in parsed_response:
                    tool_name = parsed_response["action"]
                    tool_input = parsed_response["action_input"]
                    
                    if self.verbose:
                        logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
                    
                    # Execute the tool
                    tool_result = self._execute_tool(tool_name, tool_input)
                    
                    if self.verbose:
                        logger.info(f"Tool result: {tool_result}")
                    
                    # Add the tool result to the conversation
                    messages.append({"role": "user", "content": f"Tool result: {tool_result}\n\nContinue with your analysis or provide a final response."})
                    
                    iterations += 1
                else:
                    # If no tool execution is needed, we have our final response
                    final_response = parsed_response.get("response", llm_content)
                    if self.verbose:
                        logger.info(f"Final response: {final_response}")
            
            # If we've hit the maximum number of iterations without a final response,
            # ask the LLM for a final response
            if final_response is None:
                messages.append({"role": "user", "content": "You've used the maximum number of tool calls. Please provide your final response to the user."})
                llm_response = self.llm.invoke(messages)
                final_response = llm_response.content
            
            return {
                "output": final_response,
                "chat_history": messages[1:]  # Exclude the system prompt
            }
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return {"output": f"Error: {str(e)}"}
    
    def process_to_string(
        self, 
        user_input: str, 
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_iterations: int = 5
    ) -> str:
        """
        Process a user input and return the agent's response as a string.
        
        Args:
            user_input: The user's question or command
            chat_history: Optional list of previous conversation messages
            max_iterations: Maximum number of tool execution iterations
            
        Returns:
            The agent's response as a string
        """
        response = self.process(user_input, chat_history, max_iterations)
        
        # Extract the output string from the response
        if isinstance(response, dict) and "output" in response:
            return response["output"]
        elif isinstance(response, str):
            return response
        return "Unable to generate a response."
    
    @property
    def verbose(self) -> bool:
        """Get the verbose setting."""
        return self._verbose
    
    @verbose.setter
    def verbose(self, value: bool) -> None:
        """Set the verbose setting."""
        self._verbose = value
    
    def __del__(self):
        """Clean up resources when the agent is deleted."""
        if hasattr(self, 'hardware') and self.hardware:
            try:
                self.hardware.close()
                logger.info("Hardware connections closed")
            except Exception as e:
                logger.error(f"Error closing hardware connections: {e}")


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