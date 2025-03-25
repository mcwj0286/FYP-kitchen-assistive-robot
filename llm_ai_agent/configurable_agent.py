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
import time
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
    
    Features:
    - Configuration-driven behavior via YAML files
    - Support for multiple LLM providers through OpenRouter
    - Automatic tool calling when supported by the model
    - Hardware integration for robotics applications
    - Multimodal capabilities with image input
    
    The agent can be configured to automatically capture images from connected cameras
    with each query, enabling visual context for the LLM's responses.
    """
    
    def __init__(
        self, 
        agent_type: str = "base_agent", 
        config_path: Optional[str] = None,
        verbose: bool = True,
        model_name: Optional[str] = None,
        use_hardware: bool = True,
        capture_image: str = ""
    ):
        """
        Initialize a configurable agent.
        
        Args:
            agent_type: Type of agent to create (must match a configuration file)
            config_path: Optional path to a custom configuration file
            verbose: Whether to enable verbose logging
            model_name: Model name to use (overrides config)
            use_hardware: Whether to use real hardware (False uses mock implementations)
            capture_image: Which camera view to capture with each prompt ["environment", "wrist", "both", ""]
        """
        self.agent_type = agent_type
        self._verbose = verbose
        self._use_hardware = use_hardware
        self._capture_image = capture_image
        
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
        self._tool_prompts = {}
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
        
        # Get hardware component configurations
        hardware_config = self.config.get('hardware', {})
        enable_camera = hardware_config.get('enable_camera', True)
        enable_speaker = hardware_config.get('enable_speaker', True)
        enable_arm = hardware_config.get('enable_arm', True)
        
        # Get image capture configuration if not explicitly provided at initialization
        if self._capture_image == "":
            capture_image = hardware_config.get('capture_image', "")
            # Only set if the value is valid
            if capture_image in ["environment", "wrist", "both", ""]:
                self._capture_image = capture_image
                if self._verbose and capture_image:
                    logger.info(f"Using configuration capture_image: {capture_image}")
        
        # Set up hardware if required and available
        if hardware_required and HARDWARE_AVAILABLE and self._use_hardware:
            try:
                self.hardware = HardwareTools(
                    use_mock=not self._use_hardware,
                    enable_camera=enable_camera,
                    enable_speaker=enable_speaker,
                    enable_arm=enable_arm
                )
                logger.info("Hardware tools initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing hardware tools: {e}")
                self.hardware = None
                hardware_required = False
        
        # Log hardware configuration
        if hardware_required and HARDWARE_AVAILABLE and self._use_hardware:
            logger.info("Hardware enabled with configuration:")
            logger.info(f"  - Camera: {'enabled' if enable_camera else 'disabled'}")
            logger.info(f"  - Speaker: {'enabled' if enable_speaker else 'disabled'}")
            logger.info(f"  - Robotic Arm: {'enabled' if enable_arm else 'disabled'}")
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
        self._tool_prompts = {}
        
        # Define a list of camera capture tools that should never be available to the LLM
        # We want these tools to be used only through the capture_image parameter
        # This ensures that all image capture goes through the predefined agent configuration
        # rather than allowing the LLM to call these tools directly
        CAMERA_CAPTURE_TOOLS = ["capture_environment", "capture_wrist", "capture_both"]
        
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
                # Add camera tools if camera is enabled, but exclude direct capture tools
                if self.hardware.camera_tools:
                    # Only add analyze_image, not the capture tools
                    self._available_tools["analyze_image"] = self.hardware.camera_tools.analyze_image
                    logger.info("Loaded camera analysis tools (capture tools excluded)")
                
                # Add speaker tools if speaker is enabled
                if self.hardware.speaker_tools:
                    self._available_tools["speak"] = self.hardware.speaker_tools.speak
                    self._available_tools["is_speaking"] = self.hardware.speaker_tools.is_speaking
                    self._available_tools["stop_speaking"] = self.hardware.speaker_tools.stop_speaking
                    logger.info("Loaded speaker tools")
                
                # Add robotic arm tools if arm is enabled
                if self.hardware.arm_tools:
                    self._available_tools["move_home"] = self.hardware.arm_tools.move_home
                    self._available_tools["move_position"] = self.hardware.arm_tools.move_position
                    self._available_tools["grasp"] = self.hardware.arm_tools.grasp
                    self._available_tools["release"] = self.hardware.arm_tools.release
                    self._available_tools["get_position"] = self.hardware.arm_tools.get_position
                    self._available_tools["move_default"] = self.hardware.arm_tools.move_default
                    logger.info("Loaded robotic arm tools")
                
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
            
            # Add camera capture tools to the excludes list
            for tool_name in CAMERA_CAPTURE_TOOLS:
                if tool_name not in excludes:
                    excludes.append(tool_name)
            
            # Load tool prompts from tool configuration files
            self._load_tool_prompts()
            
            # Here we would process categories to include/exclude tools
            # For now, we'll just log the configuration
            if categories:
                logger.info(f"Tool categories: {categories}")
            if includes:
                logger.info(f"Tools to include: {includes}")
            
            logger.info(f"Tools to exclude: {excludes}")
            # Remove excluded tools
            for tool_name in excludes:
                if tool_name in self._available_tools:
                    del self._available_tools[tool_name]
                    logger.info(f"Excluded tool: {tool_name}")
        
        # Final check to ensure camera capture tools are never available
        for tool_name in CAMERA_CAPTURE_TOOLS:
            if tool_name in self._available_tools:
                del self._available_tools[tool_name]
                logger.info(f"Forcibly excluded camera capture tool: {tool_name}")
        
        logger.info(f"Loaded {len(self._available_tools)} tools: {', '.join(self._available_tools.keys())}")
    
    def _load_tool_prompts(self):
        """Load tool prompts from configuration files."""
        try:
            # Get the tool configuration directory
            config_path = self.config_loader.config_dir
            tools_config_dir = os.path.join(config_path, 'tools')
            
            # Check if the directory exists
            if not os.path.isdir(tools_config_dir):
                logger.warning(f"Tool configuration directory not found: {tools_config_dir}")
                return
            
            # Import yaml
            import yaml
            
            # Load all tool configuration files
            for filename in os.listdir(tools_config_dir):
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    file_path = os.path.join(tools_config_dir, filename)
                    
                    try:
                        with open(file_path, 'r') as f:
                            tool_config = yaml.safe_load(f)
                            
                            # Check if the file contains tool prompts
                            if 'tool_prompts' in tool_config:
                                # Load tool prompts
                                for tool_name, prompt_info in tool_config['tool_prompts'].items():
                                    self._tool_prompts[tool_name] = prompt_info
                                    
                                logger.info(f"Loaded tool prompts from {filename}")
                    except Exception as e:
                        logger.error(f"Error loading tool configuration from {filename}: {e}")
        
        except Exception as e:
            logger.error(f"Error loading tool prompts: {e}")
    
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
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """
        Execute a tool with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool
            
        Returns:
            Result of the tool execution
        """
        if tool_name not in self._available_tools:
            return {"error": f"Tool '{tool_name}' not found"}
        
        try:
            # Get the tool function
            tool_func = self._available_tools[tool_name]
            
            # Handle parameter name mapping for specific tools
            if tool_name == "calculator":
                # Map 'expression' to 'input_str'
                if "expression" in parameters and "input_str" not in parameters:
                    parameters = {"input_str": parameters.get("expression")}
            
            # NOTE: No mapping for text_processor - it already expects 'text' parameter
            
            # Execute the tool with parameters
            result = tool_func(**parameters)
            
            # Return the result
            return result
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}")
            return {"error": f"Error executing tool: {str(e)}"}
    
    def process(self, user_input: str, history=None) -> Dict[str, Any]:
        """
        Process user input and generate a response.
        
        Args:
            user_input: The user's input text
            history: Optional conversation history
            
        Returns:
            Dictionary containing the response and any additional data
        """
        if history is None:
            history = []
        
        # Ensure history is in the right format
        _history = self._format_history(history)
        
        # Check if we should capture an image
        # The capture_image parameter can be:
        # - "environment": Capture from the environment camera only
        # - "wrist": Capture from the wrist-mounted camera only
        # - "both": Capture from both cameras and send both images
        # - "": Disable image capture (default)
        image_data_uris = []
        if self._capture_image and self.hardware and self.hardware.camera_tools:
            try:
                if self._verbose:
                    logger.info(f"Capturing image from {self._capture_image} camera")
                
                # Capture the requested camera view
                if self._capture_image == "environment":
                    result = self.hardware.camera_tools.capture_environment()
                    if isinstance(result, dict) and "image" in result:
                        image_data_uris.append({
                            "url": result["image"],
                            "description": f"Environment camera: {result.get('description', '')}"
                        })
                        logger.info("Environment image captured successfully")
                elif self._capture_image == "wrist":
                    result = self.hardware.camera_tools.capture_wrist()
                    if isinstance(result, dict) and "image" in result:
                        image_data_uris.append({
                            "url": result["image"],
                            "description": f"Wrist camera: {result.get('description', '')}"
                        })
                        logger.info("Wrist image captured successfully")
                elif self._capture_image == "both":
                    result = self.hardware.camera_tools.capture_both()
                    if isinstance(result, dict):
                        # Handle environment camera
                        if result.get("environment") and isinstance(result["environment"], dict) and "image" in result["environment"]:
                            image_data_uris.append({
                                "url": result["environment"]["image"],
                                "description": f"Environment camera: {result['environment'].get('description', '')}"
                            })
                        
                        # Handle wrist camera
                        if result.get("wrist") and isinstance(result["wrist"], dict) and "image" in result["wrist"]:
                            image_data_uris.append({
                                "url": result["wrist"]["image"],
                                "description": f"Wrist camera: {result['wrist'].get('description', '')}"
                            })
                        
                        if image_data_uris:
                            logger.info(f"Captured {len(image_data_uris)} camera images successfully")
                        else:
                            logger.warning("No images captured from 'both' camera option")
                else:
                    logger.warning(f"Invalid capture_image value: {self._capture_image}")
            except Exception as e:
                logger.error(f"Error capturing image: {e}")
        
        # Create user message - either text-only or multimodal with image(s)
        if image_data_uris:
            user_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_input
                    }
                ]
            }
            
            # Add each image to the content
            for image_data in image_data_uris:
                user_message["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_data["url"],
                        "detail": "high"
                    }
                })
                
                # If the first image has a description, add it to the user input
                if "description" in image_data and image_data_uris.index(image_data) == 0:
                    user_message["content"][0]["text"] = f"{user_input}\n\nImage source: {image_data['description']}"
                
                # For additional images, add a note about them in the text
                if len(image_data_uris) > 1 and image_data_uris.index(image_data) > 0:
                    user_message["content"][0]["text"] += f"\nAdditional image: {image_data['description']}"
        else:
            user_message = {"role": "user", "content": user_input}
        
        # Add the user input to the conversation
        messages = _history + [user_message]
        
        # Define the response format for structured outputs
        response_format = {
            "thought": "Your reasoning process (not shown to the user)",
            "reply": "Your response to the user",
            "tool_calls": [
                {
                    "tool_name": "ToolName",
                    "parameters": {
                        "param1": "value1",
                        "param2": "value2"
                    }
                }
            ]
        }
        
        # Construct the initial prompt with system message and available tools
        messages = self._construct_prompt(messages)
        
        try:
            # Print the input to the first API call
            print("\n=======================first api call========")
            print(json.dumps(messages, indent=2))
            print("==========================================\n")
            
            # Get the initial response from the language model
            initial_response = self.llm.invoke(messages)
            
            # Print the response from the first API call
            print("\n=======================first api response========")
            print(initial_response.content)
            print("==========================================\n")
            
            # Extract structured response (thought, reply, tool_calls)
            structured_response = self._extract_structured_response(initial_response.content)
            
            # Log the structured response
            if self._verbose:
                logger.info(f"Initial structured response: {json.dumps(structured_response, indent=2)}")
            
            # Check if there are tool calls
            if structured_response.get("tool_calls", []):
                tool_results = []
                
                # Process each tool call
                for tool_call in structured_response["tool_calls"]:
                    tool_name = tool_call.get("tool_name")
                    parameters = tool_call.get("parameters", {})
                    
                    # Generate a unique call ID
                    call_id = f"{tool_name}_{int(time.time())}"
                    
                    # Execute the tool
                    tool_result = self._execute_tool(tool_name, parameters)
                    
                    # Extract any image data from the tool result
                    image_data = None
                    if isinstance(tool_result, dict):
                        # Handle direct image in result
                        if "image" in tool_result:
                            image_data = {
                                "url": tool_result.pop("image"),
                                "description": tool_result.get("description", f"{tool_name} result image")
                            }
                        # Handle environment/wrist structure for 'both' image capture
                        elif "environment" in tool_result and isinstance(tool_result["environment"], dict) and "image" in tool_result["environment"]:
                            env_image = tool_result["environment"].pop("image")
                            wrist_image = None
                            if "wrist" in tool_result and isinstance(tool_result["wrist"], dict) and "image" in tool_result["wrist"]:
                                wrist_image = tool_result["wrist"].pop("image")
                            
                            # For multiple images, we'll use the first one in the response and log about the others
                            image_data = {
                                "url": env_image,
                                "description": "Environment camera image"
                            }
                            if wrist_image:
                                logger.info("Multiple images in tool result, using environment camera image for response")
                    
                    # Format the tool response
                    tool_response = {
                        "call_id": call_id,
                        "tool_name": tool_name,
                        "data": tool_result
                    }
                    
                    # Add image data if present
                    if image_data:
                        tool_response["image"] = image_data
                    
                    tool_results.append(tool_response)
                    
                    if self._verbose:
                        logger.info(f"Tool response: {json.dumps(tool_response, indent=2)}")
                
                # Add the initial response to conversation
                messages.append({
                    "role": "assistant", 
                    "content": json.dumps(structured_response)
                })
                
                # Add the tool responses to conversation
                tool_response_content = {
                    "tool_responses": tool_results,
                    "instructions": "Please analyze these tool results and provide a helpful response to the user. Make sure to include relevant information from the tool results. Even if there was an error with the tool, provide the best response you can."
                }
                
                # Extract image data from tool responses to include in the next message
                images_from_tools = []
                for result in tool_results:
                    if "image" in result and isinstance(result["image"], dict) and "url" in result["image"]:
                        images_from_tools.append(result["image"])
                
                # Add image URLs to the tool response content if available
                if images_from_tools:
                    tool_response_content["images"] = images_from_tools
                
                tool_response_message = {
                    "role": "user", 
                    "content": json.dumps(tool_response_content)
                }
                messages.append(tool_response_message)
                
                # Get the final response after processing tool results
                # Create a new messages list without the system message
                user_messages = [msg for msg in messages if msg.get("role") != "system"]
                final_messages = self._construct_prompt(user_messages)
                
                # Print the input to the second API call
                print("\n=======================second api call========")
                print(json.dumps(final_messages, indent=2))
                print("==========================================\n")
                
                final_response = self.llm.invoke(final_messages)
                
                # Print the response from the second API call
                print("\n=======================second api response========")
                print(final_response.content)
                print("==========================================\n")
                
                # Extract the final structured response
                final_structured = self._extract_structured_response(final_response.content)
                
                # If the final structured response is empty or doesn't have a reply, create a fallback
                if not final_structured.get("reply"):
                    # Create a fallback response that includes tool information
                    has_errors = any("error" in result.get("data", {}) if isinstance(result.get("data"), dict) else False 
                                    for result in tool_results)
                    
                    if has_errors:
                        final_structured["reply"] = "I encountered an error when trying to use the tool. " + \
                                                   "Please check your request and try again."
                    else:
                        # Create a response that includes the tool results
                        tool_data = []
                        for result in tool_results:
                            data = result.get("data", "No data")
                            tool_name = result.get("tool_name", "tool")
                            tool_data.append(f"Result from {tool_name}: {data}")
                        
                        final_structured["reply"] = "Here are the results: " + " ".join(tool_data)
                
                # Add the final response to conversation for history
                messages.append({
                    "role": "assistant", 
                    "content": json.dumps(final_structured)
                })
                
                # Return the final structured response
                return {
                    "output": final_structured.get("reply", ""),
                    "thought": final_structured.get("thought", ""),
                    "tool_calls": structured_response.get("tool_calls", []),
                    "tool_results": tool_results,
                    "final_response": final_structured
                }
            else:
                # No tool calls, return the initial response
                messages.append({
                    "role": "assistant", 
                    "content": json.dumps(structured_response)
                })
                
                return {
                    "output": structured_response.get("reply", ""),
                    "thought": structured_response.get("thought", ""),
                    "tool_calls": [],
                    "tool_results": [],
                    "final_response": structured_response
                }
        
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return {"output": f"I encountered an error: {str(e)}"}
    
    def _extract_structured_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extract a structured response from the LLM output.
        Attempts to parse JSON, falls back to generating structured format if not valid JSON.
        
        Args:
            response_text: Raw text response from the LLM
            
        Returns:
            Dictionary with thought, reply, and tool_calls
        """
        # First check if the response contains a code block with JSON
        code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
        if code_block_match:
            # If we found a code block, try to parse its contents as JSON
            json_str = code_block_match.group(1)
            try:
                parsed_json = json.loads(json_str)
                # Ensure the JSON has the expected structure
                if isinstance(parsed_json, dict):
                    # Make sure all required fields exist
                    parsed_json["thought"] = parsed_json.get("thought", "")
                    parsed_json["reply"] = parsed_json.get("reply", "")
                    parsed_json["tool_calls"] = parsed_json.get("tool_calls", [])
                    return parsed_json
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                if self._verbose:
                    logger.warning(f"Failed to parse JSON from code block: {e}")
        
        # If code block extraction failed, try to parse the whole response as JSON
        try:
            # Try to parse as JSON first
            response_json = json.loads(response_text)
            
            # Check if it has the expected keys
            if isinstance(response_json, dict) and ("thought" in response_json or "reply" in response_json):
                # Ensure all expected keys exist
                response_json["thought"] = response_json.get("thought", "")
                response_json["reply"] = response_json.get("reply", "")
                response_json["tool_calls"] = response_json.get("tool_calls", [])
                return response_json
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        
        # If not valid JSON or doesn't have the right format, extract manually
        
        # Look for tool call patterns using regex
        tool_calls = []
        tool_pattern = r"Tool:\s*(\w+)\s*\(([^)]*)\)"
        for match in re.finditer(tool_pattern, response_text):
            tool_name = match.group(1)
            params_text = match.group(2)
            
            # Parse parameters (simple key=value format)
            params = {}
            for param in params_text.split(','):
                if '=' in param:
                    key, value = param.split('=', 1)
                    params[key.strip()] = value.strip()
            
            tool_calls.append({
                "tool_name": tool_name,
                "parameters": params
            })
        
        # Split text into thought and reply
        if "Thought:" in response_text and "Reply:" in response_text:
            thought_match = re.search(r"Thought:(.*?)(?=Reply:|$)", response_text, re.DOTALL)
            reply_match = re.search(r"Reply:(.*?)(?=Tool:|$)", response_text, re.DOTALL)
            
            thought = thought_match.group(1).strip() if thought_match else ""
            reply = reply_match.group(1).strip() if reply_match else response_text
        else:
            # No explicit thought/reply structure, treat whole text as reply
            thought = ""
            reply = response_text
        
        return {
            "thought": thought,
            "reply": reply,
            "tool_calls": tool_calls
        }
    
    def _construct_prompt(self, conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Construct a prompt for the language model.
        
        Args:
            conversation: The conversation history
            
        Returns:
            A list of messages for the language model
        """
        messages = []
        
        # Add system message with structured response format instructions
        system_message = f"""{self._system_prompt}

To use tools, respond in the following JSON format:
{{
  "thought": "Your reasoning process (not shown to the user)",
  "reply": "Your response to the user",
  "tool_calls": [
    {{
      "tool_name": "ToolName",
      "parameters": {{
        "param1": "value1",
        "param2": "value2"
      }}
    }}
  ]
}}

Available tools:
"""
        
        # Add available tools to the system message using tool prompts if available
        for tool_name in self._available_tools.keys():
            if tool_name in self._tool_prompts:
                # Get tool prompt information
                tool_prompt = self._tool_prompts[tool_name]
                description = tool_prompt.get('description', '')
                parameters = tool_prompt.get('parameters', [])
                example = tool_prompt.get('example', '')
                
                # Format parameters
                param_names = [param.get('name', '') for param in parameters]
                param_str = ", ".join(param_names)
                
                # Add to system message
                system_message += f"- {tool_name}: {description} [Parameters: {param_str}]\n"
                if example:
                    system_message += f"  Example: {example}\n"
                system_message += "\n"
            else:
                # If no tool prompt is available, use basic format
                func = self._available_tools[tool_name]
                doc = func.__doc__ or "No description available."
                first_line = doc.strip()
                if "\n" in first_line:
                    first_line = first_line[:first_line.find("\n")]
                system_message += f"- {tool_name}: {first_line}\n"
        
        # Add additional instructions for final responses
        system_message += """
When tool results are provided, analyze them and provide a final response in the same JSON format.
Make your final responses comprehensive and include information from the tool results.

IMPORTANT: Use the EXACT parameter names specified above for each tool.
"""
        
        messages.append({"role": "system", "content": system_message})
        
        # Add conversation history
        for message in conversation:
            messages.append(message)
        
        return messages
    
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
        response = self.process(user_input, chat_history)
        
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
    
    @property
    def capture_image(self) -> str:
        """Get the capture_image setting."""
        return self._capture_image
    
    @capture_image.setter
    def capture_image(self, value: str) -> None:
        """
        Set the capture_image setting.
        
        Args:
            value: The camera view to capture ("environment", "wrist", "both", or "" for none)
        """
        if value not in ["environment", "wrist", "both", ""]:
            logger.warning(f"Invalid capture_image value: {value}. Using empty string (disabled).")
            self._capture_image = ""
        else:
            self._capture_image = value
            if self._verbose:
                if value:
                    logger.info(f"Image capture enabled with camera: {value}")
                else:
                    logger.info("Image capture disabled")
    
    def print_system_prompt(self) -> None:
        """
        Print the complete system prompt that will be sent to the LLM.
        This includes the base system prompt from configuration plus
        all tool descriptions and usage instructions.
        """
        # Create the system message with the same format as in _construct_prompt
        system_message = f"""{self._system_prompt}

To use tools, respond in the following JSON format:
{{
  "thought": "Your reasoning process (not shown to the user)",
  "reply": "Your response to the user",
  "tool_calls": [
    {{
      "tool_name": "ToolName",
      "parameters": {{
        "param1": "value1",
        "param2": "value2"
      }}
    }}
  ]
}}

Available tools:
"""
        
        # Add available tools to the system message using tool prompts if available
        for tool_name in self._available_tools.keys():
            if tool_name in self._tool_prompts:
                # Get tool prompt information
                tool_prompt = self._tool_prompts[tool_name]
                description = tool_prompt.get('description', '')
                parameters = tool_prompt.get('parameters', [])
                example = tool_prompt.get('example', '')
                
                # Format parameters
                param_names = [param.get('name', '') for param in parameters]
                param_str = ", ".join(param_names)
                
                # Add to system message
                system_message += f"- {tool_name}: {description} [Parameters: {param_str}]\n"
                if example:
                    system_message += f"  Example: {example}\n"
                system_message += "\n"
            else:
                # If no tool prompt is available, use basic format
                func = self._available_tools[tool_name]
                doc = func.__doc__ or "No description available."
                first_line = doc.strip()
                if "\n" in first_line:
                    first_line = first_line[:first_line.find("\n")]
                system_message += f"- {tool_name}: {first_line}\n"
        
        # Add additional instructions for final responses
        system_message += """
When tool results are provided, analyze them and provide a final response in the same JSON format.
Make your final responses comprehensive and include information from the tool results.

IMPORTANT: Use the EXACT parameter names specified above for each tool.
"""
        
        # Print the complete system prompt
        print("\n=== SYSTEM PROMPT ===\n")
        print(system_message)
        print("\n====================\n")
        
        # Return the formatted system message in case it's needed
        return system_message
    
    def __del__(self):
        """Clean up resources when the agent is deleted."""
        if hasattr(self, 'hardware') and self.hardware:
            try:
                self.hardware.close()
                logger.info("Hardware connections closed")
            except Exception as e:
                logger.error(f"Error closing hardware connections: {e}")
    
    def _format_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Format conversation history into the required format.
        
        Args:
            history: Raw conversation history
            
        Returns:
            Formatted conversation history
        """
        formatted_history = []
        
        for message in history:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role in ["user", "assistant", "system"]:
                formatted_history.append({"role": role, "content": content})
            else:
                # Default unknown roles to user
                formatted_history.append({"role": "user", "content": content})
        
        return formatted_history


# Example usage
if __name__ == "__main__":
    # Create a configurable agent with the base configuration
    # agent = ConfigurableAgent(agent_type="base_agent", verbose=True)
    
    # # Test a simple query
    # response = agent.process_to_string("What is 15 multiplied by 32?")
    # print(f"Base agent response: {response}\n")
    
    # # Create a kitchen assistant agent
    # kitchen_agent = ConfigurableAgent(agent_type="kitchen_assistant", verbose=True)
    
    # # Test a kitchen-related query
    # response = kitchen_agent.process_to_string("Can you analyze this recipe: 2 cups flour, 1 cup sugar, 3 eggs?")
    # print(f"Kitchen agent response: {response}")
    
    # Create a vision-enabled kitchen assistant agent
    vision_agent = ConfigurableAgent(
        agent_type="vision_agent", 
        verbose=True,
        capture_image="environment"  # Automatically capture environment images
    )
    
    vision_agent.print_system_prompt()
    
    # # Print the full system prompt for the vision agent
    # print("\nPrinting the full system prompt for the vision agent:")
    # vision_agent.print_system_prompt()
    
    # print("vison test 1")
    # # Test a vision query
    # response = vision_agent.process_to_string("What do you see in the image? ")
    # print(f"Vision agent response: {response}")
    
    # print("vison test 2")
    # # Change the camera view
    # vision_agent.capture_image = "wrist"
    # response = vision_agent.process_to_string("what do you see in the image?")
    # print(f"Updated vision agent response: {response}")
    
    # print("vison test 3")
    # vision_agent.capture_image = "both"
    # response = vision_agent.process_to_string("what do you see in the image?")
    # print(f"Text-only agent response: {response}")