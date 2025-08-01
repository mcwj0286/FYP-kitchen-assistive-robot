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
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
DEBUG = os.getenv("DEBUG", "false") == "true"
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

class LLM:
    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: int = 3096, verbose: bool = True):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._verbose = verbose
    def invoke(self, messages: List[Dict[str, Any]]) -> Any:
        """
        Directly calls the OpenRouter API with the given messages, supporting both text and images.
        
        Args:
            messages: A list of message objects in the format [{role: "system", content: "..."}, {role: "user", content: [...]}, ...]
            
        Returns:
            A response object with a content attribute containing the LLM's response text
        """
        import requests
        
        
        # API endpoint and auth
        api_endpoint = "https://openrouter.ai/api/v1/chat/completions"
        api_key = os.getenv('OPENROUTER_API_KEY')
        
        if not api_key:
            logger.error("OPENROUTER_API_KEY environment variable is not set")
            class ErrorResponse:
                content = "Error: OpenRouter API key not found. Please set the OPENROUTER_API_KEY environment variable."
            return ErrorResponse()
        
        # Construct the payload
        payload = {
            "model": self.model_name or os.getenv('MODEL_NAME', "anthropic/claude-3-opus-20240229"),
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("YOUR_SITE_URL", "https://example.com"),
            "X-Title": os.getenv("YOUR_SITE_NAME", "AI Assistant"),
        }
        
        try:
            # Make the request to OpenRouter API
            response = requests.post(api_endpoint, headers=headers, json=payload)
            response.raise_for_status()
            
            # Parse the JSON response
            result = response.json()
            
            if result and "choices" in result and len(result["choices"]) > 0:
                output = result["choices"][0]["message"]["content"].strip()
                token_usage = result["usage"]["total_tokens"] if "usage" in result else "n/a"
                
                # Log token usage for monitoring
                if self._verbose:
                    logger.info(f"OpenRouter API call: {token_usage} total tokens used")
                
                # Create a response object with content attribute to match LangChain's output format
                class LLMResponse:
                    def __init__(self, content):
                        self.content = content
                
                return LLMResponse(output)
            else:
                logger.error("No valid response received from OpenRouter API")
                logger.error(result)
                class ErrorResponse:
                    content = "Error: Failed to get valid response from LLM"
                return ErrorResponse()
                
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {e}")
            class ErrorResponse:
                content = f"Error during LLM call: {str(e)}"
            return ErrorResponse()


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
        capture_image: str = "",
        max_tool_iterations: int = 10,
        enable_conversation_logging: bool = False,
        log_directory: Optional[str] = "conversation_logs",
        enable_display: bool = True
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
                           If set to empty string (""), the value will be read from the YAML config file's
                           hardware.capture_image setting. Explicit values passed here take priority over
                           the configuration file.
            max_tool_iterations: Maximum number of tool iterations to allow (for multi-turn tool calling)
            enable_conversation_logging: Whether to log the full conversation to a file
            log_directory: Directory where conversation logs will be stored (default: current working directory)
            enable_display: Whether to open a window to display the camera stream
        """
        self.agent_type = agent_type
        self._verbose = verbose
        self._use_hardware = use_hardware
        self._capture_image = capture_image
        self._max_tool_iterations = max_tool_iterations
        self._enable_conversation_logging = enable_conversation_logging
        self._log_directory = log_directory
        self._enable_display = enable_display
        self.display_thread = None
        self._stop_display = False
        
        # Set up conversation logger if enabled
        if self._enable_conversation_logging:
            self._setup_conversation_logger()
        
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
                from .tools import get_hardware_instance
                self.hardware = get_hardware_instance()
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
            self.llm = LLM(model_name=model, verbose=self._verbose)
            
            # Apply model parameters
            temperature = model_defaults.get('temperature', 0.7)
            max_tokens = model_defaults.get('max_tokens', 3096)
            
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
        
        # Start display thread AFTER hardware and tools are initialized
        if self._enable_display and self.hardware and self.hardware.camera_tools:
            try:
                # Import cv2 here to prevent import errors if OpenCV is not installed
                import cv2
                import threading
                self.open_display_thread()
                logger.info("Camera display thread started")
            except ImportError:
                logger.error("OpenCV (cv2) is required for camera display. Install with: pip install opencv-python")
            except Exception as e:
                logger.error(f"Error starting display thread: {e}")
        elif self._enable_display:
            logger.warning("Display enabled but hardware or camera tools not available")
    
    def _load_tools(self):
        """Load tools based on configuration."""
        self._available_tools = {}
        self._tool_prompts = {}
        
        # Import all tools from unified tools module
        try:
            from llm_ai_agent.tools import get_all_tools
            
            # Get all available tools
            all_tools = get_all_tools()
            
            logger.info("Loaded all tools from unified tools module")
        except ImportError as e:
            logger.warning(f"Could not import tools: {e}")
            all_tools = {}
        
        # Load tools from configuration
        tool_config = self.config.get('tools', {})
        if tool_config:
            logger.info("Processing tool configuration")
            
            # Get the include list from the configuration
            includes = tool_config.get('include', [])
            
            # Load tool prompts from tool configuration files
            self._load_tool_prompts()
            
            # Filter available tools to only include those in the include list
            for tool_name, tool_func in all_tools.items():
                if tool_name in includes:
                    self._available_tools[tool_name] = tool_func
                    logger.info(f"Added tool from configuration: {tool_name}")
        else:
            # If no tool configuration is provided, don't add any tools
            logger.warning("No tool configuration found")
        
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
    
    def open_display_thread(self):
        """
        Open a separate thread to display the camera streams.
        The thread will run until the agent is deleted or stop_display_thread is called.
        """
        import threading
        # Stop any existing display thread
        self.stop_display_thread()
        
        # Reset the stop flag
        self._stop_display = False
        
        # Start a new thread
        self.display_thread = threading.Thread(target=self._display_image)
        self.display_thread.daemon = True  # Make thread a daemon so it exits when main program exits
        self.display_thread.start()
        
        if self._verbose:
            logger.info("Display thread started")
    
    def _display_image(self):
        """
        Continuously display camera frames until stopped.
        This runs in a separate thread to avoid blocking the main program.
        """
        import cv2
        import time
        
        if not self.hardware or not self.hardware.camera_tools:
            logger.error("Hardware or camera tools not available for display")
            return
        
        try:
            # Get the existing camera interface instead of creating a new one
            camera_interface = self.hardware.camera_tools.get_current_frames()
            
            if not camera_interface:
                logger.error("No camera interface available")
                return
            
            # Create a window for each camera
            window_created = False
            
            # Display loop
            while not self._stop_display:
                try:
                    # Get frames from all cameras - use cached frames when possible
                    frames = self.hardware.camera_tools.capture_frames()
                    
                    if not frames:
                        logger.warning("No camera frames received")
                        time.sleep(0.1)
                        continue
                    
                    # Display each camera frame
                    for cam_id, (success, frame) in frames.items():
                        if success and frame is not None:
                            if not window_created:
                                # Create windows on first successful frame
                                cv2.namedWindow(f"Camera {cam_id}", cv2.WINDOW_NORMAL)
                                window_created = True
                            
                            # Display the frame
                            cv2.imshow(f"Camera {cam_id}", frame)
                    
                    # Process any window events (needed for window to be responsive)
                    # Wait for 30ms (about 30 FPS) or until a key is pressed
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break
                        
                    # Short sleep to avoid using too much CPU
                    time.sleep(0.01)
                    
                except Exception as e:
                    logger.error(f"Error in display loop: {e}")
                    time.sleep(0.5)  # Sleep longer on error
        
        except Exception as e:
            logger.error(f"Error in display thread: {e}")
        finally:
            # Clean up
            cv2.destroyAllWindows()
            logger.info("Display thread terminated")
    
    def stop_display_thread(self):
        """
        Stop the display thread if it's running.
        """
        if self.display_thread and self.display_thread.is_alive():
            self._stop_display = True
            # Give the thread time to clean up
            import time
            time.sleep(0.5)
            # Wait for thread to terminate (with timeout)
            self.display_thread.join(timeout=2.0)
            if self.display_thread.is_alive():
                logger.warning("Display thread did not terminate properly")
            else:
                logger.info("Display thread stopped")
            self.display_thread = None
    
    def _capture_current_images(self) -> List[Dict[str, Any]]:
        """
        Capture current images based on the camera configuration.
        
        Returns:
            List of image data dictionaries with url and description
        """
        image_data_uris = []
        if not self._capture_image or not self.hardware or not self.hardware.camera_tools:
            return image_data_uris
            
        try:
            if self._verbose:
                logger.info(f"Capturing image from {self._capture_image} camera")
            
            # Use cached frames when possible by using the camera_tools methods
            # which have been modified to use the cache
            
            # Capture the requested camera view
            if self._capture_image == "environment":
                result = self.hardware.camera_tools.capture_environment()
                if isinstance(result, dict) and "image" in result:
                    image_data_uris.append({
                        "url": result["image"],
                        "description": f"Environment camera: {result.get('description', '')}"
                    })
                    if self._verbose:
                        logger.info("Environment image captured successfully")
            elif self._capture_image == "wrist":
                result = self.hardware.camera_tools.capture_wrist()
                if isinstance(result, dict) and "image" in result:
                    image_data_uris.append({
                        "url": result["image"],
                        "description": f"Wrist camera: {result.get('description', '')}"
                    })
                    if self._verbose:
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
                        if self._verbose:
                            logger.info(f"Captured {len(image_data_uris)} camera images successfully")
                    else:
                        if self._verbose:
                            logger.warning("No images captured from 'both' camera option")
            else:
                if self._verbose:
                    logger.warning(f"Invalid capture_image value: {self._capture_image}")
        except Exception as e:
            logger.error(f"Error capturing image: {e}")
            
        return image_data_uris
        
    def _create_message_with_images(self, content: str, image_data_uris: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a message with text and optional images.
        
        Args:
            content: The text content
            image_data_uris: List of image data dictionaries
            
        Returns:
            A message dictionary with content
        """
        if not image_data_uris:
            return {"role": "user", "content": content}
            
        # Create multimodal message with text and images
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": content
                }
            ]
        }
        
        # First prepare the text content with clear image descriptions
        if len(image_data_uris) > 1:
            # For multiple images, add numbered descriptions
            image_descriptions = "\n\n"
            for i, img_data in enumerate(image_data_uris):
                image_descriptions += f"Image {i+1}: {img_data.get('description', f'Camera view {i+1}')}\n"
            message["content"][0]["text"] += image_descriptions
        elif "description" in image_data_uris[0]:
            # For single image, maintain original format
            message["content"][0]["text"] += f"\n\nImage source: {image_data_uris[0]['description']}"
        
        # Now add each image to the content
        for image_data in image_data_uris:
            message["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": image_data["url"],
                    "detail": "high"
                }
            })
            
        return message

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
        
        # Ensure history is in the right format and strip out old images
        _history = self._format_history(history)
        
        # Capture initial images
        image_data_uris = self._capture_current_images()
        
        # Create user message - either text-only or multimodal with image(s)
        user_message = self._create_message_with_images(user_input, image_data_uris)
        
        # Add the user input to the conversation
        messages = _history + [user_message]
        
        # Construct the initial prompt with system message and available tools
        messages = self._construct_prompt(messages)
        
        # Log the system prompt and user input
        if self._enable_conversation_logging:
            # Find and log the system message
            for msg in messages:
                if msg["role"] == "system":
                    self._log_conversation("system", msg["content"])
            
            # Log the user input
            self._log_conversation("user", user_input)
        
        try:
            # Initialize variables for multi-turn tool calling
            iteration = 0
            final_response = None
            all_tool_results = []
            is_complete = False
            
            # Start the tool calling loop - continues until completion or max iterations
            while not is_complete and iteration < self._max_tool_iterations:
                iteration += 1
                
                if self._verbose:
                    logger.info(f"Starting tool iteration {iteration}/{self._max_tool_iterations}")
                
                # Print the input to the API call if in debug mode
                if DEBUG:
                    print(f"\n=======================api call (iteration {iteration})========")
                    print(json.dumps(messages, indent=2))
                    print("==========================================\n")
                
                # Get the response from the language model for this iteration
                current_response = self.llm.invoke(messages)
                
                # Log the assistant's response
                if self._enable_conversation_logging:
                    self._log_conversation("assistant", current_response.content, iteration)
                
                # Extract structured response including is_complete flag
                structured_response = self._extract_structured_response(current_response.content)
                
                # Log the structured response for this iteration
                if self._verbose:
                    logger.info(f"Iteration {iteration} structured response: {json.dumps(structured_response, indent=2)}")
                #For record demo video purpose, print the reply
                else:
                    if structured_response['is_complete'] == False:
                        print(f"{structured_response['reply']}")
                # Check if this response is marked as complete or has no tool calls
                is_complete = structured_response.get("is_complete", False)
                if not structured_response.get("tool_calls"):
                    is_complete = True
                
                if is_complete:
                    # This is the final response
                    final_response = structured_response
                    if self._verbose:
                        logger.info(f"Tool execution complete at iteration {iteration}")
                    break
                    
                # Process tool calls for this iteration
                iteration_tool_results = []
                
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
                        "parameters": parameters,
                        "data": tool_result
                    }
                    
                    # Add image data if present
                    if image_data:
                        tool_response["image"] = image_data
                    
                    iteration_tool_results.append(tool_response)
                    
                    if self._verbose:
                        logger.info(f"Tool response: {json.dumps(tool_response, indent=2)}")
                
                # Add all tool results from this iteration
                all_tool_results.extend(iteration_tool_results)
                
                # Add the response to the conversation - convert multimodal messages to text-only for history
                assistant_content = json.dumps(structured_response)
                messages.append({
                    "role": "assistant", 
                    "content": assistant_content
                })
                
                # Add the tool responses to conversation
                tool_response_content = {
                    "tool_responses": iteration_tool_results,
                    "iteration": iteration,
                    "max_iterations": self._max_tool_iterations,
                    "instructions": "Analyze these tool results and decide if you have enough information to respond to the user or need to call additional tools."
                }
                
                # Capture fresh images for the next iteration if enabled
                fresh_image_data_uris = self._capture_current_images()
                
                # Create a tool response message with the fresh images
                if fresh_image_data_uris:
                    tool_response_message = self._create_message_with_images(json.dumps(tool_response_content), fresh_image_data_uris)
                else:
                    tool_response_message = {
                        "role": "user", 
                        "content": json.dumps(tool_response_content)
                    }
                
                # Clear any previous image-containing messages from the conversation history
                # We'll keep the text-only messages but replace the full conversation with the freshly captured images
                filtered_messages = []
                for msg in messages:
                    # Keep the system message as is
                    if msg["role"] == "system":
                        filtered_messages.append(msg)
                    # For other messages, extract only text if they contain images
                    elif isinstance(msg["content"], list) and any(item.get("type") == "image_url" for item in msg["content"] if isinstance(item, dict)):
                        # This is a multimodal message - extract only text
                        text_content = ""
                        for item in msg["content"]:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_content += item.get("text", "")
                        
                        # Add message with only the text content
                        filtered_messages.append({"role": msg["role"], "content": text_content})
                    else:
                        # Standard message without images - add as is
                        filtered_messages.append(msg)
                
                # Replace messages with the filtered version and add the new message with fresh images
                messages = filtered_messages
                messages.append(tool_response_message)
                
                # Log the tool responses
                if self._enable_conversation_logging:
                    self._log_conversation("tool", json.dumps(tool_response_content), iteration)
                
                if self._verbose:
                    logger.info(f"Completed iteration {iteration} with {len(iteration_tool_results)} tool results")
            
            # If we exited the loop without setting final_response (reached max iterations)
            if final_response is None:
                if self._verbose:
                    logger.warning(f"Reached maximum iterations ({self._max_tool_iterations}) without completion")
                
                # Capture final set of fresh images
                final_images = self._capture_current_images()
                
                # Create a special prompt asking for a final response
                final_message_content = f"You've reached the maximum number of tool iterations ({self._max_tool_iterations}). Please provide your final response based on the information gathered so far."
                
                # Filter out previous images from the conversation history again
                filtered_messages = []
                for msg in messages:
                    # Keep the system message as is
                    if msg["role"] == "system":
                        filtered_messages.append(msg)
                    # For other messages, extract only text if they contain images
                    elif isinstance(msg["content"], list) and any(item.get("type") == "image_url" for item in msg["content"] if isinstance(item, dict)):
                        # This is a multimodal message - extract only text
                        text_content = ""
                        for item in msg["content"]:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_content += item.get("text", "")
                        
                        # Add message with only the text content
                        filtered_messages.append({"role": msg["role"], "content": text_content})
                    else:
                        # Standard message without images - add as is
                        filtered_messages.append(msg)
                
                # Replace messages with the filtered version
                messages = filtered_messages
                
                # Add the message with fresh images if available
                if final_images:
                    final_message = self._create_message_with_images(final_message_content, final_images)
                else:
                    final_message = {
                        "role": "user",
                        "content": final_message_content
                    }
                
                messages.append(final_message)
                
                # Get the final response
                final_llm_response = self.llm.invoke(messages)
                final_response = self._extract_structured_response(final_llm_response.content)
                final_response["is_complete"] = True  # Force completion
            
            # Add the final response to conversation for history
            messages.append({
                "role": "assistant", 
                "content": json.dumps(final_response)
            })
            
            # Log the final response
            if self._enable_conversation_logging and final_response:
                self._log_conversation("final", final_response)
            
            # Return the comprehensive response with all tool results
            return {
                "output": final_response.get("reply", ""),
                "thought": final_response.get("thought", ""),
                "tool_calls": [tool_call for result in all_tool_results for tool_call in [{"tool_name": result["tool_name"], "parameters": result.get("parameters", {})}]],
                "tool_results": all_tool_results,
                "iterations": iteration,
                "is_complete": final_response.get("is_complete", True),
                "final_response": final_response
            }
        
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            
            # Log the error
            if self._enable_conversation_logging:
                self._log_conversation("error", f"Error processing input: {e}")
            
            return {"output": f"I encountered an error: {str(e)}"}
    
    def _extract_structured_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extract a structured response from the LLM output.
        Attempts to parse JSON, falls back to generating structured format if not valid JSON.
        
        Args:
            response_text: Raw text response from the LLM
            
        Returns:
            Dictionary with thought, reply, tool_calls, and is_complete flag
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
                    parsed_json["plan"] = parsed_json.get("plan", "")
                    parsed_json["reply"] = parsed_json.get("reply", "")
                    parsed_json["tool_calls"] = parsed_json.get("tool_calls", [])
                    parsed_json["is_complete"] = parsed_json.get("is_complete", False)
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
                response_json["plan"] = response_json.get("plan", "")
                response_json["tool_calls"] = response_json.get("tool_calls", [])
                response_json["is_complete"] = response_json.get("is_complete", False)
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
        
        # If there are no tool calls, assume the response is complete
        is_complete = len(tool_calls) == 0
        
        return {
            "thought": thought,
            "reply": reply,
            "tool_calls": tool_calls,
            "is_complete": is_complete
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
  "plan": "Your action plan. It should be updated based on latest observation.",
  "is_complete": false,  // Set to true when you have enough information to respond
  "reply": "Your response to the user. If you are plan to use tools call and the task is not complete, describe your action in short and imcomplete setences.Example: "I'm using calculator ...",
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

You can call multiple tools in sequence across multiple turns to complete complex tasks. 
After each tool result, decide if you:
1. Have enough information to respond to the user (set "is_complete": true)
2. Need to call more tools (provide new "tool_calls" and keep "is_complete": false)

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
  "is_complete": false,  // Set to true when you have enough information to respond
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

You can call multiple tools in sequence across multiple turns to complete complex tasks. 
After each tool result, decide if you:
1. Have enough information to respond to the user (set "is_complete": true)
2. Need to call more tools (provide new "tool_calls" and keep "is_complete": false)

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
        # Stop the display thread if it's running
        if hasattr(self, 'display_thread') and self.display_thread:
            try:
                self.stop_display_thread()
            except Exception as e:
                logger.error(f"Error stopping display thread: {e}")
        
        # Clean up hardware
        if hasattr(self, 'hardware') and self.hardware:
            try:
                self.hardware.close()
                logger.info("Hardware connections closed")
            except Exception as e:
                logger.error(f"Error closing hardware connections: {e}")
    
    def _format_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Format conversation history into the required format.
        Strip out images from previous messages to reduce token usage.
        
        Args:
            history: Raw conversation history
            
        Returns:
            Formatted conversation history with images removed from older messages
        """
        formatted_history = []
        
        for message in history:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role in ["user", "assistant", "system"]:
                # Check if this is a multimodal message with images
                if isinstance(content, list) and any(item.get("type") == "image_url" for item in content if isinstance(item, dict)):
                    # This is a multimodal message with images - extract only the text
                    text_content = ""
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_content += item.get("text", "")
                    
                    # Add message with only the text content
                    formatted_history.append({"role": role, "content": text_content})
                else:
                    # Standard message without images - add as is
                    formatted_history.append({"role": role, "content": content})
            else:
                # Default unknown roles to user
                formatted_history.append({"role": "user", "content": content})
        
        return formatted_history
    
    def _setup_conversation_logger(self):
        """Set up a dedicated logger for conversation history."""
        # Create a formatter for the conversation log
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        
        # Create a conversation logger
        self.conversation_logger = logging.getLogger(f"conversation.{self.agent_type}")
        self.conversation_logger.setLevel(logging.INFO)
        
        # Create a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"conversation_{self.agent_type}_{timestamp}.log"
        # Create log directory if it doesn't exist
        if self._log_directory:
            os.makedirs(self._log_directory, exist_ok=True)
            log_path = os.path.join(self._log_directory, log_filename)
        else:
            log_path = log_filename
        
        # Add a file handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        self.conversation_logger.addHandler(file_handler)
        
        # Prevent logs from propagating to the root logger
        self.conversation_logger.propagate = False
        
        logger.info(f"Conversation logging enabled to {log_path}")
    
    def _log_conversation(self, role: str, content: Any, iteration: int = None):
        """
        Log a conversation message to the conversation log file.
        
        Args:
            role: The role of the message sender ('system', 'user', 'assistant', 'tool')
            content: The content of the message
            iteration: Optional iteration number for tool calls
        """
        if not self._enable_conversation_logging or not hasattr(self, 'conversation_logger'):
            return
        
        # Format the message based on role and content type
        iteration_str = f" [Iteration {iteration}]" if iteration is not None else ""
        header = f"=== {role.upper()}{iteration_str} ==="
        
        # Handle different content types appropriately
        if isinstance(content, dict):
            # Format dictionary content as JSON in code block
            message = f"{header}\n\n```json\n{json.dumps(content, indent=2)}\n```"
        elif role == "assistant" and isinstance(content, str):
            # Try to parse assistant responses as JSON and format them
            try:
                json_content = json.loads(content)
                message = f"{header}\n\n```json\n{json.dumps(json_content, indent=2)}\n```"
            except json.JSONDecodeError:
                # If it's not valid JSON, use the original format
                message = f"{header}\n{content}"
        else:
            # Format string content directly
            message = f"{header}\n{content}"
        
        # Add a separator for readability
        message += "\n" + "-" * 80
        
        # Log the formatted message
        self.conversation_logger.info(message)


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
        capture_image="environment",  # Automatically capture environment images
        max_tool_iterations=5
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