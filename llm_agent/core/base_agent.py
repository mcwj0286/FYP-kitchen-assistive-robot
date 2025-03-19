import os
import time
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Tool(ABC):
    """
    Abstract base class for all tools that can be used by agents.
    """
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the tool's functionality.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            Any: The result of the tool execution.
        """
        pass

class MemorySystem:
    """
    Memory system for storing and retrieving information.
    Supports both short-term and long-term memory.
    """
    def __init__(
        self, 
        enable_short_term_memory: bool = True, 
        enable_long_term_memory: bool = False,
        storage_path: Optional[str] = None
    ):
        self.enable_short_term_memory = enable_short_term_memory
        self.enable_long_term_memory = enable_long_term_memory
        self.storage_path = storage_path
        
        # Initialize memory stores
        self.short_term_memory = [] if enable_short_term_memory else None
        self.long_term_memory = {} if enable_long_term_memory else None
        
        # Create storage directory if needed
        if enable_long_term_memory and storage_path:
            os.makedirs(storage_path, exist_ok=True)
            
        self.logger = logging.getLogger(__name__)
    
    def store(self, data: Dict[str, Any], memory_type: str = "short_term") -> bool:
        """
        Store data in the specified memory type.
        
        Args:
            data: The data to store.
            memory_type: Type of memory to use ("short_term" or "long_term").
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if memory_type == "short_term" and self.enable_short_term_memory:
                self.short_term_memory.append({
                    "timestamp": time.time(),
                    "data": data
                })
                return True
                
            elif memory_type == "long_term" and self.enable_long_term_memory:
                # Generate a unique key for storing the data
                key = str(time.time())
                
                if self.storage_path:
                    # Store to file system
                    file_path = os.path.join(self.storage_path, f"{key}.json")
                    with open(file_path, 'w') as f:
                        json.dump(data, f)
                else:
                    # Store in memory
                    self.long_term_memory[key] = data
                return True
                
            return False
        except Exception as e:
            self.logger.error(f"Error storing data in {memory_type} memory: {e}")
            return False
    
    def retrieve(self, query: Dict[str, Any] = None, memory_type: str = "short_term", limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve data from the specified memory type.
        
        Args:
            query: Optional query to filter the results.
            memory_type: Type of memory to query ("short_term" or "long_term").
            limit: Maximum number of results to return.
            
        Returns:
            List[Dict[str, Any]]: The retrieved data.
        """
        try:
            if memory_type == "short_term" and self.enable_short_term_memory:
                if not self.short_term_memory:
                    return []
                
                # Simple filtering based on query
                if query:
                    results = []
                    for entry in self.short_term_memory:
                        match = True
                        for key, value in query.items():
                            if key not in entry["data"] or entry["data"][key] != value:
                                match = False
                                break
                        if match:
                            results.append(entry)
                    return results[-limit:]
                else:
                    return self.short_term_memory[-limit:]
                    
            elif memory_type == "long_term" and self.enable_long_term_memory:
                results = []
                
                if self.storage_path:
                    # Retrieve from file system
                    files = os.listdir(self.storage_path)
                    files.sort(reverse=True)  # Most recent first
                    
                    for file_name in files[:limit]:
                        if file_name.endswith('.json'):
                            file_path = os.path.join(self.storage_path, file_name)
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                                
                                # Apply query filtering
                                if query:
                                    match = True
                                    for key, value in query.items():
                                        if key not in data or data[key] != value:
                                            match = False
                                            break
                                    if match:
                                        results.append({
                                            "timestamp": float(file_name.split('.')[0]),
                                            "data": data
                                        })
                                else:
                                    results.append({
                                        "timestamp": float(file_name.split('.')[0]),
                                        "data": data
                                    })
                else:
                    # Retrieve from memory
                    keys = sorted(self.long_term_memory.keys(), reverse=True)
                    
                    for key in keys[:limit]:
                        data = self.long_term_memory[key]
                        
                        # Apply query filtering
                        if query:
                            match = True
                            for q_key, q_value in query.items():
                                if q_key not in data or data[q_key] != q_value:
                                    match = False
                                    break
                            if match:
                                results.append({
                                    "timestamp": float(key),
                                    "data": data
                                })
                        else:
                            results.append({
                                "timestamp": float(key),
                                "data": data
                            })
                
                return results
                
            return []
        except Exception as e:
            self.logger.error(f"Error retrieving data from {memory_type} memory: {e}")
            return []
    
    def clear(self, memory_type: str = "short_term") -> bool:
        """
        Clear the specified memory type.
        
        Args:
            memory_type: Type of memory to clear ("short_term" or "long_term").
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if memory_type == "short_term" and self.enable_short_term_memory:
                self.short_term_memory = []
                return True
                
            elif memory_type == "long_term" and self.enable_long_term_memory:
                if self.storage_path:
                    # Clear files
                    for file_name in os.listdir(self.storage_path):
                        if file_name.endswith('.json'):
                            os.remove(os.path.join(self.storage_path, file_name))
                else:
                    # Clear memory
                    self.long_term_memory = {}
                return True
                
            return False
        except Exception as e:
            self.logger.error(f"Error clearing {memory_type} memory: {e}")
            return False

# Fallback implementations for the functions that may be missing
def upload_images_to_cloudinary(image_paths):
    """Fallback implementation for upload_images_to_cloudinary."""
    logging.warning("Using fallback upload_images_to_cloudinary. No actual upload is performed.")
    result = {}
    for cam_id, path in image_paths.items():
        result[cam_id] = f"file://{path}"  # Just return a file URL
    return result

def upload_image_to_server(server_url, file_path):
    """Fallback implementation for upload_image_to_server."""
    logging.warning(f"Using fallback upload_image_to_server for {server_url}. No actual upload is performed.")
    return f"file://{file_path}"  # Just return a file URL

def call_llm_with_images(prompt, image_urls, model_name=None, system_prompt=None, debug=False):
    """Fallback implementation for call_llm_with_images."""
    logging.warning("Using fallback call_llm_with_images. Returning mock response.")
    return f"This is a mock LLM response to the prompt: {prompt}\nSystem prompt: {system_prompt or 'None'}\nImages: {list(image_urls.keys())}"

def save_images(frames):
    """Fallback implementation for save_images."""
    logging.warning("Using fallback save_images. No actual saving is performed.")
    import tempfile
    import cv2
    import os
    
    image_paths = {}
    for cam_id, frame in frames.items():
        # Create a temporary file for the image
        fd, temp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        
        # Save the frame as an image
        cv2.imwrite(temp_path, frame)
        
        # Add the path to the dictionary
        image_paths[cam_id] = temp_path
    
    return image_paths

class BaseAgent:
    """
    Base agent class that provides core functionality for all agent types.
    """
    def __init__(
        self,
        name: str,
        description: str,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        server_url: Optional[str] = None,
        enable_short_term_memory: bool = True,
        enable_long_term_memory: bool = False,
        memory_storage_path: Optional[str] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            name: The name of the agent.
            description: A description of the agent's purpose.
            model_name: The name of the LLM model to use.
            system_prompt: The system prompt to use for LLM interactions.
            server_url: URL of the server to upload images to.
            enable_short_term_memory: Whether to enable short-term memory.
            enable_long_term_memory: Whether to enable long-term memory.
            memory_storage_path: Path to store long-term memory (if enabled).
        """
        self.name = name
        self.description = description
        
        # LLM configuration
        self.model_name = model_name or os.getenv("MODEL_NAME")
        self.system_prompt = system_prompt or os.getenv("SYSTEM_PROMPT")
        
        # Server configuration
        self.server_url = server_url or os.getenv("IMAGE_SERVER_URL")
        
        # Initialize memory system
        if memory_storage_path and not os.path.isabs(memory_storage_path):
            memory_storage_path = os.path.join(os.getcwd(), memory_storage_path)
            
        self.memory = MemorySystem(
            enable_short_term_memory=enable_short_term_memory,
            enable_long_term_memory=enable_long_term_memory,
            storage_path=memory_storage_path
        )
        
        # Initialize tool registry
        self.tools = {}
        
        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        
        # Try to import the necessary functions from get_prompt.py
        try:
            # Try absolute import first
            import sys
            import importlib.util
            
            # Try to find get_prompt.py in the llm_agent directory
            module_path = None
            
            # Look in the current directory and parent directories
            current_dir = os.path.dirname(os.path.abspath(__file__))
            dirs_to_check = [current_dir]
            
            # Add parent directory if not at root
            parent_dir = os.path.dirname(current_dir)
            if parent_dir and parent_dir != current_dir:
                dirs_to_check.append(parent_dir)
                
                # Add grandparent directory
                grandparent_dir = os.path.dirname(parent_dir)
                if grandparent_dir and grandparent_dir != parent_dir:
                    dirs_to_check.append(grandparent_dir)
            
            # Check each directory for get_prompt.py
            for dir_path in dirs_to_check:
                possible_path = os.path.join(dir_path, "get_prompt.py")
                if os.path.exists(possible_path):
                    module_path = possible_path
                    break
                    
                # Also check in an llm_agent subdirectory
                possible_path = os.path.join(dir_path, "llm_agent", "get_prompt.py")
                if os.path.exists(possible_path):
                    module_path = possible_path
                    break
            
            if module_path:
                self.logger.info(f"Found get_prompt.py at {module_path}")
                
                # Import the module
                spec = importlib.util.spec_from_file_location("get_prompt", module_path)
                get_prompt = importlib.util.module_from_spec(spec)
                sys.modules["get_prompt"] = get_prompt
                spec.loader.exec_module(get_prompt)
                
                # Get the functions
                self.upload_images_to_cloudinary = getattr(get_prompt, "upload_images_to_cloudinary", upload_images_to_cloudinary)
                self.upload_image_to_server = getattr(get_prompt, "upload_image_to_server", upload_image_to_server)
                self.call_llm_with_images = getattr(get_prompt, "call_llm_with_images", call_llm_with_images)
                self.save_images = getattr(get_prompt, "save_images", save_images)
            else:
                # Use fallback implementations
                self.logger.warning("Could not find get_prompt.py. Using fallback implementations.")
                self.upload_images_to_cloudinary = upload_images_to_cloudinary
                self.upload_image_to_server = upload_image_to_server
                self.call_llm_with_images = call_llm_with_images
                self.save_images = save_images
                
        except Exception as e:
            self.logger.error(f"Error importing required modules: {e}")
            self.logger.warning("Using fallback implementations.")
            
            # Use fallback implementations
            self.upload_images_to_cloudinary = upload_images_to_cloudinary
            self.upload_image_to_server = upload_image_to_server
            self.call_llm_with_images = call_llm_with_images
            self.save_images = save_images
    
    def register_tool(self, tool: Tool) -> bool:
        """
        Register a tool with the agent.
        
        Args:
            tool: The tool to register.
            
        Returns:
            bool: True if registration was successful, False otherwise.
        """
        try:
            if tool.name in self.tools:
                self.logger.warning(f"Tool with name '{tool.name}' already exists. Overwriting.")
                
            self.tools[tool.name] = tool
            self.logger.info(f"Tool '{tool.name}' registered successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error registering tool '{tool.name}': {e}")
            return False
    
    def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool from the agent.
        
        Args:
            tool_name: The name of the tool to unregister.
            
        Returns:
            bool: True if unregistration was successful, False otherwise.
        """
        try:
            if tool_name in self.tools:
                del self.tools[tool_name]
                self.logger.info(f"Tool '{tool_name}' unregistered successfully.")
                return True
            else:
                self.logger.warning(f"Tool '{tool_name}' not found.")
                return False
        except Exception as e:
            self.logger.error(f"Error unregistering tool '{tool_name}': {e}")
            return False
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        Get a registered tool.
        
        Args:
            tool_name: The name of the tool to get.
            
        Returns:
            Optional[Tool]: The tool if found, None otherwise.
        """
        return self.tools.get(tool_name)
    
    def execute_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """
        Execute a registered tool.
        
        Args:
            tool_name: The name of the tool to execute.
            *args: Arguments to pass to the tool.
            **kwargs: Keyword arguments to pass to the tool.
            
        Returns:
            Any: The result of the tool execution.
        """
        tool = self.get_tool(tool_name)
        if tool:
            try:
                self.logger.info(f"Executing tool '{tool_name}'")
                result = tool.execute(*args, **kwargs)
                return result
            except Exception as e:
                self.logger.error(f"Error executing tool '{tool_name}': {e}")
                return None
        else:
            self.logger.warning(f"Tool '{tool_name}' not found.")
            return None
    
    def upload_images(self, image_paths: Dict[str, str]) -> Dict[str, str]:
        """
        Upload images to a server or cloud service.
        
        Args:
            image_paths: Dictionary of camera IDs to image file paths.
            
        Returns:
            Dict[str, str]: Dictionary of camera IDs to image URLs.
        """
        if not image_paths:
            self.logger.warning("No images provided for upload.")
            return {}
            
        try:
            if self.server_url:
                # Use server upload
                uploaded_urls = {}
                for cam_id, file_path in image_paths.items():
                    url = self.upload_image_to_server(self.server_url, file_path)
                    if url:
                        uploaded_urls[cam_id] = url
                return uploaded_urls
            else:
                # Use Cloudinary upload
                return self.upload_images_to_cloudinary(image_paths)
        except Exception as e:
            self.logger.error(f"Error uploading images: {e}")
            return {}
    
    def call_llm(
        self, 
        prompt: str, 
        image_urls: Optional[Dict[str, str]] = None, 
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        debug: bool = False
    ) -> str:
        """
        Call the language model with a prompt and optional images.
        
        Args:
            prompt: The prompt to send to the LLM.
            image_urls: Optional dictionary of image URLs to include.
            system_prompt: Optional system prompt to override the default.
            model_name: Optional model name to override the default.
            debug: Whether to enable debug mode.
            
        Returns:
            str: The LLM response.
        """
        try:
            return self.call_llm_with_images(
                prompt, 
                image_urls or {}, 
                model_name=model_name or self.model_name, 
                system_prompt=system_prompt or self.system_prompt,
                debug=debug
            )
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            return f"error: {str(e)}"
    
    def set_system_prompt(self, system_prompt: str) -> None:
        """
        Set the system prompt for the agent.
        
        Args:
            system_prompt: The new system prompt.
        """
        self.system_prompt = system_prompt
        self.logger.info("System prompt updated.")
    
    def set_model_name(self, model_name: str) -> None:
        """
        Set the model name for the agent.
        
        Args:
            model_name: The new model name.
        """
        self.model_name = model_name
        self.logger.info(f"Model name updated to '{model_name}'.")
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Process input data and generate a response.
        This method must be implemented by subclasses.
        
        Args:
            input_data: The input data to process.
            
        Returns:
            Any: The processing result.
        """
        pass 