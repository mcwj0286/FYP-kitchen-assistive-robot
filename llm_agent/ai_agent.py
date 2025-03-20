import os
import json
import time
import requests
import base64
import datetime
import logging
import backoff
from typing import Dict, List, Any, Callable, Optional, Union
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_api_calls.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("llm_agent")

class Tool:
    """Base class for defining tools that the agent can use."""
    
    def __init__(self, name: str, description: str, function: Callable):
        """
        Initialize a tool.
        
        Args:
            name (str): The name of the tool.
            description (str): Description of what the tool does.
            function (Callable): The function to call when the tool is used.
        """
        self.name = name
        self.description = description
        self.function = function
    
    def __call__(self, *args, **kwargs):
        """Call the tool's function."""
        return self.function(*args, **kwargs)

class BaseAgent:
    """
    A base AI agent that can perform API calls and use tools.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the agent.
        
        Args:
            api_key (str, optional): API key for the LLM service. If None, will use OPENROUTER_API_KEY env var.
            model_name (str, optional): Name of the model to use. If None, will use MODEL_NAME env var.
            base_url (str): Base URL for the API.
            system_prompt (str, optional): Default system prompt. If None, will use SYSTEM_PROMPT env var.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either directly or via OPENROUTER_API_KEY environment variable")
        
        self.model_name = model_name or os.getenv("MODEL_NAME")
        if not self.model_name:
            raise ValueError("Model name must be provided either directly or via MODEL_NAME environment variable")
        
        self.base_url = base_url
        self.system_prompt = system_prompt or os.getenv("SYSTEM_PROMPT") or "You are a helpful assistant."
        self.tools = {}  # Dictionary to store available tools

    def add_tool(self, tool: Tool) -> None:
        """
        Add a tool to the agent.
        
        Args:
            tool (Tool): The tool to add.
        """
        self.tools[tool.name] = tool
        
    def remove_tool(self, tool_name: str) -> None:
        """
        Remove a tool from the agent.
        
        Args:
            tool_name (str): The name of the tool to remove.
        """
        if tool_name in self.tools:
            del self.tools[tool_name]
    
    def _format_tools_for_prompt(self) -> str:
        """Format tools information for inclusion in the system prompt."""
        if not self.tools:
            return ""
        
        tools_desc = "You have access to the following tools:\n\n"
        for name, tool in self.tools.items():
            tools_desc += f"- {name}: {tool.description}\n"
        
        tools_desc += "\nTo use a tool, respond with: [TOOL] <tool_name> <arguments> [/TOOL]"
        return tools_desc
    
    def _extract_tool_calls(self, response: str) -> List[Dict[str, str]]:
        """
        Extract tool calls from the model's response.
        
        Args:
            response (str): The model's response text.
            
        Returns:
            List[Dict[str, str]]: List of extracted tool calls with name and arguments.
        """
        tool_calls = []
        import re
        
        # Extract content between [TOOL] and [/TOOL] tags
        pattern = r'\[TOOL\](.*?)\[/TOOL\]'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for match in matches:
            parts = match.strip().split(maxsplit=1)
            if len(parts) >= 1:
                tool_name = parts[0].strip()
                args = parts[1].strip() if len(parts) > 1 else ""
                
                tool_calls.append({
                    "name": tool_name,
                    "arguments": args
                })
        
        return tool_calls
    
    def call_api(
        self, 
        messages: List[Dict[str, Any]], 
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Make a direct call to the LLM API.
        
        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries.
            max_tokens (int, optional): Maximum number of tokens for the response.
            temperature (float, optional): Temperature for response generation.
            
        Returns:
            Dict[str, Any]: The API response.
        """
        endpoint = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Log the API request
        request_id = f"req_{int(time.time())}_{id(self)}"
        log_messages = [
            # Remove image content to avoid huge logs
            {k: v if k != 'content' or not isinstance(v, list) else 'Image content removed for logging' 
             for k, v in msg.items()}
            for msg in messages
        ]
        
        log_data = {
            'model': self.model_name,
            'messages': log_messages,
            'max_tokens': max_tokens,
            'temperature': temperature
        }
        
        logger.info(f"LLM API Request [{request_id}]: {json.dumps(log_data, indent=2)}")
        
        # Implement retry mechanism with exponential backoff for rate limits
        max_retries = 3
        retry_delay = 2  # Base delay in seconds
        
        for retry in range(max_retries):
            try:
                start_time = time.time()
                response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
                response_time = time.time() - start_time
                
                # Check for rate limit (429) errors
                if response.status_code == 429:
                    wait_time = retry_delay * (2 ** retry)  # Exponential backoff
                    logger.warning(f"Rate limit exceeded. Waiting {wait_time} seconds before retry {retry+1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                response_json = response.json()
                
                # Log the API response
                logger.info(f"LLM API Response [{request_id}] (took {response_time:.2f}s):\n{json.dumps(response_json, indent=2)}")
                
                return response_json
                
            except requests.exceptions.RequestException as e:
                if retry < max_retries - 1 and (
                    getattr(e.response, "status_code", 0) == 429 or 
                    isinstance(e, requests.exceptions.Timeout)
                ):
                    wait_time = retry_delay * (2 ** retry)
                    logger.warning(f"API call failed with error: {str(e)}. Retrying in {wait_time}s ({retry+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    # Log API errors
                    error_msg = f"LLM API Error [{request_id}]: {str(e)}"
                    if hasattr(e, 'response') and e.response:
                        error_msg += f"\nStatus code: {e.response.status_code}"
                        try:
                            error_msg += f"\nResponse: {e.response.text}"
                        except:
                            pass
                    logger.error(error_msg)
                    raise
        
        # If we've exhausted all retries
        error_msg = f"LLM API Error [{request_id}]: Max retries ({max_retries}) exceeded"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    def process_with_image(
        self, 
        prompt: str, 
        image_path: str = None, 
        image_data_uri: str = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000
    ) -> str:
        """
        Process a prompt with an image attachment.
        
        Args:
            prompt (str): The user's text prompt.
            image_path (str, optional): Path to the image file.
            image_data_uri (str, optional): Direct data URI for the image. 
                                           If provided, image_path is ignored.
            system_prompt (str, optional): System prompt to override the default.
            max_tokens (int, optional): Maximum tokens for the response.
            
        Returns:
            str: The model's response text.
        """
        # Log the image processing request
        logger.info(f"Processing image request with prompt: '{prompt}'")
        
        try:
            # Get image data
            if image_data_uri:
                # Use the provided data URI directly
                logger.info("Using provided image data URI")
            elif image_path:
                logger.info(f"Image path: {image_path}")
                
                # Check if image exists
                if not os.path.exists(image_path):
                    error_msg = f"Image file not found: {image_path}"
                    logger.error(error_msg)
                    return f"Error: {error_msg}"
                
                # Log image metadata
                image_size = os.path.getsize(image_path) / 1024  # KB
                logger.info(f"Image size: {image_size:.2f} KB")
                
                # Encode the image
                with open(image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                
                image_data_uri = f"data:image/jpeg;base64,{encoded_string}"
                logger.info(f"Image encoded successfully. Data URI length: {len(image_data_uri)} chars")
            else:
                error_msg = "No image provided (need either image_path or image_data_uri)"
                logger.error(error_msg)
                return f"Error: {error_msg}"
            
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": system_prompt or self.system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_uri
                            }
                        }
                    ]
                }
            ]
            
            # Call API with retry for rate limits
            logger.info("Making API call with image data...")
            try:
                response = self.call_api(messages, max_tokens=max_tokens)
                
                if response and "choices" in response and len(response["choices"]) > 0:
                    result = response["choices"][0]["message"]["content"].strip()
                    logger.info(f"Received successful response (length: {len(result)} chars)")
                    return result
                else:
                    # Check if there's an error field in the response
                    if response and "error" in response:
                        error_info = response["error"]
                        error_message = error_info.get("message", "Unknown error")
                        error_code = error_info.get("code", "")
                        
                        logger.error(f"API error: {error_message} (code: {error_code})")
                        
                        if error_code == 429:
                            # Rate limit error
                            raise Exception(f"Rate limit exceeded (429): {error_message}")
                        else:
                            return f"Error from API: {error_message}"
                    else:
                        logger.error("Invalid or empty response structure from API")
                        return "Error: No valid response received"
            except Exception as e:
                logger.error(f"API call failed: {str(e)}")
                
                # Reraise if it's a rate limit error so it can be caught by the analyze_captured_image method
                if "429" in str(e) or "rate limit" in str(e).lower():
                    raise
                    
                return f"Error: API call failed - {str(e)}"
                
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            
            # Reraise if it's a rate limit error
            if "429" in str(e) or "rate limit" in str(e).lower():
                raise
                
            return f"Error processing image: {str(e)}"
    
    def process(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000, 
        use_tools: bool = True,
        max_iterations: int = 5
    ) -> str:
        """
        Process a text prompt, potentially using tools if available.
        
        Args:
            prompt (str): The user's text prompt.
            system_prompt (str, optional): System prompt to override the default.
            max_tokens (int, optional): Maximum tokens for the response.
            use_tools (bool): Whether to allow the agent to use tools.
            max_iterations (int): Maximum number of tool use iterations.
            
        Returns:
            str: The model's final response.
        """
        # Log the processing request
        logger.info(f"Processing request with prompt: '{prompt}'")
        logger.info(f"Use tools: {use_tools}, Max iterations: {max_iterations}, Max tokens: {max_tokens}")
        
        process_id = f"proc_{int(time.time())}_{id(self)}"
        
        # Initial system prompt, including tools info if applicable
        base_system_prompt = system_prompt or self.system_prompt
        if use_tools and self.tools:
            tool_info = self._format_tools_for_prompt()
            full_system_prompt = f"{base_system_prompt}\n\n{tool_info}"
            logger.info(f"Using system prompt with tools info (length: {len(full_system_prompt)} chars)")
        else:
            full_system_prompt = base_system_prompt
            logger.info(f"Using system prompt without tools info (length: {len(full_system_prompt)} chars)")
        
        # Initialize conversation history
        conversation = [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        iterations = 0
        final_response = ""
        
        while iterations < max_iterations:
            logger.info(f"[{process_id}] Starting iteration {iterations+1}/{max_iterations}")
            
            # Get model response
            try:
                logger.info(f"[{process_id}] Calling API with conversation length: {len(conversation)} messages")
                api_response = self.call_api(conversation, max_tokens=max_tokens)
                
                if not api_response or "choices" not in api_response or not api_response["choices"]:
                    logger.error(f"[{process_id}] Failed to get valid response from API")
                    return "Error: Failed to get valid response from API"
                
                model_response = api_response["choices"][0]["message"]["content"]
                logger.info(f"[{process_id}] Received model response (length: {len(model_response)} chars)")
                
                # Check for tool calls
                if use_tools and self.tools:
                    tool_calls = self._extract_tool_calls(model_response)
                    
                    if not tool_calls:
                        # No more tools to call, we're done
                        logger.info(f"[{process_id}] No tool calls extracted, finishing process")
                        final_response = model_response
                        break
                    
                    logger.info(f"[{process_id}] Extracted {len(tool_calls)} tool calls")
                    
                    # Process each tool call
                    tool_results = []
                    all_camera_captures = True  # Flag to check if all tool calls are camera captures
                    
                    for i, call in enumerate(tool_calls):
                        tool_name = call["name"]
                        arguments = call["arguments"]
                        
                        # Check if this is a camera capture command
                        is_camera_capture = (tool_name == "camera" and 
                                            arguments.strip().startswith("capture"))
                        
                        # If it's not a camera capture, we won't skip the tool result return to LLM
                        if not is_camera_capture:
                            all_camera_captures = False
                        
                        logger.info(f"[{process_id}] Processing tool call {i+1}/{len(tool_calls)}: {tool_name}")
                        
                        if tool_name in self.tools:
                            try:
                                logger.info(f"[{process_id}] Executing tool '{tool_name}' with args: {arguments}")
                                result = self.tools[tool_name](arguments)
                                logger.info(f"[{process_id}] Tool '{tool_name}' executed successfully")
                                tool_results.append(f"[TOOL RESULT: {tool_name}] {result} [/TOOL RESULT]")
                            except Exception as e:
                                error_msg = f"[{process_id}] Error executing tool '{tool_name}': {str(e)}"
                                logger.error(error_msg)
                                tool_results.append(f"[TOOL ERROR: {tool_name}] {str(e)} [/TOOL ERROR]")
                                all_camera_captures = False  # There was an error, so we can't skip
                        else:
                            logger.warning(f"[{process_id}] Tool '{tool_name}' not found")
                            tool_results.append(f"[TOOL ERROR: {tool_name}] Tool not found [/TOOL ERROR]")
                            all_camera_captures = False  # Tool not found, can't skip
                    
                    # Add model response to conversation
                    conversation.append({"role": "assistant", "content": model_response})
                    
                    # If all tool calls were successful camera captures, we don't need to send the results back to LLM
                    # This will be handled directly by the hardware_agent_example.py
                    if all_camera_captures and len(tool_calls) > 0 and tool_results:
                        logger.info(f"[{process_id}] All tool calls were camera captures, finishing process")
                        final_response = "Camera capture successful. Images are ready for analysis."
                        break
                    
                    # For other tools, add tool results to conversation
                    if tool_results:
                        tool_results_text = "\n\n".join(tool_results)
                        logger.info(f"[{process_id}] Adding tool results to conversation (length: {len(tool_results_text)} chars)")
                        conversation.append({"role": "user", "content": f"Here are the results of the tool calls:\n\n{tool_results_text}\n\nPlease provide your final response based on these results."})
                else:
                    # No tools used, just return the response
                    logger.info(f"[{process_id}] No tools used, returning direct response")
                    final_response = model_response
                    break
            
            except Exception as e:
                error_msg = f"[{process_id}] Error during processing iteration {iterations+1}: {str(e)}"
                logger.error(error_msg)
                return f"Error: {str(e)}"
                
            iterations += 1
        
        logger.info(f"[{process_id}] Process completed after {iterations} iterations")
        logger.info(f"[{process_id}] Final response length: {len(final_response)} chars")
        
        return final_response
    
    def process_api_request(
        self, 
        api_url: str, 
        method: str = "GET", 
        headers: Optional[Dict[str, str]] = None,
        payload: Optional[Dict[str, Any]] = None, 
        description: str = "API call"
    ) -> Dict[str, Any]:
        """
        Make an API request to an external service.
        
        Args:
            api_url (str): The URL for the API endpoint.
            method (str): HTTP method (GET, POST, etc).
            headers (Dict[str, str], optional): HTTP headers.
            payload (Dict[str, Any], optional): Request payload for POST, PUT, etc.
            description (str): Description of the API call for logging.
            
        Returns:
            Dict[str, Any]: The API response.
        """
        if not headers:
            headers = {}
        
        try:
            if method.upper() == "GET":
                response = requests.get(api_url, headers=headers)
            elif method.upper() == "POST":
                response = requests.post(api_url, headers=headers, json=payload)
            elif method.upper() == "PUT":
                response = requests.put(api_url, headers=headers, json=payload)
            elif method.upper() == "DELETE":
                response = requests.delete(api_url, headers=headers)
            else:
                return {"error": f"Unsupported HTTP method: {method}"}
            
            response.raise_for_status()
            
            try:
                return response.json()
            except ValueError:
                return {"text_response": response.text}
            
        except Exception as e:
            return {"error": str(e)}

    # Add a new method to analyze images captured by the camera tool
    def analyze_captured_image(
        self,
        camera_tool,
        prompt: str = "Describe what you see in this image in detail.",
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000
    ) -> str:
        """
        Analyze the most recently captured image from the camera tool.
        
        Args:
            camera_tool: The camera tool instance containing the captured image
            prompt (str): The prompt to send to the LLM about the image
            system_prompt (str, optional): System prompt to override the default
            max_tokens (int): Maximum tokens for the response
            
        Returns:
            str: The model's analysis of the image
        """
        # Check if the camera tool has a saved image
        image_path = camera_tool.get_last_image_path()
        
        if image_path and os.path.exists(image_path):
            logger.info(f"Analyzing image from path: {image_path}")
            try:
                return self.process_with_image(
                    prompt=prompt,
                    image_path=image_path,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens
                )
            except Exception as e:
                error_msg = f"Error analyzing image: {str(e)}"
                logger.error(error_msg)
                
                # If it's a rate limit error, provide a more user-friendly message
                if "429" in str(e) or "rate limit" in str(e).lower():
                    return "I'm unable to analyze the image at this moment due to API rate limits. The image has been saved at " + \
                           f"{image_path}. Please try again in a few moments."
                
                return f"Error analyzing image: {str(e)}"
        
        # If no image path, try to get the image data directly
        image_data_uri = camera_tool.get_b64_image()
        if image_data_uri:
            logger.info("Analyzing image from direct data URI")
            try:
                return self.process_with_image(
                    prompt=prompt,
                    image_data_uri=image_data_uri,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens
                )
            except Exception as e:
                error_msg = f"Error analyzing image: {str(e)}"
                logger.error(error_msg)
                
                # If it's a rate limit error, provide a more user-friendly message
                if "429" in str(e) or "rate limit" in str(e).lower():
                    return "I'm unable to analyze the image at this moment due to API rate limits. Please try again in a few moments."
                
                return f"Error analyzing image: {str(e)}"
        
        return "Error: No image available for analysis"


# Example usage:
if __name__ == "__main__":
    # Initialize agent
    agent = BaseAgent()
    
    # Define a simple calculator tool
    def calculator(args):
        """Simple calculator tool that evaluates basic math expressions."""
        try:
            # This is for demonstration only - in production, use a safer evaluation method
            result = eval(args)
            return f"The result is: {result}"
        except Exception as e:
            return f"Error calculating: {str(e)}"
    
    # Add the calculator tool to the agent
    agent.add_tool(Tool(
        name="calculator",
        description="Performs basic arithmetic calculations. Usage: calculator 2 + 2",
        function=calculator
    ))
    
    # Test with a prompt that might use tools
    response = agent.process(
        "I need to calculate 24 * 7, what's the result?",
        use_tools=True
    )
    
    print(f"Agent Response: {response}") 