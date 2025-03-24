from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, Optional, Union, Callable
import logging
import abc
import json
import re

# Set environment variables for OpenRouter API key and base URL
import os
from dotenv import load_dotenv
load_dotenv()

# Import hardware tools
try:
    from hardware_tools import HardwareTools, CameraTools, SpeakerTools, RoboticArmTools
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    logging.warning("Hardware tools module not found. Hardware capabilities will not be available.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tool functions
def text_processor(text: str) -> str:
    """
    Process text by counting words, characters, and providing basic statistics.
    
    Args:
        text: The input text to process
        
    Returns:
        A string with statistics about the text
    """
    if not text:
        return "No text provided."
    
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    avg_word_length = char_count / word_count if word_count > 0 else 0
    
    return f"Text Analysis:\n- Word count: {word_count}\n- Character count: {char_count}\n- Average word length: {avg_word_length:.2f}"

def calculator_tool(input_str: str) -> str:
    """
    Calculate the result of a mathematical expression.
    
    Args:
        input_str: A mathematical expression as a string (e.g., "2 + 2", "3 * 4")
        
    Returns:
        The result of the calculation as a string
    """
    try:
        # Using Python's eval but only for mathematical operations
        # This is safe as long as we're careful about what we allow
        # Convert ^ to ** for exponentiation
        input_str = input_str.replace("^", "**")
        
        # Clean up the input by removing any non-mathematical characters
        allowed_chars = set("0123456789+-*/() .**")
        cleaned_input = ''.join(c for c in input_str if c in allowed_chars)
        
        # Evaluate the expression
        result = eval(cleaned_input)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

def echo_tool(text: str) -> str:
    """
    Simply echoes back the input text.
    
    Args:
        text: The text to echo
        
    Returns:
        The input text prefixed with "Echo: "
    """
    return f"Echo: {text}"

class BaseAgent(abc.ABC):
    """
    Base class for all agents using structured output approach.
    """
    
    def __init__(
        self, 
        model_name: Optional[str] = None, 
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        system_prompt: Optional[str] = None,
        verbose: bool = True,
        use_hardware: bool = HARDWARE_AVAILABLE
    ):
        """
        Initialize the base agent with LLM settings.
        
        Args:
            model_name: The name of the LLM model to use
            api_key: API key for the LLM provider
            api_base: Base URL for API calls
            system_prompt: System prompt for the agent
            verbose: Whether to display detailed processing information
            use_hardware: Whether to use hardware tools if available
        """
        # Set up the language model
        self.llm = ChatOpenAI(
            openai_api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url=api_base or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            model_name=model_name or os.getenv("MODEL_NAME", "anthropic/claude-3-opus-20240229"),
            default_headers={
                "HTTP-Referer": os.getenv("YOUR_SITE_URL", "https://example.com"),
                "X-Title": os.getenv("YOUR_SITE_NAME", "AI Assistant"),
            }
        )
        
        self.verbose = verbose
        self.use_hardware = use_hardware
        
        # Initialize hardware tools if available and requested
        self.hardware = None
        if self.use_hardware and HARDWARE_AVAILABLE:
            try:
                self.hardware = HardwareTools(use_mock=not HARDWARE_AVAILABLE)
                logger.info("Hardware tools initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing hardware tools: {e}")
                self.hardware = None
                self.use_hardware = False
        
        # Initialize available tools first
        self._available_tools = self._get_available_tools()
        # Then initialize system prompt using those tools
        self._system_prompt = system_prompt or self._get_default_prompt()
        
        logger.info(f"Base Agent initialized with model: {model_name or os.getenv('MODEL_NAME')}")
    
    def _get_available_tools(self) -> Dict[str, Callable]:
        """
        Get the available tools as a dictionary mapping tool names to functions.
        Override in subclasses to provide specialized tools.
        
        Returns:
            Dictionary of tool name to tool function
        """
        # Base implementation provides basic tools
        return {
            "calculator": calculator_tool,
            "text_processor": text_processor
        }
    
    def _get_default_prompt(self) -> str:
        """
        Get the default system prompt for this agent.
        Override in subclasses to provide specialized prompts.
        
        Returns:
            Default system prompt as a string
        """
        tools_info = self._format_tools_for_prompt()
        
        return f"""You are a helpful AI assistant.

You have access to the following tools:
{tools_info}

When you need to use a tool, respond ONLY in the following format:
```json
{{
  "thought": "Your internal reasoning about what to do",
  "action": "tool_name",
  "action_input": "input to the tool"
}}
```

The available tool names are: {', '.join(self._get_available_tools().keys())}

If you don't need to use a tool, respond ONLY in the following format:
```json
{{
  "thought": "Your internal reasoning about the response",
  "response": "Your response to the user"
}}
```

Think carefully about when to use tools versus when to respond directly.
"""
    
    def _format_tools_for_prompt(self) -> str:
        """
        Format the available tools information for inclusion in the system prompt.
        
        Returns:
            String with formatted tool information
        """
        tools_info = ""
        for name, func in self._available_tools.items():
            doc = func.__doc__ or "No description available."
            # Fix: Extract first line without using split with backslash
            first_line = doc.strip()
            if "\n" in first_line:
                first_line = first_line[:first_line.find("\n")]
            tools_info += f"- {name}: {first_line}" + "\n"
        return tools_info
    
    @property
    def system_prompt(self) -> str:
        """Get the agent's system prompt."""
        return self._system_prompt
    
    @system_prompt.setter
    def system_prompt(self, prompt: str) -> None:
        """
        Set a new system prompt.
        
        Args:
            prompt: The new system prompt
        """
        self._system_prompt = prompt
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract structured data.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            Parsed response as a dictionary
        """
        try:
            # Extract JSON from the response using regex
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without the markdown code block
                json_match = re.search(r'({.*})', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    logger.warning("Could not extract JSON from response")
                    return {"response": response}
            
            # Parse the JSON
            parsed = json.loads(json_str)
            return parsed
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
    
    def __del__(self):
        """
        Clean up hardware resources when the agent is deleted.
        """
        if self.hardware:
            try:
                self.hardware.close()
                logger.info("Hardware connections closed")
            except Exception as e:
                logger.error(f"Error closing hardware connections: {e}")


class KitchenAssistantAgent(BaseAgent):
    """
    Specialized agent for kitchen assistance.
    Extends BaseAgent with kitchen-specific tools and prompts.
    """
    
    def __init__(
        self, 
        model_name: Optional[str] = None, 
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        verbose: bool = True,
        use_hardware: bool = HARDWARE_AVAILABLE
    ):
        """
        Initialize the kitchen assistant agent.
        
        Args:
            model_name: The name of the LLM model to use
            api_key: API key for the LLM provider
            api_base: Base URL for API calls
            verbose: Whether to display detailed processing information
            use_hardware: Whether to use hardware tools if available
        """
        # Kitchen-specific system prompt will be generated in the parent class
        # using the kitchen-specific tools
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            system_prompt=None,  # Will be generated based on tools
            verbose=verbose,
            use_hardware=use_hardware
        )
        
        # After initialization, update the system prompt with kitchen-specific context
        kitchen_context = """You are a helpful kitchen assistant AI that controls physical hardware devices.

You have access to the following hardware capabilities:
- Robot arm for manipulating objects in the kitchen
- Cameras for observing the environment and objects
- Speaker for communicating through voice

Use your available tools appropriately to help the user with kitchen-related tasks.
Always think step by step when handling complex cooking instructions or recipes.
When using hardware tools, remember that robot commands are sent without waiting for completion.
If a task requires sequential robot movements, pause between commands by explaining steps to the user.
"""
        
        # Combine the tool information with the kitchen context
        tools_info = self._format_tools_for_prompt()
        self._system_prompt = f"{kitchen_context}\n\nYou have access to the following tools:\n{tools_info}\n\n"
        self._system_prompt += """When you need to use a tool, respond ONLY in the following format:
```json
{
  "thought": "Your internal reasoning about what to do",
  "action": "tool_name",
  "action_input": "input to the tool"
}
```

The available tool names are: """ + ', '.join(self._get_available_tools().keys()) + """

If you don't need to use a tool, respond ONLY in the following format:
```json
{
  "thought": "Your internal reasoning about the response",
  "response": "Your response to the user"
}
```

Think carefully about when to use tools versus when to respond directly.
"""
        
        logger.info("Kitchen Assistant Agent initialized")
    
    def _get_available_tools(self) -> Dict[str, Callable]:
        """
        Get the available tools for the kitchen assistant.
        
        Returns:
            Dictionary of kitchen-specific tools
        """
        # Start with the base tools
        tools = super()._get_available_tools()
        
        # Add kitchen-specific tools
        tools["echo"] = echo_tool
        
        # Add hardware tools if available
        if self.use_hardware and self.hardware:
            # Camera tools
            tools["capture_environment"] = self.hardware.camera_tools.capture_environment
            tools["capture_wrist"] = self.hardware.camera_tools.capture_wrist
            tools["analyze_image"] = self.hardware.camera_tools.analyze_image
            
            # Speaker tools
            tools["speak"] = self.hardware.speaker_tools.speak
            tools["is_speaking"] = self.hardware.speaker_tools.is_speaking
            tools["stop_speaking"] = self.hardware.speaker_tools.stop_speaking
            
            # Robotic arm tools
            tools["move_home"] = self.hardware.arm_tools.move_home
            tools["move_position"] = self.hardware.arm_tools.move_position
            tools["grasp"] = self.hardware.arm_tools.grasp
            tools["release"] = self.hardware.arm_tools.release
            tools["get_position"] = self.hardware.arm_tools.get_position
            tools["move_default"] = self.hardware.arm_tools.move_default
            
            logger.info(f"Added {len(tools) - 2} hardware tools to the Kitchen Assistant Agent")
        
        return tools


# Example usage
if __name__ == "__main__":
    # Create a general base agent
    base_agent = BaseAgent(verbose=True)
    print("=== Base Agent Test ===")
    response = base_agent.process_to_string("What is 15 multiplied by 32 plus 7?")
    print(f"Response: {response}\n")
    
    # Create a kitchen assistant agent
    kitchen_agent = KitchenAssistantAgent(verbose=True)
    print("=== Kitchen Agent Test ===")
    response = kitchen_agent.process_to_string("Can you analyze this recipe: 2 cups flour, 1 cup sugar, 3 eggs?")
    print(f"Response: {response}")
    
    # Test hardware tools if available
    if HARDWARE_AVAILABLE:
        print("\n=== Hardware Tools Test ===")
        response = kitchen_agent.process_to_string("Can you check what the robot arm is seeing?")
        print(f"Response: {response}")
