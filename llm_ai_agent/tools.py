#!/usr/bin/env python3
"""
Tools for the Kitchen Assistive Robot AI.
This module provides a collection of tools that can be used by agents.
"""

import re
import logging
from typing import Dict, Any, Optional, List
import os
import sys
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment variables to control hardware components
ENABLE_CAMERA = os.getenv("ENABLE_CAMERA", "true").lower() == "true"
ENABLE_SPEAKER = os.getenv("ENABLE_SPEAKER", "true").lower() == "true"
ENABLE_ARM = os.getenv("ENABLE_ARM", "true").lower() == "true"

logger.info(f"Hardware components enabled via env vars: Camera={ENABLE_CAMERA}, Speaker={ENABLE_SPEAKER}, Arm={ENABLE_ARM}")

# Import hardware tools
try:
    from .hardware_tools import CameraTools, SpeakerTools, RoboticArmTools
    HARDWARE_AVAILABLE = True
    logger.info("Successfully imported hardware tools")
except ImportError as e:
    logger.warning(f"Could not import hardware tools: {e}")
    HARDWARE_AVAILABLE = False

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

def text_processor(text: str) -> str:
    """
    Process text by counting words, characters, and performing other text operations.
    
    Args:
        text: The text to process
        
    Returns:
        Information about the text
    """
    try:
        # Count words
        words = text.split()
        word_count = len(words)
        
        # Count characters
        char_count = len(text)
        char_count_no_spaces = len(text.replace(" ", ""))
        
        # Count sentences (simple approximation)
        sentence_count = len(re.findall(r'[.!?]+', text)) or 1
        
        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / max(1, word_count)
        
        # Calculate average sentence length
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # Generate result
        result = (
            f"Text Analysis:\n"
            f"- Words: {word_count}\n"
            f"- Characters (with spaces): {char_count}\n"
            f"- Characters (without spaces): {char_count_no_spaces}\n"
            f"- Sentences: {sentence_count}\n"
            f"- Average word length: {avg_word_length:.2f} characters\n"
            f"- Average sentence length: {avg_sentence_length:.2f} words"
        )
        
        return result
    except Exception as e:
        return f"Error processing text: {str(e)}"

def echo_tool(text: str) -> str:
    """
    Simply echoes back the input text.
    
    Args:
        text: The text to echo
        
    Returns:
        The input text prefixed with "Echo: "
    """
    return f"Echo: {text}"

# Memory access tools

def get_memory_path() -> str:
    """Helper function to get the path to the memory directory."""
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # The memory directory is a subdirectory of the current directory
    memory_dir = os.path.join(current_dir, "memory")
    return memory_dir

def get_action_plans() -> str:
    """
    Retrieve all action plans stored in memory.
    
    This tool provides access to predefined robot action sequences, including
    step-by-step procedures for tasks like opening jars or retrieving items.
    
    Returns:
        A formatted string containing all available action plans
    """
    try:
        memory_dir = get_memory_path()
        action_plans_file = os.path.join(memory_dir, "action_plan.yaml")
        
        if not os.path.exists(action_plans_file):
            return "Error: Action plans file not found in memory directory."
        
        with open(action_plans_file, 'r') as file:
            action_plans = yaml.safe_load(file)
        
        if not action_plans:
            return "No action plans found in memory."
        
        # Format the output for readability
        result = "Available Action Plans:\n\n"
        for plan_name, plan_details in action_plans.items():
            result += f"Plan: {plan_name}\n"
            result += f"Goal: {plan_details.get('goal', 'No goal specified')}\n"
            
            steps = plan_details.get('steps', [])
            if steps:
                result += "Steps:\n"
                for step in steps:
                    step_num = step.get('step_num', '?')
                    description = step.get('description', 'No description')
                    result += f"  {step_num}. {description}\n"
            else:
                result += "No steps defined for this plan.\n"
            
            result += "\n"
        
        return result
    except yaml.YAMLError as e:
        return f"Error parsing action plans YAML: {e}"
    except Exception as e:
        return f"Error retrieving action plans: {e}"

def get_action_positions() -> str:
    """
    Retrieve all stored robot arm positions from memory.
    
    This tool provides access to predefined robot arm coordinates for specific actions,
    such as positions for grasping objects or performing tasks.
    
    Returns:
        A formatted string containing all stored action positions
    """
    try:
        memory_dir = get_memory_path()
        positions_file = os.path.join(memory_dir, "action_position.yaml")
        
        if not os.path.exists(positions_file):
            return "Error: Action positions file not found in memory directory."
        
        with open(positions_file, 'r') as file:
            positions = yaml.safe_load(file)
        
        if not positions:
            return "No action positions found in memory."
        
        # Format the output for readability
        result = "Available Action Positions:\n\n"
        for action_name, coordinates in positions.items():
            result += f"Action: {action_name}\n"
            result += f"Cartesian position: {coordinates}\n\n"
        
        return result
    except yaml.YAMLError as e:
        return f"Error parsing action positions YAML: {e}"
    except Exception as e:
        return f"Error retrieving action positions: {e}"

def get_item_locations() -> str:
    """
    Retrieve the locations of all known items in the environment.
    
    This tool provides access to the stored coordinates of items that the
    robot knows about, such as objects on tables or in the workspace.
    
    Returns:
        A formatted string containing all known item locations
    """
    try:
        memory_dir = get_memory_path()
        locations_file = os.path.join(memory_dir, "item_location.yaml")
        
        if not os.path.exists(locations_file):
            return "Error: Item locations file not found in memory directory."
        
        with open(locations_file, 'r') as file:
            locations_data = yaml.safe_load(file)
        
        if not locations_data or 'items' not in locations_data:
            return "No item locations found in memory."
        
        items = locations_data.get('items', {})
        
        # Format the output for readability
        result = "Known Item Locations:\n\n"
        for item_name, item_data in items.items():
            result += f"Item: {item_name}\n"
            coordinates = item_data.get('coordinates', 'Unknown')
            result += f"Coordinates: {coordinates}\n\n"
        
        return result
    except yaml.YAMLError as e:
        return f"Error parsing item locations YAML: {e}"
    except Exception as e:
        return f"Error retrieving item locations: {e}"

# Add more tools as needed

# Initialize hardware tools
hardware_tools = {}

if HARDWARE_AVAILABLE:
    # Initialize camera tools if enabled
    if ENABLE_CAMERA:
        try:
            camera_tools = CameraTools()
            # Add camera tools
            hardware_tools["capture"] = camera_tools.capture_environment
            hardware_tools["capture_environment"] = camera_tools.capture_environment
            hardware_tools["capture_wrist"] = camera_tools.capture_wrist
            hardware_tools["capture_both"] = camera_tools.capture_both
            logger.info("Camera tools initialized successfully")
        except Exception as e:
            logger.warning(f"Error initializing camera tools: {e}")
    else:
        logger.info("Camera tools disabled via ENABLE_CAMERA environment variable")

    # Initialize speaker tools if enabled
    if ENABLE_SPEAKER:
        try:
            speaker_tools = SpeakerTools()
            # Add speaker tools
            hardware_tools["speak"] = speaker_tools.speak
            hardware_tools["is_speaking"] = speaker_tools.is_speaking
            hardware_tools["stop_speaking"] = speaker_tools.stop_speaking
            hardware_tools["play_audio"] = speaker_tools.play_audio
            logger.info("Speaker tools initialized successfully")
        except Exception as e:
            logger.warning(f"Error initializing speaker tools: {e}")
    else:
        logger.info("Speaker tools disabled via ENABLE_SPEAKER environment variable")

    # Initialize robotic arm tools if enabled
    if ENABLE_ARM:
        try:
            arm_tools = RoboticArmTools()
            # Add robotic arm tools
            hardware_tools["move_home"] = arm_tools.move_home
            hardware_tools["move_default"] = arm_tools.move_default
            hardware_tools["move_position"] = arm_tools.move_position
            hardware_tools["grasp"] = arm_tools.grasp
            hardware_tools["release"] = arm_tools.release
            hardware_tools["get_position"] = arm_tools.get_position
            logger.info("Robotic arm tools initialized successfully")
        except Exception as e:
            logger.warning(f"Error initializing robotic arm tools: {e}")
    else:
        logger.info("Robotic arm tools disabled via ENABLE_ARM environment variable")

# Dictionary of all available tools
TOOLS = {
    "calculator": calculator_tool,
    "text_processor": text_processor,
    "echo": echo_tool,
    "get_action_plans": get_action_plans,
    "get_action_positions": get_action_positions,
    "get_item_locations": get_item_locations,
}

# Add hardware tools to TOOLS dictionary
TOOLS.update(hardware_tools)

def get_tool(name: str):
    """
    Get a tool by name.
    
    Args:
        name: Name of the tool
        
    Returns:
        Tool function or None if not found
    """
    return TOOLS.get(name)

def get_all_tools() -> Dict[str, callable]:
    """
    Get all available tools.
    
    Returns:
        Dictionary of all tools
    """
    return TOOLS.copy()

def get_tool_names() -> List[str]:
    """
    Get the names of all available tools.
    
    Returns:
        List of tool names
    """
    return list(TOOLS.keys())

# Example usage
if __name__ == "__main__":
    # Test the tools
    print("Calculator: ", calculator_tool("2 + 2 * 3"))
    print("Text Processor: ", text_processor("This is a test. This is only a test."))
    print("Echo: ", echo_tool("Hello, world!"))
    
    # Test memory tools
    print("\nAction Plans:")
    print(get_action_plans())
    
    print("\nAction Positions:")
    print(get_action_positions())
    
    print("\nItem Locations:")
    print(get_item_locations())
    
    # Print all available tools
    print("\nAvailable tools:")
    for name in get_tool_names():
        print(f"- {name}") 