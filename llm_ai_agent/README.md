# LLM AI Agent Framework

A flexible, configuration-based framework for creating AI agents with different capabilities for kitchen assistance and other tasks.

## Current Progress

As of the latest update, we have successfully implemented:

- ✅ Complete YAML-based configuration system
- ✅ Unified `ConfigurableAgent` class that replaces all previous agent types
- ✅ Centralized tools management via `tools.py`
- ✅ Hardware abstraction layer with real and mock implementations
- ✅ Comprehensive error handling for LLM responses
- ✅ Interactive testing interface with enhanced usability
- ✅ Cross-platform support for hardware tools
- ✅ Test suite for configurations, tools, and hardware interfaces
- ✅ Automatic image capture functionality with multiple camera support
- ✅ Multimodal vision capabilities with controlled access to camera tools
- ✅ System prompt debugging and inspection capabilities
- ✅ Unified hardware tools integration with environment variable control
- ✅ Simplified configuration with streamlined tool inclusion
- ✅ Memory access tools for retrieving action plans, positions, and item locations

## System Architecture

The framework follows a modular, configuration-driven architecture:

- `agents.py`: Central module for creating and managing agents
- `configurable_agent.py`: The main agent implementation that loads behavior from configuration
- `config_loader.py`: Utilities for loading and processing YAML configurations
- `tools.py`: Collection of basic tools and unified hardware tools that agents can use
- `interactive.py`: Command-line interface for interacting with agents
- `hardware_tools.py`: Hardware interface implementations with mock fallbacks
- `memory/`: Directory containing YAML files for stored knowledge about the environment

### Key Design Principles

1. **Configuration over Code**: Agent behavior is defined in YAML files
2. **Single Agent Implementation**: `ConfigurableAgent` is the only agent class needed
3. **Hardware Abstraction**: Seamlessly switch between real and mock hardware
4. **Modular Tool System**: Easily add, remove, or customize tools
5. **Robust Error Handling**: Graceful handling of LLM response failures
6. **Controlled Hardware Access**: Camera capture managed via configuration rather than direct tool calls
7. **Multimodal Capabilities**: Support for text and image inputs to the LLM
8. **Environment Variable Control**: Hardware components can be selectively enabled/disabled
9. **Persistent Memory**: Structured storage of environmental knowledge and action procedures

## Quick Start

```python
from llm_ai_agent.agents import create_agent

# Create a base agent
agent = create_agent(agent_type="base_agent")

# Process a user request
response = agent.process_to_string("What is 15 multiplied by 32?")
print(response)

# Create a kitchen assistant agent with hardware disabled
kitchen_agent = create_agent(agent_type="kitchen_assistant", use_hardware=False)

# Process a kitchen-related query
response = kitchen_agent.process_to_string("Can you analyze this recipe: 2 cups flour, 1 cup sugar, 3 eggs?")
print(response)

# Create a vision-enabled agent
vision_agent = create_agent(
    agent_type="vision_agent", 
    use_hardware=True,
    capture_image="environment"  # Automatically capture from environment camera with each request
)

# Process a vision query (image is automatically captured and included)
response = vision_agent.process_to_string("What objects do you see in front of me?")
print(response)

# Use memory tools with the kitchen assistant
kitchen_agent = create_agent(agent_type="kitchen_assistant")
response = kitchen_agent.process_to_string("Where is the coffee located?")
print(response)  # Will access memory to find coffee location

response = kitchen_agent.process_to_string("Do you have any action plans for opening a jar?")
print(response)  # Will retrieve action plans from memory
```

## Interactive Mode

Run the interactive script to test your agents:

```bash
# Use the base agent
python -m llm_ai_agent.interactive

# Use the kitchen assistant agent with hardware disabled
python -m llm_ai_agent.interactive --agent kitchen_assistant --no-hardware

# Use the vision agent with environment camera enabled
python -m llm_ai_agent.interactive --agent vision_agent --capture-image environment

# List available agent configurations
python -m llm_ai_agent.interactive --list-configs
```

## Environment Variables

You can control hardware components using environment variables (in a `.env` file):

```bash
# Hardware component configuration
ENABLE_CAMERA=true    # Enable/disable camera tools
ENABLE_SPEAKER=true   # Enable/disable speaker tools
ENABLE_ARM=false      # Enable/disable robotic arm tools

# OpenRouter API configuration
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL_NAME=anthropic/claude-3-opus-20240229
```

This allows you to selectively enable only the hardware components you need or have available.

## Creating New Agent Types

1. Create a new YAML configuration file in `configs/agents/`
2. Define the agent's properties (system prompt, tools, etc.)
3. Optionally inherit from an existing agent type

Example:

```yaml
# Vision Agent Configuration
agent_type: vision_agent
description: A vision-enabled agent for kitchen assistance
version: 1.0.0
inherits_from: kitchen_assistant

# Custom system prompt
system_prompt: |
  You are a specialized recipe assistant that can help analyze recipes,
  suggest substitutions, and provide cooking guidance...

# Tool configuration
tools:
  include:
    - calculator
    - text_processor
    - speak
    - capture
    - get_action_plans
    - get_item_locations
```

## Extending with New Tools

Add new tools in the `tools.py` module:

```python
def new_tool(input_str: str) -> str:
    """
    Description of what the tool does.
    
    Args:
        input_str: Input for the tool
        
    Returns:
        Result of the tool operation
    """
    # Tool implementation
    return f"Processed: {input_str}"

# Add to the TOOLS dictionary
TOOLS["new_tool"] = new_tool
```

Then reference the tool in your agent configuration:

```yaml
tools:
  include:
    - new_tool
```

## Using Hardware

The framework supports hardware abstraction for robotic components:

```python
# Use real hardware
agent = create_agent(agent_type="kitchen_assistant", use_hardware=True)

# Use mock implementations (for testing)
agent = create_agent(agent_type="kitchen_assistant", use_hardware=False)
```

You can also control hardware via environment variables in your `.env` file:

```
ENABLE_CAMERA=true
ENABLE_SPEAKER=true
ENABLE_ARM=false
```

## Memory System

The framework includes tools for accessing structured memory stored in YAML files:

```python
from llm_ai_agent.tools import get_action_plans, get_action_positions, get_item_locations

# Get all predefined action plans
plans = get_action_plans()
print(plans)  # Displays all available action sequences

# Get stored robot arm positions
positions = get_action_positions()
print(positions)  # Displays coordinates for specific actions

# Get known item locations in the environment
items = get_item_locations()
print(items)  # Displays coordinates of known objects
```

Memory is organized into three primary categories:

1. **Action Plans** (`memory/action_plan.yaml`): Step-by-step procedures for completing tasks
2. **Action Positions** (`memory/action_position.yaml`): Stored coordinates for specific robot actions
3. **Item Locations** (`memory/item_location.yaml`): Coordinates of known objects in the environment

Agents can access this memory to:
- Retrieve predefined sequences for common tasks like opening jars
- Access specific arm positions for tasks
- Locate known items in the environment

Memory tools are automatically integrated into agents that include them in their tool configuration.

## Vision Capabilities

The framework now supports automatic image capture with each request through the `capture_image` setting:

```python
# Create an agent with environment camera enabled
agent = create_agent(
    agent_type="vision_agent", 
    capture_image="environment"
)

# Change the camera view dynamically
agent.capture_image = "wrist"  # Switch to wrist camera
agent.capture_image = "both"   # Use both cameras
agent.capture_image = ""       # Disable automatic image capture
```

Available camera options:
- `"environment"`: Use the environment camera (showing the general scene)
- `"wrist"`: Use the wrist-mounted camera (showing what's in front of the gripper)
- `"both"`: Use both cameras together
- `""` (empty string): Disable automatic image capture

This approach prevents the LLM from directly calling camera capture tools, ensuring all image capture is controlled through configuration.

## Configuration System

The agent configuration system uses YAML files in these directories:

- `configs/agents/`: Agent type definitions
- `configs/tools/`: Tool category definitions

Currently implemented configurations:
- `base_agent.yaml`: A general-purpose agent with basic tools
- `kitchen_assistant.yaml`: A specialized agent for kitchen assistance with hardware control
- `vision_agent.yaml`: A vision-enabled agent with camera capabilities
- `memory_tools.yaml`: Tool configurations for memory access capabilities

Tool configuration has been simplified - now you just need to include the tools you want to use:

```yaml
tools:
  include:
    - calculator
    - text_processor
    - speak
    - capture
    - move_home
    - get_action_plans
    - get_action_positions
    - get_item_locations
```

## LLM Response Handling

The system implements robust LLM response handling with:

1. JSON extraction from code blocks
2. Fallback parsing approaches for various response formats
3. Error recovery for None/empty responses
4. Tool execution based on parsed actions
5. Proper chat history management
6. Support for multimodal (text+image) inputs

## Future Development

Planned enhancements include:

- [ ] Additional specialized agent types
- [ ] Expanded tool categories for more capabilities
- [ ] Improved vision analysis for kitchen environments
- [ ] Better multi-step task planning and execution
- [ ] Enhanced personalization options
- [ ] Integration with external APIs for broader knowledge
- [ ] Memory update capabilities to modify stored knowledge