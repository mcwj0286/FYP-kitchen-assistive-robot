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

## System Architecture

The framework follows a modular, configuration-driven architecture:

- `agents.py`: Central module for creating and managing agents
- `configurable_agent.py`: The main agent implementation that loads behavior from configuration
- `config_loader.py`: Utilities for loading and processing YAML configurations
- `tools.py`: Collection of basic tools that agents can use
- `interactive.py`: Command-line interface for interacting with agents
- `hardware_tools.py`: Hardware interface implementations with mock fallbacks

### Key Design Principles

1. **Configuration over Code**: Agent behavior is defined in YAML files
2. **Single Agent Implementation**: `ConfigurableAgent` is the only agent class needed
3. **Hardware Abstraction**: Seamlessly switch between real and mock hardware
4. **Modular Tool System**: Easily add, remove, or customize tools
5. **Robust Error Handling**: Graceful handling of LLM response failures
6. **Controlled Hardware Access**: Camera capture managed via configuration rather than direct tool calls
7. **Multimodal Capabilities**: Support for text and image inputs to the LLM

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
  categories:
    - information_tools.basic
  include:
    - calculator
    - text_processor
  exclude: []

# Model configuration
model_defaults:
  temperature: 0.3
  max_tokens: 2048
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

Tool categories include:
- `hardware_tools.camera`: Tools for environment perception (now controlled via configuration)
- `hardware_tools.speaker`: Tools for user communication
- `hardware_tools.arm`: Tools for robotic arm control
- `information_tools.basic`: Basic information processing tools
- `information_tools.search`: Information search and retrieval tools

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