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

## Multi-turn Tool Calling Approaches

The framework supports sequential tool calling to handle complex tasks that require multiple tools to be used in a coordinated sequence. We present two approaches to implementing multi-turn tool calling:

### 1. Enhanced Multi-turn Tool Calling

This approach extends the existing agent architecture to support multiple iterations of tool calls within a single conversation turn.

#### Design Overview

```
┌──────────────────────────────────────────────────────┐
│                  ConfigurableAgent                   │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────┐  ┌─────────────────────────────────┐   │
│  │ Planning │  │ Iterative Tool Execution Loop  │   │
│  │  Phase   │──▶  ┌─────┐  ┌─────┐  ┌─────┐     │   │
│  └─────────┘  │  │Tool1 │─▶│Tool2 │─▶│Tool3 │─...│   │
│               │  └─────┘  └─────┘  └─────┘     │   │
│               └─────────────────────────────────┘   │
│                              │                       │
│                              ▼                       │
│                    ┌─────────────────┐              │
│                    │ Final Response  │              │
│                    └─────────────────┘              │
└──────────────────────────────────────────────────────┘
```

#### Implementation Details

1. **Configuration Options**:
   ```python
   def __init__(
       self,
       agent_type: str = "base_agent",
       # ... existing parameters ...
       max_tool_iterations: int = 5,  # Maximum iterations of tool calls
       enable_planning: bool = True,  # Whether to enable planning phase
   ):
   ```

2. **Planning Phase** (Optional):
   - An initial API call generates a structured plan for tool usage
   - The plan outlines which tools will likely be needed and in what sequence
   - This becomes part of the conversation context

3. **Iterative Tool Execution Loop**:
   - Runs for `max_tool_iterations` iterations or until completion
   - Each iteration:
     - Gets LLM response given current state
     - Executes any requested tools
     - Adds results to conversation context
     - Asks LLM if more tool calls are needed

4. **Final Response Generation**:
   - After tools are complete or max iterations reached
   - Summarizes findings from all tool calls
   - Provides a comprehensive response to the user

5. **Response Format for Iterations**:
   ```json
   {
     "thought": "Reasoning about the current state and what to do next",
     "reply": "Response to user (only used if is_complete is true)",
     "is_complete": false,  // Set to true when ready to respond
     "tool_calls": [
       {
         "tool_name": "NextToolToCall",
         "parameters": { ... }
       }
     ]
   }
   ```

#### Pros:
- **Context Continuity**: Maintains a single conversation context throughout
- **Dynamic Planning**: Can adjust the sequence of tool calls based on intermediate results
- **Efficient**: Fewer API calls compared to hierarchical approaches
- **Simpler Implementation**: Extends existing agent architecture
- **Centralized Error Handling**: All errors managed in one place
- **Easy Debugging**: Single conversation flow is easier to trace

#### Cons:
- **Context Window Limits**: Long sequences may exceed LLM context windows
- **Less Specialized Prompting**: Uses one system prompt for all tool calling stages
- **Potentially Less Structured**: Planning is optional and less rigid

### 2. Hierarchical Agents-as-Tools

This approach creates a hierarchy of specialized agents that act as tools for higher-level agents, each handling a specific part of the reasoning process.

#### Design Overview

```
┌────────────────────────────────────────────────────────┐
│                 Comprehensive Agent                     │
└──────────────────┬─────────────────────┬───────────────┘
                   │                     │
                   ▼                     ▼
┌────────────────────────┐    ┌─────────────────────────┐
│  Action Planning Agent  │    │  Action Execution Agent │
│  (Generate Plans)       │    │  (Execute Steps)        │
└──────────┬─────────────┘    └───────────┬─────────────┘
           │                              │
           ▼                              ▼
┌────────────────────────┐    ┌─────────────────────────┐
│ Memory & Knowledge     │    │ Hardware & Basic Tools   │
│ Tools                  │    │                         │
└────────────────────────┘    └─────────────────────────┘
```

#### Implementation Details

1. **Agent Hierarchy**:
   - **Comprehensive Agent**: Handles user interaction and coordinates between specialized agents
   - **Action Planning Agent**: Converts high-level user requests into structured action plans
   - **Action Execution Agent**: Converts action steps into specific tool calls

2. **Action Planning Agent**:
   - Takes high-level instructions (e.g., "open the jar for me")
   - Accesses memory tools for existing plans if needed
   - Returns a structured plan with ordered steps
   - Example:
     ```json
     {
       "goal": "Open a jar for the user",
       "steps": [
         {"step_num": 1, "description": "Move to open jar position"},
         {"step_num": 2, "description": "Announce user to put the jar on the gripper"},
         // ... additional steps
       ]
     }
     ```

3. **Action Execution Agent**:
   - Takes one step from the plan (e.g., "Move to open jar position")
   - Breaks it down into required tool calls (e.g., get position, move arm)
   - Executes each tool and monitors results
   - Reports success/failure back to comprehensive agent
   - Example:
     ```json
     {
       "step": "Move to open jar position",
       "tools_used": ["get_action_positions", "move_position"],
       "result": "Successfully moved to jar opening position",
       "success": true
     }
     ```

4. **Information Flow**:
   ```
   User → Comprehensive Agent → Action Planning Agent → Comprehensive Agent
   → Action Execution Agent (Step 1) → Comprehensive Agent
   → Action Execution Agent (Step 2) → ... → Final Response
   ```

#### Pros:
- **Clear Separation of Concerns**: Each agent is specialized for its specific task
- **Reusable Components**: Planning and execution agents can be used across different systems
- **Specialized Prompting**: Each agent has optimized system prompts for its role
- **Natural Hierarchical Modeling**: Mirrors how humans break down complex tasks
- **Scalable for Complex Tasks**: Well-suited for multi-step, multi-tool procedures
- **Potential for Parallelization**: Multiple execution agents could work simultaneously

#### Cons:
- **API Call Overhead**: Each level in hierarchy requires additional API calls
- **Context Fragmentation**: Each agent has limited view of the overall conversation
- **Complex State Management**: Requires careful handling of state between agents
- **Implementation Complexity**: More code and configuration needed
- **Error Propagation Challenges**: Errors must be properly propagated up the chain

### Implementation Recommendations

Choose the approach that best fits your needs:

- **Use Enhanced Multi-turn Tool Calling if**:
  - You need a quicker implementation with less overhead
  - Token efficiency and context preservation are important
  - Your tasks require flexible, dynamic tool sequences
  - You want simpler error handling and debugging

- **Use Hierarchical Agents-as-Tools if**:
  - Your tasks naturally fit a planning → execution model
  - You need highly specialized reasoning at different levels
  - You want maximum reusability of components
  - The complexity benefits from clear separation of concerns
  - You have complex robotic or multi-step workflows

- **Consider a Hybrid Approach**: Keep the overall conversation in one agent, but implement specialized execution tools that can handle multi-step procedures internally.

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