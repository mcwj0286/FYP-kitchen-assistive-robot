# Kitchen Assistive Robot AI Agent (LangChain Implementation)

This directory contains the LangChain implementation of the AI agent system for the kitchen assistive robot.

## Implementation Plan

1. **Basic Agent Setup**
   - Implement a foundation agent with LangChain
   - Add simple tools and test basic functionality
   - Establish proper environment configuration

2. **Tool Migration**
   - Convert hardware tools (robot arm, camera, speaker)
   - Implement API tools (search, weather, stocks)
   - Test each tool independently

3. **Action Planning Integration**
   - Create tools for action plan loading and execution
   - Integrate with existing YAML-based plans
   - Implement plan validation and error handling

4. **Advanced Features**
   - Add monitoring capabilities
   - Implement vision analysis with camera tools
   - Support complex multi-step tasks

5. **Testing and Optimization**
   - Develop comprehensive tests
   - Optimize performance
   - Improve error handling and recovery

## Architecture

We're implementing a multi-agent system using OOP principles with a class hierarchy:

```
BaseAgent
  ├── Core functionality (LLM setup, tool management, processing)
  │
  ├── KitchenAssistantAgent
  │     └── Kitchen-specific tools and prompts
  │
  ├── MonitoringAgent
  │     └── Monitoring-specific tools and prompts
  │
  └── ActionPlanningAgent
        └── Plan execution specific tools and prompts
```

This approach provides:
- **Extensibility**: Easy to add new specialized agents
- **Code Reuse**: Common functionality defined in BaseAgent
- **Consistent Interface**: Standard methods across all agents
- **Modularity**: Each agent focuses on specific responsibilities

## Agent Configuration System

We've implemented a YAML-based configuration system for flexible agent customization. This approach allows for:
- Defining different agent types with specific capabilities
- Customizing system prompts and tool sets without code changes
- Creating specialized agents for different tasks
- Managing agent configurations through version control

### File Structure

The agent configuration system follows this file structure:

```
llm_ai_agent/
├── configs/                        # Directory for all configuration files
│   ├── agents/                     # Agent-specific configurations
│   │   ├── base_agent.yaml         # Base agent configuration
│   │   ├── kitchen_assistant.yaml  # Kitchen assistant agent
│   │   ├── sous_chef.yaml          # Sous chef specialized agent
│   │   └── ...                     # Other agent types
│   │
│   ├── tools/                      # Tool definitions and categories
│   │   ├── hardware_tools.yaml     # Hardware tool definitions
│   │   ├── information_tools.yaml  # Information retrieval tools
│   │   └── planning_tools.yaml     # Planning and execution tools
│   │
│   └── prompts/                    # Reusable prompt templates
│       ├── base_prompts.yaml       # Common prompt components
│       └── specialized_prompts.yaml # Task-specific prompts
│
├── config_loader.py                # Configuration loading utilities
└── ...                             # Other project files
```

### YAML Configuration Format

Agent configurations use a structured YAML format that supports inheritance and customization:

#### Agent Configuration Example

```yaml
# configs/agents/kitchen_assistant.yaml
agent_type: kitchen_assistant
description: "Specialized agent for kitchen assistance with hardware control"
inherits_from: base_agent  # Inherit common properties from base agent

# System prompt (supports multi-line format)
system_prompt: |
  You are a helpful kitchen assistant AI that controls physical hardware devices.
  
  You have access to the following hardware capabilities:
  - Robot arm for manipulating objects in the kitchen
  - Cameras for observing the environment and objects
  - Speaker for communicating through voice
  
  Use your available tools appropriately to help the user with kitchen-related tasks.
  Always think step by step when handling complex cooking instructions or recipes.

# Tool configuration
tools:
  # Include tool categories
  categories:
    - hardware.camera
    - hardware.speaker
    - hardware.arm
    - information.basic
  
  # Explicitly include specific tools
  include:
    - calculator
    - text_processor
    - speak
    - capture_environment
  
  # Explicitly exclude specific tools
  exclude:
    - weather_tool  # Example tool to exclude

# Hardware requirements
hardware_required: true

# Model settings
model_defaults:
  temperature: 0.4
  max_tokens: 1000
```

#### Tool Categories Configuration

```yaml
# configs/tools/hardware_tools.yaml
categories:
  camera:
    description: "Camera and vision tools"
    tools:
      - capture_environment
      - capture_wrist
      - analyze_image
  
  speaker:
    description: "Speech and audio tools"
    tools:
      - speak
      - is_speaking
      - stop_speaking
  
  arm:
    description: "Robotic arm control tools"
    tools:
      - move_home
      - move_position
      - grasp
      - release
      - get_position
      - move_default
```

### Configuration Inheritance

Agents can inherit properties from other agents:

1. **Base Properties**: The parent agent provides default properties
2. **Tool Inheritance**: Child agents can inherit tools from parent agents
3. **Override Capability**: Child agents can override specific properties
4. **Tool Customization**: Add or remove specific tools

This allows for creating specialized agents while maintaining consistency.

### Creating a New Agent Type

To create a new agent type:

1. **Create a YAML configuration file** in `configs/agents/` directory
2. **Define the basic properties** (agent_type, description)
3. **Set the parent** using `inherits_from` if extending an existing agent
4. **Customize the system prompt** for the agent's specific role
5. **Configure the tool set** using categories, includes, and excludes
6. **Set any specific requirements** or model defaults

### Using the Configuration System

The agent configuration system is used in code as follows:

```python
from config_loader import AgentConfigLoader

# Load configuration
config_loader = AgentConfigLoader()

# Create an agent using configuration
agent = ConfigurableAgent(agent_type="kitchen_assistant")

# Process user query
response = agent.process_to_string("Can you help me with this recipe?")
```

The interactive script supports specifying the agent type:

```bash
# Run with a specific agent type
python interactive.py --agent kitchen_assistant

# Use a custom configuration file
python interactive.py --agent custom_agent --config path/to/config.yaml
```

## Structured Output Approach

We've implemented a structured output approach for our agents instead of using function calling, which ensures compatibility with a wider range of LLM providers like OpenRouter. 

The agent's responses follow this JSON format:
```json
// When using tools:
{
  "thought": "Internal reasoning about what to do",
  "action": "tool_name",
  "action_input": "input to the tool"
}

// When providing a direct response:
{
  "thought": "Internal reasoning about the response",
  "response": "Response to the user"
}
```

This ReAct (Reasoning + Acting) pattern provides:
- Clear separation between thinking and acting
- Transparent chain-of-thought reasoning
- Structured output that's easy to parse
- Compatibility with models that don't support function calling

## Hardware Integration

The Kitchen Assistant Agent integrates with the following hardware components:

### Camera System
- Environment camera: Provides a view of the overall kitchen environment
- Wrist camera: Mounted on the robot arm for detailed views of objects being manipulated
- Image analysis capabilities for basic scene understanding

### Speaker System
- Cross-platform text-to-speech functionality to communicate with users
- Automatic platform detection (macOS, Linux, Windows)
- Dynamic TTS command selection based on available tools
- Asynchronous speech processing with proper process management
- Audio playback capabilities for alerts and notifications

### Robotic Arm (Kinova)
- Cartesian movement capabilities in 3D space
- Angular control of joints
- Gripper control for grasping objects
- Position sensing and monitoring

All hardware interactions are implemented as tools that can be called by the agent, with built-in fallback to mock implementations when hardware is not available or for testing.

## Cross-Platform Support

The system has been designed to work across different operating systems:

### Development Environment (macOS)
- Uses native `say` command for text-to-speech
- Uses `afplay` for audio file playback
- Automatic detection of available commands

### Deployment Environment (Linux)
- Dynamically selects available TTS engines:
  - `espeak`: Lightweight speech synthesizer
  - `festival`: More advanced, higher quality speech
  - `pico2wave`: High-quality multilingual TTS
- Uses `aplay` or `play` for audio file playback
- Graceful fallback to mock implementation if no TTS engines are available

### Testing Tools
- Comprehensive cross-platform test script (`test_cross_platform_speech.py`)
- Platform detection and command availability reporting
- Independent component testing

## Contributing

### Adding New Agent Types

To contribute a new agent type:

1. **Create a new YAML configuration** file in `configs/agents/`
2. **Test the agent** using the interactive script
3. **Document the agent's purpose** and unique capabilities
4. **Submit a pull request** with your configuration and documentation

### Adding New Tools

To add new tools:

1. **Implement the tool function** following the standard pattern
2. **Add the tool to the appropriate category** in tool configuration files
3. **Update the tool registry** in `config_loader.py`
4. **Create tests** for the new tool
5. **Document the tool** with clear descriptions and examples

### Modifying System Prompts

When modifying system prompts:

1. **Make changes in the YAML configuration** files
2. **Test the changes** with various queries
3. **Version the prompts** if making significant changes
4. **Document the reasoning** behind prompt modifications

### Best Practices

- **Use inheritance** to avoid duplicating configuration
- **Keep system prompts focused** on the agent's specific role
- **Test tool combinations** to ensure compatibility
- **Version your configurations** to track changes over time
- **Document your reasoning** for specific design choices

## Progress

- [x] Project setup
- [x] Basic LLM configuration
- [x] BaseAgent implementation
- [x] KitchenAssistantAgent implementation
- [x] Interactive testing interface
- [x] Structured output approach implementation
- [x] Hardware tool integration
  - [x] Camera interface
  - [x] Speaker interface
    - [x] Cross-platform speech implementation (macOS/Linux)
    - [x] Platform-aware TTS engine selection
  - [x] Robotic arm interface
- [ ] YAML-based agent configuration system
- [ ] Action planning integration
- [ ] Advanced features
- [ ] Complete test suite

## Deployment Preparation

### Linux Environment Setup

To prepare a Linux system for deployment, follow these steps:

1. **Install Required Dependencies:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y python3-pip python3-dev

   # Install text-to-speech engines (at least one is required)
   sudo apt-get install -y espeak        # Lightweight option
   # OR
   sudo apt-get install -y festival      # More natural sounding
   # OR
   sudo apt-get install -y libttspico-utils # High quality multilingual

   # Install audio playback
   sudo apt-get install -y alsa-utils    # For aplay
   # OR
   sudo apt-get install -y sox           # For play command
   ```

2. **Install Python Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Test the Speech Functionality:**
   ```bash
   # Check which TTS engines are available
   python -m llm_ai_agent.tests.test_cross_platform_speech --info
   
   # Test the speech interface
   python -m llm_ai_agent.tests.test_cross_platform_speech
   ```

4. **Verify Other Hardware Components:**
   ```bash
   # Test camera interfaces if available
   python -m llm_ai_agent.tests.test_hardware --camera
   
   # Test robotic arm if available
   python -m llm_ai_agent.tests.test_hardware --arm
   ```

## Usage

### Basic Usage
```python
from base_agent import BaseAgent

# Create a simple agent
agent = BaseAgent(verbose=True)

# Process a user query
response = agent.process_to_string("What is the square root of 144?")
print(response)
```

### Creating Specialized Agents
```python
from base_agent import BaseAgent

class MySpecializedAgent(BaseAgent):
    def __init__(self, **kwargs):
        # Initialize with custom settings
        super().__init__(**kwargs)
        
        # After initialization, you can customize the prompt
        custom_context = "You are a specialized agent that..."
        self._system_prompt = custom_context
        
    def _get_available_tools(self) -> Dict[str, Callable]:
        # Get base tools
        tools = super()._get_available_tools()
        
        # Add specialized tools
        tools["my_custom_tool"] = my_custom_tool_function
        
        return tools
```

### Using Hardware Tools
```python
from base_agent import KitchenAssistantAgent

# Create a kitchen assistant agent with hardware capabilities
agent = KitchenAssistantAgent(verbose=True)

# Process commands that use hardware
response = agent.process_to_string("Can you move the robot arm to grasp the cup?")
print(response)

# The agent will automatically use hardware tools when appropriate:
# - Camera tools: capture_environment, capture_wrist, analyze_image
# - Speaker tools: speak, is_speaking, stop_speaking
# - Robotic arm tools: move_home, move_position, grasp, release, get_position, move_default
```

### Using Configuration-Based Agents
```python
from config_loader import AgentConfigLoader
from base_agent import ConfigurableAgent

# Load configuration
config_loader = AgentConfigLoader()

# List available agent types
agent_types = config_loader.get_available_agent_types()
print(f"Available agent types: {agent_types}")

# Create a configurable agent
agent = ConfigurableAgent(agent_type="kitchen_assistant")

# Process user query
response = agent.process_to_string("Can you help me with this recipe?")
print(response)
```

### Interactive Mode
```bash
# Run the interactive testing script
python interactive.py

# Run with the kitchen assistant agent
python interactive.py --agent kitchen_assistant

# Run with a custom configuration file
python interactive.py --agent custom_agent --config path/to/config.yaml
```

## Cross-Platform Speech Interface

The system includes a cross-platform speech interface that works on both macOS and Linux:

- On macOS, the system uses the built-in `say` command
- On Linux, the system uses the `espeak` text-to-speech engine

Test scripts are available in the `tests` directory to verify functionality on both platforms.

## Installation

### macOS

No additional setup is required for speech functionality on macOS.

### Linux

Run the `linux_setup.sh` script in the project root to install the required dependencies:

```bash
chmod +x linux_setup.sh
./linux_setup.sh
```

This will install the `espeak` package required for text-to-speech functionality.

## YAML-Based Agent Configuration System

The Kitchen Assistant Robot uses a flexible YAML-based configuration system for defining and customizing different types of agents. This approach separates the agent's configuration (prompts, tools, etc.) from its implementation code, allowing for easier modification and extension.

### Key Benefits

- **Separation of Configuration from Code**: Modify agent behavior without changing code
- **Inheritance**: Create specialized agents that build on base configurations
- **Centralized Tool Management**: Define tools once and reuse across different agent types
- **Easy Customization**: Adjust prompts, tool availability, and model parameters through configuration files
- **Versioning**: Track changes to agent configurations over time

### File Structure

The configuration files are organized as follows:

```
llm_ai_agent/configs/
├── agents/              # Agent type definitions
│   ├── base_agent.yaml  # Base agent configuration
│   └── kitchen_assistant.yaml  # Kitchen-specific agent
├── tools/               # Tool category definitions
│   ├── hardware_tools.yaml     # Hardware-related tools
│   └── information_tools.yaml  # Information processing tools
└── prompts/             # Optional separate prompt templates
    └── kitchen_tasks.yaml      # Task-specific prompts
```

### YAML Configuration Format

#### Agent Configuration (agents/*.yaml)

```yaml
agent_type: "kitchen_assistant"  # Unique identifier for this agent type
description: "Kitchen assistant with hardware control capabilities"
version: "1.0.0"
inherits_from: "base_agent"  # Optional parent configuration to extend

# System prompt that defines the agent's behavior
system_prompt: |
  You are a kitchen assistant AI that can control a robotic arm
  and provide guidance on cooking tasks.
  
  Always break down complex tasks into simple steps.
  When using hardware tools, ensure safety by checking
  surroundings first.

# Tool configuration
tools:
  # Tool categories to include
  categories:
    - "hardware_tools.camera"
    - "hardware_tools.speaker"
    - "hardware_tools.arm"
    - "information_tools.basic"
  
  # Additional individual tools to include
  include:
    - "calculator"
    - "text_processor"
  
  # Tools to exclude (overrides categories and include)
  exclude: []

# Whether hardware is required for this agent
hardware_required: true

# Default model parameters
model_defaults:
  temperature: 0.2
  max_tokens: 2048
  model_name: "gpt-4"
```

#### Tool Categories (tools/*.yaml)

```yaml
# Define categories of related tools
categories:
  camera:
    description: "Tools for environment perception"
    tools:
      - name: "capture_environment"
        description: "Capture an image from the environment camera"
        hardware_required: true
        parameters:
          - name: "resolution"
            type: "string"
            description: "Resolution to capture (low, medium, high)"
            optional: true
            default: "medium"
      
      - name: "analyze_image"
        description: "Analyze the most recently captured image"
        hardware_required: false
  
  speaker:
    description: "Tools for user communication"
    tools:
      - name: "speak"
        description: "Convert text to speech"
        hardware_required: true
        parameters:
          - name: "text"
            type: "string"
            description: "Text to speak"
```

### Creating New Agent Types

To create a new agent type:

1. Create a new YAML file in the `configs/agents/` directory
2. Specify a unique `agent_type` name
3. Optionally specify `inherits_from` to build on an existing agent type
4. Define the `system_prompt` that guides the agent's behavior
5. Configure the `tools` section to specify which tools the agent can use
6. Set appropriate default model parameters

### Adding New Tool Categories

To add new tools:

1. Create or edit a YAML file in the `configs/tools/` directory
2. Add a new category under the `categories` key
3. For each tool, specify:
   - `name`: Unique identifier
   - `description`: What the tool does
   - `hardware_required`: Whether hardware is needed
   - `parameters`: Input parameters with their types and descriptions

### Using the Configuration System

The `AgentConfigLoader` class (in `config_loader.py`) handles loading and processing these YAML configurations. To create an agent using the configuration system:

```python
from config_loader import AgentConfigLoader

# Load configurations
config_loader = AgentConfigLoader()

# Get configuration for a specific agent type
kitchen_agent_config = config_loader.get_agent_config("kitchen_assistant")

# Get the list of tools for the agent
tools = config_loader.get_agent_tools("kitchen_assistant")

# Get the system prompt
system_prompt = config_loader.get_agent_system_prompt("kitchen_assistant")

# Create the agent with the loaded configuration
agent = KitchenAssistantAgent(
    system_prompt=system_prompt,
    tools=tools,
    **kitchen_agent_config.get("model_defaults", {})
)
```

### Future Extensions

The configuration system is designed to be extended with:

- **Action Planning**: Define sequences of actions for common tasks
- **Multi-Step Reasoning**: Configure different reasoning strategies
- **Tool Permissions**: Control which tools are available in different contexts
- **Personalization**: Customize agent behavior for different users
