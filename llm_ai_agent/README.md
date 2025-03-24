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
   python llm_ai_agent/test_cross_platform_speech.py --info
   
   # Test the speech interface
   python llm_ai_agent/test_cross_platform_speech.py
   ```

4. **Verify Other Hardware Components:**
   ```bash
   # Test camera interfaces if available
   python llm_ai_agent/test_hardware.py --camera
   
   # Test robotic arm if available
   python llm_ai_agent/test_hardware.py --arm
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

### Interactive Mode
```bash
# Run the interactive testing script
python interactive.py

# Run with the kitchen assistant agent (has hardware capabilities)
python interactive.py --agent kitchen
```
