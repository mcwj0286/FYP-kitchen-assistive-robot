# AI Agent System for Robot Control

This package provides a comprehensive AI agent system that can control hardware devices and make API calls. The system includes a base agent architecture, hardware interfaces, and a robust logging system.

## Overview

The system consists of the following main components:

1. **BaseAgent**: Core AI agent that can process natural language inputs, make API calls, and use tools
2. **Hardware Tools**: Interfaces for controlling physical devices (robot arm, cameras, speaker)
3. **API Tools**: Interfaces for making web API calls (search, weather, stock prices, etc.)
4. **Logging System**: Comprehensive logging of all LLM API interactions

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file with the following variables:

```
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key
MODEL_NAME=anthropic/claude-3-opus-20240229
SYSTEM_PROMPT="You are a helpful assistant that can control hardware devices."

# Optional API keys for tools
OPENWEATHER_API_KEY=your_openweather_api_key  # For weather tool
ALPHA_VANTAGE_API_KEY=your_alphavantage_api_key  # For stock price tool
SEARCH_API_KEY=your_search_api_key  # For Google search tool
GOOGLE_CSE_ID=your_google_cse_id  # For Google search tool
```

## Hardware Integration

The system can interface with the following hardware:

1. **Kinova Robot Arm**: Controlled via the `RobotArmTool`
2. **Cameras**: Accessed through the `CameraTool`
3. **Speaker**: Text-to-speech and audio playback via `SpeakerTool`

All hardware interfaces are managed by the `HardwareToolManager` which handles initialization and cleanup.

## LLM API Integration

The agent uses OpenRouter to access powerful language models. Key features include:

- Text-based prompt processing
- Image analysis capabilities
- Tool use for extending functionality
- Robust error handling
- Comprehensive logging

## Logging System

The system includes a comprehensive logging framework that records all interactions with the LLM API:

- All requests and responses are logged to `llm_api_calls.log`
- Log entries include unique request IDs for tracking
- Image data is summarized to avoid huge logs
- Response times are recorded
- Tool usage is tracked

## Action Plan Execution

The system includes an action plan execution feature that allows you to define structured steps for the robot to follow. This is useful for creating repeatable tasks that require a sequence of operations.

### Running Action Plans

To run the system in action plan execution mode:

```bash
python -m llm_agent.hardware_agent_example --action-plan
```

This will list available action plans defined in the configuration files and allow you to select one to execute.

### Defining Action Plans

Action plans are defined in YAML files located in the `actions_config` directory:

- `action_plan.yaml`: Contains the definitions of action plans, including goals and step sequences
- `predefined_loaction.yaml`: Contains predefined locations for robot arm movements

Example action plan for opening a jar:

```yaml
open_jar_assistance:
  goal: "Open a jar for the user"
  steps:
    - step_num: 1
      description: "Move to open jar position"
    - step_num: 2
      description: "Announce user to put the jar on the gripper"
    - step_num: 3
      description: "Wait for user to put the jar on the gripper"
    - step_num: 4
      description: "Close the gripper"
    - step_num: 5
      description: "Roll left to open the jar"
```

For more details, see the [Action Plan Execution System documentation](actions_config/README.md).

## Usage Examples

### Basic Usage

```python
from llm_agent.ai_agent import BaseAgent
from dotenv import load_dotenv

load_dotenv()

# Initialize the agent
agent = BaseAgent()

# Process a text prompt
response = agent.process("What's the weather like in New York?")
print(response)

# Process an image
image_response = agent.process_with_image(
    prompt="What can you see in this image?",
    image_path="path/to/image.jpg"
)
print(image_response)
```

### Using Hardware Tools

```python
from llm_agent.ai_agent import BaseAgent
from llm_agent.hardware_tools import HardwareToolManager

# Initialize the hardware tool manager
tool_manager = HardwareToolManager()

# Initialize the agent
agent = BaseAgent()

# Add hardware tools to the agent
tools = tool_manager.get_all_tools()
for tool in tools:
    agent.add_tool(tool)

# Process a prompt that might use hardware tools
response = agent.process(
    "Take a picture and tell me what you see.",
    use_tools=True
)
print(response)

# Clean up
tool_manager.close_all()
```

## Testing

You can test the system using the provided test script:

```bash
python -m test_hardware_tools
```

This script tests:
- The speaker functionality
- LLM API call logging

### Mock Tools for Testing

The system includes a mock tools mode for testing without physical hardware. This allows you to test the logic and workflows without connecting to actual cameras, robot arms, or speakers.

#### Running in Mock Mode

To use the mock tools:

```bash
python -m llm_agent.hardware_agent_example --mock
```

You can also combine this with other modes:

```bash
# Run action plan execution with mock tools
python -m llm_agent.hardware_agent_example --action-plan --mock

# Run demo sequence with mock tools
python -m llm_agent.hardware_agent_example --demo --mock
```

#### Testing with Mock Tools

A dedicated test script is available to verify mock tool functionality:

```bash
python -m test_mock_tools

# Or run specific test groups
python -m test_mock_tools --hardware  # Test only mock hardware tools
python -m test_mock_tools --agent     # Test agent with mock tools
python -m test_mock_tools --action-plan # Test action plan execution
```

#### Mock Implementation Features

The mock tools provide realistic simulations:

1. **Robot Arm**: Simulates movements, gripper actions, and provides visual representations of the arm state.
2. **Camera**: Generates test images for different scenarios (empty scene, jar on table, jar on gripper, etc.) and handles image analysis.
3. **Speaker**: Simulates text-to-speech functionality with different voices and speech parameters.

Mock tools store debug images in the `debug_mock/` and `debug_images/` directories for inspection.

#### Switching Mock Scenarios

For testing image detection flows, you can control which scenario the mock camera shows:

```python
# Change the camera view to show a jar on the table
os.environ["MOCK_CAMERA_SCENARIO"] = "jar_on_table"

# Switch to jar on gripper
os.environ["MOCK_CAMERA_SCENARIO"] = "jar_on_gripper"

# Switch to opened jar
os.environ["MOCK_CAMERA_SCENARIO"] = "jar_opened"
```

The action plan executor will automatically prompt for scenario changes during interactive testing.

## Troubleshooting

### Speaker Issues

If the speaker is not working:

- On macOS, check available voices with `say -v '?'`
- Ensure volume is not muted
- Check for proper permissions

### LLM API Issues

If experiencing issues with the LLM API:

- Check your API key in the .env file
- Review the logs in `llm_api_calls.log`
- Ensure you have an active internet connection

## License

MIT License
