# AI Agent System for Kitchen Assistive Robot

This package provides a comprehensive AI agent system for controlling a kitchen assistive robot. The system combines language model intelligence with hardware control capabilities to create an interactive robot assistant that can help with kitchen tasks.

## System Overview

The system architecture consists of these main components:

1. **Core Agent System**
   - **BaseAgent**: LLM-powered AI core that processes natural language, analyzes images, and controls tools
   - **Hardware Tools**: Interfaces for robot arm, cameras, and speaker
   - **API Tools**: Web service integrations for search, weather, stock prices, etc.
   - **Logging System**: Comprehensive tracking of all API interactions and tool usage

2. **Hardware Integration**
   - **Kinova Robot Arm**: Precise control for kitchen manipulation tasks
   - **Camera System**: Visual perception for object recognition and environmental awareness
   - **Speaker System**: Voice feedback and communication with users

3. **Action Plan System**
   - **Predefined Action Sequences**: YAML-defined workflows for common kitchen tasks
   - **Execution Engine**: Step-by-step execution with error handling and recovery
   - **Visual Verification**: Uses cameras and LLM analysis to verify task progress

4. **Mock Tools System**
   - **Hardware Simulation**: Test robot functionality without physical hardware
   - **Scenario Generation**: Simulate different kitchen situations for testing
   - **Visual Debugging**: Generate debug images of simulated hardware states

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
SYSTEM_PROMPT="You are a helpful assistant that can control kitchen hardware devices."

# Optional API keys for tools
OPENWEATHER_API_KEY=your_openweather_api_key  # For weather tool
ALPHA_VANTAGE_API_KEY=your_alphavantage_api_key  # For stock price tool
SEARCH_API_KEY=your_search_api_key  # For Google search tool
GOOGLE_CSE_ID=your_google_cse_id  # For Google search tool
```

## Core Functionality

### Hardware Control

The system interfaces with the following hardware:

1. **Kinova Robot Arm**: 
   - Position control in Cartesian and joint space
   - Gripper control for grasping objects
   - Pre-defined movement sequences for common tasks
   - Safety limits and collision prevention

2. **Cameras**: 
   - Multi-camera capture for different views
   - Real-time image analysis using LLM vision capabilities
   - Object detection for ingredients, tools, and kitchen items
   - Task verification through visual feedback

3. **Speaker**: 
   - Text-to-speech for user instructions and feedback
   - Multiple voice options and control parameters
   - Volume, rate, and pitch adjustment
   - Platform-specific optimizations for macOS and Linux

All hardware interfaces are managed by the `HardwareToolManager` which handles initialization and cleanup. **All hardware is fully initialized at startup**, ensuring immediate availability of all devices without delay when they're first used.

### LLM Intelligence

The agent uses OpenRouter to access powerful language models with these capabilities:

- **Natural Language Understanding**: Process user requests about kitchen tasks
- **Image Analysis**: Identify objects, assess situations, and verify task completion
- **Task Planning**: Break down complex requests into executable steps
- **Tool Usage**: Select and execute appropriate hardware actions
- **Error Handling**: Detect issues and implement recovery strategies
- **Contextual Awareness**: Maintain conversation context during complex interactions

### Action Plan System

The action plan system enables predefined task sequences:

1. **Action Plan Definition**: YAML configuration specifies goals and step sequences
2. **Predefined Locations**: Saved positions for robot arm movements
3. **Step Types**:
   - Movement commands for robot positioning
   - Announcements for user instructions
   - Wait conditions with visual verification
   - Gripper actions for manipulation
   - Complex sequences for tasks like opening jars

Example tasks include:
- Opening jars for users with limited hand strength
- Retrieving items from shelves
- Assisting with food preparation
- Monitoring cooking processes

## Usage Modes

### Interactive Mode

```bash
python -m llm_agent.hardware_agent_example
```

This starts an interactive session where you can:
- Give natural language commands to the robot
- Request camera captures and analysis
- Control robot arm and gripper directly
- Use the speaker for voice output

### Action Plan Mode

```bash
python -m llm_agent.hardware_agent_example --action-plan
```

This mode:
1. Lists available predefined action plans
2. Allows selection of a plan to execute
3. Executes steps in sequence with progress updates
4. Handles errors with retry, skip, or abort options

### Demo Mode

```bash
python -m llm_agent.hardware_agent_example --demo
```

Runs a demonstration sequence showing all capabilities:
1. Speaking introduction
2. Camera capture and analysis
3. Robot arm movement
4. Integration of multiple tools

### Mock Testing Mode

```bash
python -m llm_agent.hardware_agent_example --mock
```

The mock mode allows development and testing without physical hardware:
- Simulates robot arm movements and positions
- Generates synthetic camera images for different scenarios
- Emulates speaker functionality
- Creates debug visualizations for verification

Combine with other modes for comprehensive testing:
```bash
# Test action plans with mock hardware
python -m llm_agent.hardware_agent_example --action-plan --mock
```

## Testing

Comprehensive test scripts verify all system components:

```bash
# Test hardware tools (speaker, API logging)
python -m test_hardware_tools

# Test mock tools functionality
python -m test_mock_tools

# Test specific mock tool groups
python -m test_mock_tools --hardware  # Test mock hardware
python -m test_mock_tools --agent     # Test agent with mock tools
python -m test_mock_tools --action-plan # Test action plan execution
```

## Development Features

### Hardware Initialization

The system immediately initializes all hardware at startup:

1. **Robot Arm**: Connects and calibrates with the Kinova arm on system launch
2. **Cameras**: Initializes all camera devices and verifies their functionality 
3. **Speaker**: Sets up the text-to-speech engine with default voice and parameters

This approach ensures that all hardware is ready for use without delays when commands are issued. The startup sequence includes:
- Hardware device detection
- Initialization of all available devices
- Verification of functionality
- Detailed status reporting

### Mock Scenario Control

For testing image detection, control the mock camera view:

```python
# Change camera scenario during testing
os.environ["MOCK_CAMERA_SCENARIO"] = "jar_on_table"  # Shows jar on table
os.environ["MOCK_CAMERA_SCENARIO"] = "jar_on_gripper"  # Shows jar on gripper
os.environ["MOCK_CAMERA_SCENARIO"] = "jar_opened"  # Shows opened jar
```

### Comprehensive Logging

- **API Call Logging**: All LLM interactions recorded in `llm_api_calls.log`
- **Response Timing**: Tracking of API response latency
- **Error Tracking**: Automatic recording of failures with context
- **Tool Usage Statistics**: Monitor which tools are used most frequently

## Troubleshooting

### Speaker Issues

If the speaker is not working:

- On macOS, check available voices with `say -v '?'`
- Default voice has been configured to "Samantha" for better reliability
- Ensure volume is not muted
- Check for proper permissions

### LLM API Issues

If experiencing issues with the LLM API:

- Check your API key in the .env file
- Review the logs in `llm_api_calls.log`
- Ensure you have an active internet connection
- The system implements automatic retry with backoff for transient errors

### Camera and Robot Arm Problems

- Verify USB connections and device permissions
- Check for hardware-specific error messages
- Mock mode can help isolate whether issues are software or hardware related
- If devices are not detected at startup, try restarting the application

## License

MIT License

For more details on specific components:
- Action Plan System: See [Action Plan Execution System](actions_config/README.md)
- Mock Tools: See test scripts in `test_mock_tools.py`
- Hardware Integration: See `hardware_tools.py` and related documentation
