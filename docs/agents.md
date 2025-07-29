# AI Agent Configuration Guide

This guide explains how to configure, customize, and deploy the AI agents in the Kitchen Assistive Robot system. The multi-agent framework provides flexible, hierarchical task execution with natural language interaction.

## Agent System Overview

The system uses a **configuration-driven architecture** where agents are defined by YAML files rather than hardcoded classes. This allows for rapid prototyping and deployment of specialized agents without code changes.

### Core Principles

1. **Single Implementation**: All agents use the `ConfigurableAgent` class
2. **YAML Configuration**: Behavior defined in configuration files
3. **Tool-Based Capabilities**: Agents gain abilities through tool inclusion
4. **Hierarchical Communication**: Agents can call other agents as tools
5. **Memory Integration**: Persistent knowledge storage and retrieval

## Agent Types

### 1. Base Agent

The foundation agent with basic reasoning capabilities.

```yaml
# configs/agents/base_agent.yaml
agent_type: base_agent
description: A general-purpose agent with basic reasoning capabilities
version: 1.0.0

system_prompt: |
  You are a helpful AI assistant. You can perform calculations, process text,
  and help users with various tasks. Always be clear, concise, and accurate
  in your responses.

tools:
  include:
    - calculator
    - text_processor

memory_access: false
hardware_enabled: false
multimodal_capable: false
```

**Usage:**
```python
from llm_ai_agent.agents import create_agent

# Create base agent
agent = create_agent(agent_type="base_agent")
response = agent.process_to_string("What is 15 multiplied by 32?")
print(response)  # Output: 480
```

### 2. Kitchen Assistant Agent

Specialized for kitchen tasks with hardware control capabilities.

```yaml
# configs/agents/kitchen_assistant.yaml
agent_type: kitchen_assistant
description: A specialized agent for kitchen assistance with hardware control
version: 1.0.0
inherits_from: base_agent

system_prompt: |
  You are a helpful kitchen assistant robot. You can help users with kitchen 
  tasks including:
  - Object manipulation (grasping, moving, placing items)
  - Recipe analysis and suggestions
  - Cooking guidance and tips
  - Kitchen safety advice
  
  You have access to:
  - Robotic arm for physical manipulation
  - Cameras for visual perception
  - Speech synthesis for communication
  - Memory system with stored knowledge about the kitchen environment
  
  Always prioritize safety and ask for confirmation before performing 
  potentially dangerous actions.

tools:
  include:
    - calculator
    - text_processor
    - speak
    - move_position
    - move_home
    - get_action_plans
    - get_action_positions
    - get_item_locations

memory_access: true
hardware_enabled: true
multimodal_capable: false
```

**Usage:**
```python
# Create kitchen assistant
kitchen_agent = create_agent(
    agent_type="kitchen_assistant",
    use_hardware=True  # Enable real hardware
)

# Example interactions
response = kitchen_agent.process_to_string("Can you help me open a jar?")
response = kitchen_agent.process_to_string("Where is the coffee located?")
response = kitchen_agent.process_to_string("Move the cup to the plate")
```

### 3. Vision Agent

Enhanced with visual perception capabilities.

```yaml
# configs/agents/vision_agent.yaml
agent_type: vision_agent
description: A vision-enabled agent for kitchen assistance
version: 1.0.0
inherits_from: kitchen_assistant

system_prompt: |
  You are a vision-enabled kitchen assistant robot. In addition to all kitchen
  assistance capabilities, you can:
  - Analyze visual scenes and identify objects
  - Understand spatial relationships between items
  - Provide detailed descriptions of what you see
  - Help with visual cooking tasks like monitoring food preparation
  
  You receive images automatically and can see the current state of the 
  kitchen environment. Use this visual information to provide more accurate
  and helpful assistance.

tools:
  include:
    - calculator
    - text_processor
    - speak
    - move_position
    - move_home
    - capture  # Vision capabilities
    - get_action_plans
    - get_action_positions
    - get_item_locations

memory_access: true
hardware_enabled: true
multimodal_capable: true
```

**Usage:**
```python
# Create vision agent with automatic image capture
vision_agent = create_agent(
    agent_type="vision_agent",
    use_hardware=True,
    capture_image="environment"  # Auto-capture from environment camera
)

# Vision-based interactions
response = vision_agent.process_to_string("What objects do you see?")
response = vision_agent.process_to_string("Is the stove turned on?")
response = vision_agent.process_to_string("How many cups are on the counter?")
```

### 4. Specialized Agents

You can create task-specific agents for specialized scenarios:

#### Object Manipulation Agent
```yaml
# configs/agents/object_manipulation_agent.yaml
agent_type: object_manipulation_agent
description: Specialized agent for object manipulation tasks
version: 1.0.0
inherits_from: kitchen_assistant

system_prompt: |
  You are a specialized object manipulation robot. Your primary function is
  to accurately grasp, move, and place objects in the kitchen environment.
  
  You excel at:
  - Precise object grasping with various approach strategies
  - Safe object transportation between locations
  - Adaptive grip adjustment based on object properties
  - Collision avoidance during manipulation
  
  Always announce your actions and ask for confirmation when working near
  fragile items or when uncertainty exists about the desired outcome.

tools:
  include:
    - move_position
    - get_action_positions
    - get_item_locations
    - capture
    - speak

focus_area: "manipulation"
safety_priority: "high"
```

#### Recipe Assistant Agent
```yaml
# configs/agents/recipe_assistant.yaml
agent_type: recipe_assistant
description: Specialized agent for recipe guidance and cooking assistance
version: 1.0.0
inherits_from: base_agent

system_prompt: |
  You are a specialized recipe assistant. You help users with:
  - Recipe analysis and ingredient substitutions
  - Cooking technique explanations
  - Timing and preparation guidance
  - Nutritional information and dietary adaptations
  
  You have extensive knowledge of cooking methods, ingredient properties,
  and culinary best practices. Always provide clear, step-by-step guidance.

tools:
  include:
    - calculator
    - text_processor
    - speak
    - get_action_plans  # For cooking procedures

focus_area: "cooking_guidance"
knowledge_domain: "culinary"
```

## Tool System

### Core Tool Categories

#### Basic Tools
```yaml
tools:
  include:
    - calculator      # Mathematical calculations
    - text_processor  # Text analysis and manipulation
```

#### Hardware Tools
```yaml
tools:
  include:
    - speak          # Text-to-speech output
    - capture        # Camera image capture
    - move_position  # Robot arm movement
    - move_home      # Return robot to home position
```

#### Memory Tools
```yaml
tools:
  include:
    - get_action_plans     # Retrieve stored procedures
    - get_action_positions # Get robot position coordinates
    - get_item_locations   # Find object locations
```

#### Information Tools
```yaml
tools:
  include:
    - analyze_recipe       # Recipe analysis
    - identify_objects     # Object recognition
    - spatial_reasoning    # 3D spatial analysis
```

### Custom Tool Development

Create new tools by adding them to `tools.py`:

```python
def custom_kitchen_tool(input_data: str) -> str:
    """
    Custom tool for specific kitchen functionality.
    
    Args:
        input_data: JSON string with tool parameters
        
    Returns:
        Tool execution result
    """
    import json
    
    try:
        params = json.loads(input_data)
        action = params.get('action')
        target = params.get('target')
        
        # Implement custom functionality
        if action == 'analyze_ingredient':
            return analyze_ingredient_properties(target)
        elif action == 'suggest_substitution':
            return suggest_ingredient_substitution(target)
        else:
            return "Unknown action for kitchen tool"
            
    except Exception as e:
        return f"Error in custom kitchen tool: {str(e)}"

# Register the tool
TOOLS["custom_kitchen_tool"] = custom_kitchen_tool
```

Then include it in agent configurations:
```yaml
tools:
  include:
    - custom_kitchen_tool
```

## Memory System

### Memory Structure

The system uses three types of persistent memory:

#### 1. Action Plans (`memory/action_plan.yaml`)
```yaml
# Stored procedures for common tasks
opening_jar:
  description: "Step-by-step process for helping user open a jar"
  steps:
    - "Move robot arm to jar opening position" 
    - "Announce to user: 'Please place the jar in my gripper'"
    - "Wait for user to position jar"
    - "Close gripper to secure jar"
    - "Announce: 'I'm holding the jar steady. You can now twist the lid'"
    - "Monitor for completion"
    - "Release gripper when task complete"

making_coffee:
  description: "Coffee preparation assistance procedure"
  steps:
    - "Check if coffee maker is available"
    - "Guide user through coffee preparation steps"
    - "Monitor brewing process"
    - "Announce when coffee is ready"
```

#### 2. Action Positions (`memory/action_position.yaml`)
```yaml
# Stored robot arm positions
jar_opening_position:
  description: "Position for holding jars steady during opening"
  coordinates: [0.5, 0.3, 0.2, 0.0, 0.0, 0.0]
  gripper_state: "open"

cup_approach_position:
  description: "Approach position for grasping cups"
  coordinates: [0.4, 0.2, 0.15, 0.0, 0.0, 0.0]
  gripper_state: "open"

plate_drop_position:
  description: "Position for placing items on the plate"
  coordinates: [0.3, 0.4, 0.12, 0.0, 0.0, 0.0]
  gripper_state: "open"
```

#### 3. Item Locations (`memory/item_location.yaml`)
```yaml
# Known object locations in the kitchen
coffee:
  location: [0.2, 0.5, 0.1]
  description: "Coffee container location"
  last_updated: "2024-01-15"

cup:
  location: [0.4, 0.2, 0.0]
  description: "Default cup position"
  last_updated: "2024-01-15"

plate:
  location: [0.3, 0.4, 0.0]
  description: "Plate position on counter"
  last_updated: "2024-01-15"
```

### Accessing Memory in Agents

```python
# Example agent interaction with memory
response = agent.process_to_string("Do you have instructions for opening a jar?")
# Agent uses get_action_plans tool to retrieve opening_jar procedure

response = agent.process_to_string("Where should I place the cup?")
# Agent uses get_item_locations to find plate location for placement
```

## Multi-Agent Communication

### Hierarchical Agent Structure

Agents can call other agents as tools for specialized tasks:

```yaml
# configs/agents/comprehensive_kitchen_agent.yaml  
system_prompt: |
  You are a comprehensive kitchen assistant that coordinates with
  specialized sub-agents to handle complex tasks.

tools:
  include:
    - action_planning_agent    # Planning specialist
    - action_execution_agent   # Execution specialist
    - vision_analysis_agent    # Vision specialist
```

### Agent-as-Tool Implementation

```python
def action_planning_agent(request: str) -> str:
    """
    Specialized agent for creating action plans.
    
    Args:
        request: Task description requiring planning
        
    Returns:
        Structured action plan
    """
    planner = create_agent(agent_type="action_planner")
    plan = planner.process_to_string(request)
    return plan

# Register specialized agents as tools
TOOLS["action_planning_agent"] = action_planning_agent
```

### Communication Patterns

#### Sequential Task Execution
```python
# Example: Complex task with multiple agents
user_request = "Make me a cup of coffee"

# Step 1: Planning agent creates task sequence
plan = comprehensive_agent.call_tool("action_planning_agent", user_request)

# Step 2: Execution agent performs each step
for step in plan.steps:
    result = comprehensive_agent.call_tool("action_execution_agent", step)
    
# Step 3: Vision agent monitors completion
status = comprehensive_agent.call_tool("vision_analysis_agent", "Check coffee maker status")
```

## Advanced Configuration

### Environment-Based Configuration

Control agent behavior through environment variables:

```bash
# .env file
AGENT_MODE=development          # Enable debug features
SAFETY_LEVEL=high              # Increase safety checks  
HARDWARE_SIMULATION=true       # Use mock hardware
RESPONSE_VERBOSITY=detailed    # Increase response detail
```

```yaml
# Use environment variables in configurations
system_prompt: |
  You are a kitchen assistant robot.
  Safety level: ${SAFETY_LEVEL:medium}
  Debug mode: ${AGENT_MODE:production}
```

### Conditional Tool Loading

```yaml
tools:
  include:
    - calculator
    - text_processor
  
  # Conditional inclusions based on environment
  hardware_tools:
    condition: "${ENABLE_HARDWARE:false}"
    tools:
      - move_position
      - speak
      - capture
      
  development_tools:
    condition: "${AGENT_MODE:production} == development"
    tools:
      - debug_logger
      - system_monitor
```

### Agent Inheritance

Create agent hierarchies with inheritance:

```yaml
# Base kitchen agent
base_kitchen_agent:
  system_prompt: "Base kitchen assistant capabilities..."
  tools: [calculator, text_processor, speak]

# Specialized inherited agent  
advanced_kitchen_agent:
  inherits_from: base_kitchen_agent
  system_prompt: |
    ${parent.system_prompt}
    
    Additionally, you have advanced capabilities including:
    - Complex meal planning
    - Nutritional analysis
    - Dietary restriction handling
  
  tools:
    inherit: true  # Include parent tools
    include:
      - nutritional_analyzer
      - meal_planner
```

## Testing and Validation

### Agent Testing Framework

```python
# tests/test_agents.py
import pytest
from llm_ai_agent.agents import create_agent

class TestKitchenAssistant:
    def setup_method(self):
        self.agent = create_agent(
            agent_type="kitchen_assistant",
            use_hardware=False  # Use mock hardware for testing
        )
    
    def test_basic_interaction(self):
        response = self.agent.process_to_string("Hello, can you help me?")
        assert "help" in response.lower()
        
    def test_memory_access(self):
        response = self.agent.process_to_string("Where is the coffee?")
        assert "coffee" in response.lower()
        
    def test_tool_usage(self):
        response = self.agent.process_to_string("What is 15 + 27?")
        assert "42" in response
```

### Configuration Validation

```bash
# Validate agent configurations
python -m llm_ai_agent.validate_configs --config_dir configs/agents/

# Test specific agent configuration
python -m llm_ai_agent.test_agent --agent kitchen_assistant --test_suite basic
```

## Deployment Strategies

### Production Deployment

```python
# production_deployment.py
from llm_ai_agent.agents import create_agent
import logging

# Configure production logging
logging.basicConfig(level=logging.INFO)

# Create production agent with optimized settings
production_agent = create_agent(
    agent_type="kitchen_assistant",
    use_hardware=True,
    max_retries=3,
    timeout=30,
    safety_mode="strict"
)

# Production monitoring
def monitor_agent_health(agent):
    """Monitor agent performance and health metrics."""
    metrics = {
        'response_time': agent.get_avg_response_time(),
        'success_rate': agent.get_success_rate(),
        'error_count': agent.get_error_count()
    }
    return metrics
```

### Multi-Instance Deployment

```python
# Deploy multiple specialized agents
agent_pool = {
    'manipulation': create_agent("object_manipulation_agent"),
    'vision': create_agent("vision_agent"),
    'recipe': create_agent("recipe_assistant"),
    'general': create_agent("kitchen_assistant")
}

def route_request(user_request):
    """Route requests to appropriate specialized agent."""
    if "grasp" in user_request or "move" in user_request:
        return agent_pool['manipulation']
    elif "see" in user_request or "look" in user_request:
        return agent_pool['vision']
    elif "recipe" in user_request or "cook" in user_request:
        return agent_pool['recipe']
    else:
        return agent_pool['general']
```

This comprehensive guide provides the foundation for effectively configuring, deploying, and managing the AI agents in the Kitchen Assistive Robot system.