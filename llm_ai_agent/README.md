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

## Progress

- [x] Project setup
- [x] Basic LLM configuration
- [x] BaseAgent implementation
- [x] KitchenAssistantAgent implementation
- [x] Interactive testing interface
- [x] Structured output approach implementation
- [ ] Hardware tool integration
- [ ] Action planning integration
- [ ] Advanced features
- [ ] Complete test suite

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

### Interactive Mode
```bash
# Run the interactive testing script
python interactive.py
```
