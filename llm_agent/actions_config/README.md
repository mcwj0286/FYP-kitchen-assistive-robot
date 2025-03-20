# Action Plan Execution System

This directory contains configuration files for the action plan execution system. The system allows you to define structured action plans that the robot can execute step by step, using computer vision and LLM analysis to adapt to the environment.

## Overview

The action plan system consists of:

1. **Action Plans**: Defined in `action_plan.yaml`
2. **Predefined Locations**: Defined in `predefined_loaction.yaml`
3. **ActionPlanExecutor**: The execution engine in `llm_agent/action_plan_executor.py`

## Action Plan Format

Action plans are defined in YAML format. Each plan has:
- A unique name
- A goal description
- A sequence of steps with step numbers and descriptions

Example:
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

## Predefined Locations

Locations for robot arm movements are defined in `predefined_loaction.yaml`. These locations are referenced by name in the action plans.

Example:
```yaml
open_jar_position: ['0.2795', '-0.1953', '0.3392', '-3.0256', '0.3381', '-1.3524', '2544.0000']
```

The format is:
- Location name: A list of coordinates including:
  - First 6 values: X, Y, Z, roll, pitch, yaw (Cartesian coordinates)
  - Last value: Gripper position

## How Steps Are Executed

Each step in an action plan is executed differently based on its description:

1. **Move to position**: For steps containing "move to", the system looks up the location in the predefined locations file and moves the robot arm to that position.

2. **Announcements**: For steps containing "announce", the system uses the speaker to make an announcement to the user.

3. **Wait for user action**: For steps containing "wait for user", the system captures images periodically and uses the LLM to analyze if the expected condition is met (e.g., jar placed on gripper).

4. **Gripper actions**: For steps containing "close the gripper" or similar, the system executes the appropriate gripper command.

5. **Complex motions**: For steps like "roll left to open jar", the system performs a series of joint movements to achieve the desired motion.

## Using the System

To use the action plan system:

1. Run the hardware agent with the `--action-plan` flag:
   ```
   python -m llm_agent.hardware_agent_example --action-plan
   ```

2. The system will list available action plans and prompt you to select one.

3. Once a plan is selected, it will execute each step in sequence, showing progress and results.

4. If a step fails, you'll be given options to retry, skip, or abort the plan.

## Creating New Plans

To create a new action plan:

1. Add a new entry to `action_plan.yaml` following the format shown above.

2. If needed, add new locations to `predefined_loaction.yaml`.

3. Make sure the step descriptions use keywords that the action plan executor can recognize.

## Best Practices

1. **Step Descriptions**: Use clear, consistent language in step descriptions. The executor matches patterns like "move to", "announce", "wait for", etc.

2. **Error Handling**: Always include appropriate error handling in your plans.

3. **Testing**: Test individual steps before executing the full plan.

4. **Locations**: Verify all locations are safe and appropriate for the task.

5. **User Interaction**: Use clear and concise announcements when interacting with users. 