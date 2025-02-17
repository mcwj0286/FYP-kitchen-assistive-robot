# FYP-kitchen-assistive-robotic

gen2jaco control : http://wiki.ros.org/jaco_ros#Installation

#workflow

1. real time action planning
user_prompt + camera images -> get_action_plan -> action_plan

2. action execution
action_plan -> execute_action_plan -> action_execution

3. review
action_execution -> get_feedback -> feedback


Inference loop:

for task in action_plan:
    execute_task(task_description)
    llm_review = get_feedback(images)
    if llm_review == "success":
        break
    else:
        action_plan = get_action_plan(llm_review)

# Model Training Procedure

-SFT
Training from scratch using human demonstrations data.

-RLLMF (Reinforcement Learning from Language Model Feedback)
Training from LLM generated reward. (PPO)   






