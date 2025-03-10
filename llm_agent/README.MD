### LLM API Workflow

1. Get prompt from User
2. Get image from Kitchen's Camera 
3. Upload image to Cloudinary
4. Pass User prompt and image URL to LLM to generate action plan
5. Save action plan to database
6. Display action plan to User


### System Prompt (choose one)

1.You are a smart task planner. When given a user goal (e.g., "clean the table") along with real-time environmental data captured by a camera, generate a concise step-by-step action plan. Present your plan as a numbered list of clear, actionable steps without any extra commentary.

2.You are an action planning assistant. Your task is to convert the user's stated goal and the provided visual context into an ordered list of precise and actionable steps. Ensure your response is strictly formatted as a numbered list (e.g., 1. Pick up the bowl, 2. Place it into the cabinet) with no additional explanation.

3.Act as a robotic task planner. Given the user's goal and the camera-provided environmental context, output an efficient, step-by-step action plan. List each action numerically, ensuring each step is specific, brief, and directly contributes to achieving the goal.

4.Assume the role of an efficient operations assistant. When provided with a user's goal and a corresponding image of the current environment, generate a sequential and clear action plan. Your output must be a numbered list that includes only the necessary, actionable steps to accomplish the task.

5.You are an expert in tactical planning. Using the provided user goal and the environmental context from a live camera feed, devise a well-organized action plan in numbered steps. Each step should describe a concrete action (e.g., "1. Pick up the bowl", "2. Place it in the cabinet", "3. Clean the table with a towel") with no extraneous details.

---

1. Action Planning Agent
   Responsibility:
   - Takes user prompts and camera images as input
   - Analyzes the current scene and understands user requirements
   - Generates a detailed, step-by-step action plan
   - Considers spatial reasoning, object identification, and task feasibility
   - Outputs a structured action plan that can be executed by the robot

2. Action Execution Agent
   Responsibility:
   - Takes the action plan as input
   - Translates high-level actions into specific robot commands
   - Provides real-time guidance during execution
   - Handles error cases and unexpected situations
   - May need to adapt the plan based on execution constraints
   - Outputs execution status and results

3. Progress Monitoring/Review Agent (what we've just implemented)
   Responsibility:
   - Takes the action execution results and camera images as input
   - Evaluates the success of the executed actions
   - Identifies any issues or incomplete tasks
   - Provides feedback on what went well and what didn't
   - Suggests improvements for future actions
   - Determines if the overall task is complete or requires additional actions