#!/usr/bin/env python3

import os
import sys
import time
from dotenv import load_dotenv
from llm_agent.ai_agent import BaseAgent
from llm_agent.hardware_tools import HardwareToolManager
from llm_agent.action_plan_executor import ActionPlanExecutor
# Import the mock tool manager
from llm_agent.mock_hardware_tools import MockHardwareToolManager

# Load environment variables
load_dotenv()

def main():
    """
    Main function to demonstrate the agent with hardware tools.
    """
    print("Initializing AI Agent with Hardware Tools...")
    
    try:
        # Agent is already initialized at the script level
        
        # Interactive loop
        print("\n" + "="*50)
        print("🤖 AI Hardware Agent")
        print("="*50)
        print("Type 'exit' or 'quit' to end the session")
        print("Type 'camera capture' to capture an image from the cameras")
        print("Type 'speak <text>' to make the agent speak")
        print("Type 'arm <command>' to control the robot arm using Cartesian control")
        print("  - arm move_home: Move arm to home position")
        print("  - arm move_default: Move arm to default position")
        print("  - arm move_position x y z rx ry rz: Move arm to Cartesian position")
        print("  - arm gripper open|close|position: Control the gripper")
        print("  - arm get_position: Get current position")
        print("="*50 + "\n")
        
        while True:
            # Get user input
            user_input = input("👤 You: ")
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit"]:
                print("🤖 Agent: Goodbye!")
                break
            
            # Handle special case for image capture and analysis
            if "camera" in user_input.lower() and "capture" in user_input.lower():
                # Let the agent process handle it through the updated natural workflow
                # which now directly analyzes the image with vision capabilities
                print("🤖 Agent: Processing camera capture and analysis request...")
                response = agent.process(
                    prompt=user_input,
                    max_tokens=1500,  # Increase max tokens for detailed image analysis
                    use_tools=True,
                    max_iterations=5
                )
                print(f"🤖 Agent: {response}")
                continue
                
            elif user_input.lower().startswith("speak "):
                text = user_input[6:].strip()
                if tool_manager.speaker_tool and text:
                    result = tool_manager.speaker_tool(f"speak {text}")
                    print(f"🤖 Speaker: {result}")
                else:
                    print("🤖 Agent: Speaker tool is not available or no text provided")
                continue
                
            elif user_input.lower().startswith("arm "):
                command = user_input[4:].strip()
                if tool_manager.robot_arm_tool and command:
                    result = tool_manager.robot_arm_tool(command)
                    print(f"🤖 Robot Arm: {result}")
                else:
                    print("🤖 Agent: Robot arm tool is not available or no command provided")
                continue
                
            # Standard text prompt with tools
            print("🤖 Agent: Thinking...")
            response = agent.process(
                prompt=user_input,
                max_tokens=1000,
                use_tools=True,
                max_iterations=5  # Allow multiple tool usages
            )
            print(f"🤖 Agent: {response}")
            
    except Exception as e:
        print(f"Error running hardware agent: {e}")

def handle_camera_capture(agent, tool_manager, user_input):
    """Handle camera capture and direct image analysis."""
    if not tool_manager.camera_tool:
        print("🤖 Agent: Camera tool is not available")
        return
        
    # Capture image (camera is already initialized)
    print("📸 Capturing image...")
    result = tool_manager.camera_tool("capture")
    print(f"🤖 Camera: {result}")
    
    # Check if the image capture was successful
    if "Successfully captured" in result:
        # With the updated BaseAgent.process method, we no longer need to manually analyze
        # the image here as the agent will automatically continue the conversation and
        # analyze the image in the next iteration.
        print("🔍 Image captured successfully. The AI will automatically analyze it in the conversation.")
        return result
    else:
        print("❌ Failed to capture image")
        return result

def demo_sequence():
    """Run a demo sequence to showcase the agent's capabilities."""
    # Note: We're already initializing the tool_manager at the script level
    
    try:
        # Tools have already been initialized
        
        print("\n=== Starting Demo Sequence ===\n")
        
        # Step 1: Speak introduction
        if tool_manager.speaker_tool:
            print("Step 1: Introduction")
            tool_manager.speaker_tool("speak Hello, I am an AI agent with hardware control capabilities. I will demonstrate what I can do.")
            time.sleep(1)
        
        # Step 2: Capture image from camera
        if tool_manager.camera_tool:
            print("\nStep 2: Capturing image from camera")
            result = tool_manager.camera_tool("capture")
            print(f"Camera result: {result}")
            time.sleep(1)
        
        # Step 3: Move robot arm to home position
        if tool_manager.robot_arm_tool:
            print("\nStep 3: Moving robot arm to home position")
            result = tool_manager.robot_arm_tool("move_home")
            print(f"Robot arm result: {result}")
            time.sleep(3)  # Wait for movement to complete
        
        # Step 4: Speak the results
        if tool_manager.speaker_tool:
            print("\nStep 4: Reporting results")
            tool_manager.speaker_tool("speak Demo sequence completed successfully. I can see, speak, and control the robot arm.")
        
        print("\n=== Demo Sequence Completed ===\n")
        
    except Exception as e:
        print(f"Error in demo sequence: {e}")

def action_plan_mode(agent, tool_manager):
    """
    Run the agent in action plan execution mode.
    This mode executes predefined action plans from YAML files.
    """
    print("\n" + "="*50)
    print("🤖 AI Action Plan Executor")
    print("="*50)
    
    # Create the action plan executor
    executor = ActionPlanExecutor(agent, tool_manager)
    
    # List available plans
    plans = executor.list_available_plans()
    if not plans:
        print("No action plans found. Please check the action_plan.yaml file.")
        return
        
    print("Available action plans:")
    for i, plan in enumerate(plans, 1):
        print(f"{i}. {plan}")
    
    while True:
        try:
            choice = input("\nSelect a plan to execute (number) or 'q' to quit: ")
            
            if choice.lower() == 'q':
                print("Exiting action plan mode.")
                break
                
            plan_index = int(choice) - 1
            if plan_index < 0 or plan_index >= len(plans):
                print(f"Invalid selection. Please enter a number between 1 and {len(plans)}.")
                continue
                
            selected_plan = plans[plan_index]
            print(f"\nExecuting plan: {selected_plan}")
            
            # Execute the selected plan
            execution_history = executor.execute_plan(selected_plan)
            
            # Show summary
            success_count = sum(1 for step in execution_history if step['success'])
            total_steps = len(execution_history)
            
            print("\nExecution Summary:")
            print(f"Plan: {selected_plan}")
            print(f"Steps: {success_count}/{total_steps} completed successfully")
            print(f"Status: {'Completed' if success_count == total_steps else 'Incomplete'}")
            
            # Ask if user wants to execute another plan
            continue_response = input("\nExecute another plan? (y/n): ").lower()
            if continue_response != 'y':
                print("Exiting action plan mode.")
                break
                
        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            print(f"Error during plan execution: {e}")
    
    print("="*50)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Agent with Hardware Control")
    parser.add_argument('--demo', action='store_true', help='Run demo sequence instead of interactive mode')
    parser.add_argument('--action-plan', action='store_true', help='Run in action plan execution mode')
    parser.add_argument('--mock', action='store_true', help='Use mock hardware tools instead of real hardware')
    
    args = parser.parse_args()
    
    # Initialize components based on mode
    print("\n=== Initializing AI Agent with Hardware ===")
    if args.mock:
        print("🔧 MOCK MODE: Using simulated hardware tools")
        tool_manager = MockHardwareToolManager()
        # Create mock directories if they don't exist
        os.makedirs("debug_mock", exist_ok=True)
        os.makedirs("debug_images", exist_ok=True)
        
        # Set environment variable to control which mock camera scenario to use
        # This can be changed programmatically during testing to simulate different scenes
        os.environ["MOCK_CAMERA_SCENARIO"] = "empty"
    else:
        print("🔧 HARDWARE MODE: Using real hardware tools")
        tool_manager = HardwareToolManager()
    
    print("\n=== Initializing AI Agent ===")
    agent = BaseAgent(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model_name=os.getenv("MODEL_NAME", "anthropic/claude-3-opus-20240229"),
        system_prompt=(
            "You are a helpful AI assistant that can control physical hardware devices. "
            "You have access to cameras, a robot arm, and a speaker. "
            "Use these tools appropriately to help the user accomplish their tasks. "
            "When using hardware tools, be precise and careful. "
            "For the robot arm, always use Cartesian control methods (not joint control). "
            "Always verify positions before making large movements. "
            "For the speaker, keep responses concise and clear. "
            "For the cameras, use them to gather visual information when needed.\n\n"
            "IMPORTANT: When you want to use a tool, you MUST use the exact format:\n"
            "[TOOL] <tool_name> <arguments> [/TOOL]\n\n"
            "For example:\n"
            "- To speak: [TOOL] speaker speak Hello, I am your AI assistant [/TOOL]\n"
            "- To capture an image: [TOOL] camera capture [/TOOL]\n"
            "- To move the arm: [TOOL] robot_arm move_position 0.2 0.3 0.4 1.5 1.5 1.5 [/TOOL]\n\n"
            "DO NOT include the tool commands in your regular text. "
            "ALWAYS use the [TOOL] format when you want to execute a tool command."
        )
    )
    
    # Add hardware tools to the agent
    print("\n=== Registering Hardware Tools with Agent ===")
    tools = tool_manager.get_all_tools()
    for tool in tools:
        agent.add_tool(tool)
        print(f"Added {tool.name} tool to agent")
    
    print("\n=== System Ready ===")
    
    try:
        if args.action_plan:
            # If in mock mode and using action plans, set the camera scenario to jar_on_table
            # to simulate the presence of a jar for testing
            if args.mock:
                os.environ["MOCK_CAMERA_SCENARIO"] = "jar_on_table"
                print("🔧 MOCK ACTION PLAN MODE: Camera set to 'jar_on_table' scenario")

            action_plan_mode(agent, tool_manager)
            
        elif args.demo:
            demo_sequence()
            
        else:
            main()
            
    finally:
        # Clean up
        print("\nShutting down hardware tools...")
        tool_manager.close_all() 