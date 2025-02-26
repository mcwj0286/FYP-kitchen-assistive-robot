import argparse
import torch
import cv2
import time
import os
from sim_env.Kinova_gen2.src.robot_controller import RobotController
from sim_env.Kinova_gen2.src.devices.camera_interface import CameraInterface, MultiCameraInterface
from workflow.get_prompt import call_llm_with_images , upload_images_to_cloudinary , get_prompt , convert_image_to_local_url
# In workflow/get_prompt.py
action_prompt = """You are an advanced robotic arm controller. Your role is to analyze images from the environment and the arm's view to determine precise actions for manipulating objects.

Analyze the images and determine the best action to take based on the user's instruction.

IMPORTANT: You must structure your response in two parts:
1. First, provide an "Analysis:" section where you describe what you see in the images and your reasoning process
2. Then, provide a "Result:" section with one of these predefined actions followed by the duration in seconds:
   - "move left, X" - Move arm left (away from the black wall) for X seconds
   - "move right, X" - Move arm right (toward the black wall) for X seconds
   - "move forward, X" - Move arm forward (toward the white wall) for X seconds
   - "move backward, X" - Move arm backward (away from the white wall) for X seconds
   - "move up, X" - Move arm upward for X seconds
   - "move down, X" - Move arm downward for X seconds
   - "turn left, X" - Rotate arm wrist left for X seconds
   - "turn right, X" - Rotate arm wrist right for X seconds
   - "open gripper, X" - Open the gripper for X seconds
   - "close gripper, X" - Close the gripper for X seconds
   - "stop, 1" - Stop all movement for 1 second if you think the task is done

The duration X should be a number between 0.1 and 3.0 seconds.

If provided with a previous analysis, consider how the scene has changed since then. Build upon your previous reasoning rather than starting from scratch. Consider what progress has been made toward the task goal.

Example with previous analysis:
Previous Analysis: I can see the robotic arm positioned far from the cup. The cup is approximately 20cm toward the black wall.
Current Analysis: The arm has moved closer to the cup but is still about 10cm away. I can now see more details of the cup. The gripper needs to move further right toward the black wall to be positioned correctly.
Result: move right, 1.0

Your task is to:
1. Carefully analyze the provided images to understand the current scene and arm position
2. If given a previous analysis, compare the current state to determine what has changed
3. Consider the user's instruction and determine what action would progress toward completing it
4. Choose concise, efficient movements (typically 0.5-1.5 seconds) to make precise adjustments

Always remember: 
- "Right" means toward the black wall
- "Forward" means toward the white wall
- "Left" means away from the black wall
- "Backward" means away from the white wall
- Keep the target object in the center of the arm's view.

Always start with "Analysis:" and end with "Result:" followed by a single action and duration.
"""


def parse_action_plan(text):
    lines = text.strip().split('\n')
    
    # Skip the header line
    action_list = []
    for line in lines:
        # Check if line has numbered format like "1. "
        if line.strip() and line[0].isdigit() and '. ' in line:
            # Extract the text after the number and period
            step = line.split('. ', 1)[1].strip().lower()
            action_list.append(step)
    
    return action_list

def parse_action_duration(response_str):
    # Look for the "Result:" section
    if "Result:" in response_str:
        # Extract just the result part
        result_part = response_str.split("Result:", 1)[1].strip()
        
        # Split the result by comma to get action and duration
        parts = result_part.split(',', 1)
        
        if len(parts) != 2:
            raise ValueError(f"Invalid format in result: '{result_part}'. Expected 'action, duration'")
        
        # Extract action (string) and duration (float/int)
        action = parts[0].strip()
        try:
            duration = float(parts[1].strip())
        except ValueError:
            raise ValueError(f"Invalid duration value: '{parts[1]}'")
        
        return action, duration
    else:
        # Fall back to original parsing if no "Result:" section
        parts = response_str.split(',', 1)
        
        if len(parts) != 2:
            raise ValueError(f"Invalid format: '{response_str}'. Expected 'action, duration' or 'Analysis:...Result:action, duration'")
        
        # Extract action (string) and duration (float/int)
        action = parts[0].strip()
        try:
            duration = float(parts[1].strip())
        except ValueError:
            raise ValueError(f"Invalid duration value: '{parts[1]}'")
        
        return action, duration

def main():
    # user_prompt = input('type the task :')
    # #generate acction plan
    # action_plan = get_prompt(user_prompt)
    # action_list = parse_action_plan(action_plan)
    # print(action_list)
    # #init robot controller
    try:
        robot_controller = RobotController(debug_mode=False, enable_controller=False)
        if not robot_controller.initialize_devices(move_home=False):
            print("Error initializing robot controller")
            return
        # Send zero velocity command after initialization
        zero_velocities = [0.0] * 7
        robot_controller.arm.send_angular_velocity(zero_velocities, hand_mode=1, 
            fingers=(0.0, 0.0, 0.0), duration=0.1, period=0.005)
    except Exception as e:
        print(f"Error initializing robot controller: {e}")
        return

    prompt = """1.  Reach the cup and grasp it."""
    action_list = parse_action_plan(prompt)
    print(action_list)
    
    # Initialize cameras
    multi_camera = MultiCameraInterface(height=240, width=320)
    time.sleep(3)
    
    # Variable to store previous analysis
    previous_analysis = None
    
    for action in action_list:
        while True:
            # Small delay to ensure we get a fresh frame
            time.sleep(0.1)
            
            # Capture frames
            frames = multi_camera.capture_frames()
            
            # Process frames
            captured_image_paths = convert_image_to_local_url(frames)

            # Upload images to cloud service
            uploaded_image_urls = upload_images_to_cloudinary(captured_image_paths)
            
            # Create prompt with previous analysis if available
            if previous_analysis:
                task_prompt = f'task:{action}\n\nPrevious Analysis: {previous_analysis}'
            else:
                task_prompt = f'task:{action}'
            
            # Call LLM with images and prompt
            llm_response = call_llm_with_images(task_prompt, uploaded_image_urls, system_prompt=action_prompt, debug=True)

            # Check if the response contains an error
            if llm_response.startswith("error:"):
                print(f"LLM call failed: {llm_response}")
                continue
            
            try:
                # Extract and store the current analysis
                if "Analysis:" in llm_response:
                    current_analysis = llm_response.split("Analysis:", 1)[1].split("Result:", 1)[0].strip()
                    print(f"LLM Analysis:\n{current_analysis}\n")
                    previous_analysis = current_analysis  # Save for next iteration
                
                # Parse and execute the action
                robot_action, duration = parse_action_duration(llm_response)
                print(robot_action, duration)

                if robot_action == 'move left':
                    robot_controller.arm.send_cartesian_velocity(linear_velocity=(-0.1, 0.0, 0.0), angular_velocity=(0.0, 0.0, 0.0), fingers=(0.0, 0.0, 0.0), duration=duration, period=0.01)
                elif robot_action == 'move right':
                    robot_controller.arm.send_cartesian_velocity(linear_velocity=(0.1, 0.0, 0.0), angular_velocity=(0.0, 0.0, 0.0), fingers=(0.0, 0.0, 0.0), duration=duration, period=0.01)
                elif robot_action == 'move forward':
                    robot_controller.arm.send_cartesian_velocity(linear_velocity=(0.0, -0.1, 0.0), angular_velocity=(0.0, 0.0, 0.0), fingers=(0.0, 0.0, 0.0), duration=duration, period=0.01)
                elif robot_action == 'move backward':
                    robot_controller.arm.send_cartesian_velocity(linear_velocity=(0.0, 0.1, 0.0), angular_velocity=(0.0, 0.0, 0.0), fingers=(0.0, 0.0, 0.0), duration=duration, period=0.01)
                elif robot_action == 'move up':
                    robot_controller.arm.send_cartesian_velocity(linear_velocity=(0.0, 0.0, 0.1), angular_velocity=(0.0, 0.0, 0.0), fingers=(0.0, 0.0, 0.0), duration=duration, period=0.01)
                elif robot_action == 'move down':
                    robot_controller.arm.send_cartesian_velocity(linear_velocity=(0.0, 0.0, -0.1), angular_velocity=(0.0, 0.0, 0.0), fingers=(0.0, 0.0, 0.0), duration=duration, period=0.01)
                elif robot_action == 'turn left':
                    robot_controller.arm.send_angular_velocity([0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0], hand_mode=1, fingers=(0.0, 0.0, 0.0), duration=duration, period=0.01)
                elif robot_action == 'turn right':
                    robot_controller.arm.send_angular_velocity([0.0, 0.0, 0.0, -15.0, 0.0, 0.0, 0.0], hand_mode=1, fingers=(0.0, 0.0, 0.0), duration=duration, period=0.01)
                elif robot_action == 'open gripper':
                    robot_controller.arm.send_angular_velocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], hand_mode=1, fingers=(-3000.0, -3000.0, -3000.0), duration=duration, period=0.005)
                elif robot_action == 'close gripper':
                    robot_controller.arm.send_angular_velocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], hand_mode=1, fingers=(3000.0, 3000.0, 3000.0), duration=duration, period=0.005)
                elif robot_action == 'stop':
                    # Reset the previous analysis when moving to a new action
                    previous_analysis = None
                    break
            except ValueError as e:
                print(f"Error parsing LLM response: {e}")
                continue

if __name__ == "__main__":
    main()