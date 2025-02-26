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

IMPORTANT: You must respond ONLY with one of these predefined actions followed by the duration in seconds:
- "move left, X" - Move arm left for X seconds
- "move right, X" - Move arm right for X seconds
- "move forward, X" - Move arm forward for X seconds
- "move backward, X" - Move arm backward for X seconds
- "move up, X" - Move arm upward for X seconds
- "move down, X" - Move arm downward for X seconds
- "turn left, X" - Rotate gripper left for X seconds
- "turn right, X" - Rotate gripper right for X seconds
- "open gripper, X" - Open the gripper for X seconds
- "close gripper, X" - Close the gripper for X seconds
- "stop, 1" - Stop all movement for 1 second if you think the task is done

The duration X should be a number between 0.1 and 3.0 seconds.

For example:
- "move left, 0.5" (moves left for half a second)
- "close gripper, 1.0" (closes gripper for 1 second)

Your task is to:
1. Carefully analyze the provided images to understand the current scene and arm position
2. Consider the user's instruction and determine what action would progress toward completing it
3. Respond with ONLY the appropriate action and duration in the exact format described above
4. Choose concise, efficient movements (typically 0.5-1.5 seconds) to make precise adjustments

DO NOT provide explanations, commentary, or multiple options. Output ONLY a single action and duration.
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
    # Split the string by comma
    parts = response_str.split(',', 1)
    
    if len(parts) != 2:
        raise ValueError(f"Invalid format: '{response_str}'. Expected 'action, duration'")
    
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

    prompt = """1.  Move to the cup.
2.  Grasp the cup."""
    action_list = parse_action_plan(prompt)
    print(action_list)
    # Initialize the robot controller
    multi_camera = MultiCameraInterface(height=240, width=320)
    time.sleep(3)
    for action in action_list:

        while True:
            # Small delay to ensure we get a fresh frame
            time.sleep(0.1)
            
            # Capture frames
            frames = multi_camera.capture_frames()
            
            # Verify we have valid frames with content
            valid_frames = False
            for cam_id, (success, frame) in frames.items():
                if success and frame is not None and not (frame.size == 0):
                    valid_frames = True
                    # Debug: display the frame
                    cv2.imshow(f"Camera {cam_id}", frame)
                    cv2.waitKey(1)
            
            if not valid_frames:
                print("No valid frames captured, retrying...")
                continue
            
            # Process frames as before
            captured_image_paths = convert_image_to_local_url(frames)

            # Use modular functions to upload images and perform the API call.
            uploaded_image_urls = upload_images_to_cloudinary(captured_image_paths)
            prompt = f'task:{action}'
            
            llm_response = call_llm_with_images(prompt, uploaded_image_urls, system_prompt=action_prompt, debug=True)

            # Check if the response contains an error
            if llm_response.startswith("error:"):
                print(f"LLM call failed: {llm_response}")
                # Maybe retry or take alternative action
                continue
            
            try:
                robot_action, duration = parse_action_duration(llm_response)
                print(robot_action, duration)

                if robot_action == 'move left':
                    robot_controller.arm.send_cartesian_velocity(linear_velocity=(-0.1, 0.0, 0.0), angular_velocity=(0.0, 0.0, 0.0), fingers=(0.0, 0.0, 0.0), duration=duration, period=0.005)
                elif robot_action == 'move right':
                    robot_controller.arm.send_cartesian_velocity(linear_velocity=(0.1, 0.0, 0.0), angular_velocity=(0.0, 0.0, 0.0), fingers=(0.0, 0.0, 0.0), duration=duration, period=0.005)
                elif robot_action == 'move forward':
                    robot_controller.arm.send_cartesian_velocity(linear_velocity=(0.0, -0.1, 0.0), angular_velocity=(0.0, 0.0, 0.0), fingers=(0.0, 0.0, 0.0), duration=duration, period=0.005)
                elif robot_action == 'move backward':
                    robot_controller.arm.send_cartesian_velocity(linear_velocity=(0.0, 0.1, 0.0), angular_velocity=(0.0, 0.0, 0.0), fingers=(0.0, 0.0, 0.0), duration=duration, period=0.005)
                elif robot_action == 'move up':
                    robot_controller.arm.send_cartesian_velocity(linear_velocity=(0.0, 0.0, 0.1), angular_velocity=(0.0, 0.0, 0.0), fingers=(0.0, 0.0, 0.0), duration=duration, period=0.005)
                elif robot_action == 'move down':
                    robot_controller.arm.send_cartesian_velocity(linear_velocity=(0.0, 0.0, -0.1), angular_velocity=(0.0, 0.0, 0.0), fingers=(0.0, 0.0, 0.0), duration=duration, period=0.005)
                elif robot_action == 'turn left':
                    robot_controller.arm.send_angular_velocity([0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0], hand_mode=1, fingers=(0.0, 0.0, 0.0), duration=duration, period=0.005)
                elif robot_action == 'turn right':
                    robot_controller.arm.send_angular_velocity([0.0, 0.0, 0.0, -15.0, 0.0, 0.0, 0.0], hand_mode=1, fingers=(0.0, 0.0, 0.0), duration=duration, period=0.005)
                elif robot_action == 'open gripper':
                    robot_controller.arm.send_angular_velocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], hand_mode=1, fingers=(3000.0, 3000.0, 3000.0), duration=duration, period=0.005)
                elif robot_action == 'close gripper':
                    robot_controller.arm.send_angular_velocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], hand_mode=1, fingers=(-3000.0, -3000.0, -3000.0), duration=duration, period=0.005)
                elif robot_action == 'stop':
                    #do nothing
                    break
            except ValueError as e:
                print(f"Error parsing LLM response: {e}")
                continue

if __name__ == "__main__":
    main()