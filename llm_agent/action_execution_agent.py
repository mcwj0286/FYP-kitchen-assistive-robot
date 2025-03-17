import os
import sys
import time
import cv2

from dotenv import load_dotenv
load_dotenv()  # Load .env variables

# Use relative imports
from .get_prompt import (
    upload_images_to_cloudinary,
    upload_image_to_server,
    call_llm_with_images,
    save_images,
)
from sim_env.Kinova_gen2.src.robot_controller import RobotController

class ActionExecutionAgent:
    """
    An agent that takes a high-level task prompt and breaks it down into a series of
    predefined robot actions (move left, move right, etc.) to accomplish the task.
    """
    
    def __init__(self, server_url=None, camera_interface=None):
        """
        Initialize the ActionExecutionAgent.
        
        Args:
            server_url (str, optional): URL of the server to upload images to.
                If None, images will be uploaded to Cloudinary.
            camera_interface: Optional camera interface to use for capturing frames.
                If None, will try to import and use MultiCameraInterface.
        """
        self.server_url = server_url or os.getenv("IMAGE_SERVER_URL")
        self.model_name = os.getenv("MODEL_NAME", "claude-3-opus-20240229")
        self.system_prompt = os.getenv("SYSTEM_PROMPT")
        self.robot_controller = RobotController(enable_controller=False)
        self.robot_controller.initialize_devices()
        self.camera_interface = camera_interface
        
        # Initialize the camera interface if not provided
        if self.camera_interface is None:
            try:
                # Try to import the MultiCameraInterface
                from sim_env.Kinova_gen2.src.devices.camera_interface import MultiCameraInterface
                self.camera_interface = MultiCameraInterface(height=240, width=320)
                time.sleep(3)  # Wait for cameras to initialize
                print("Camera interface initialized successfully.")
            except ImportError:
                print("Warning: MultiCameraInterface not found. No images will be captured.")
                self.camera_interface = None
        
        # Initialize the action prompt with cartesian position information
        self.action_prompt = """You are an advanced robotic arm controller. Your role is to analyze images from the environment and the arm's view to determine precise actions for manipulating objects.

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
   - "turn left, X" - Rotate arm around X axis (yaw) left for X seconds
   - "turn right, X" - Rotate arm around X axis (yaw) right for X seconds
   - "roll left, X" - Rotate arm around Z axis (roll) left for X seconds
   - "roll right, X" - Rotate arm around Z axis (roll) right for X seconds
   - "pitch up, X" - Rotate arm around Y axis (pitch) up for X seconds
   - "pitch down, X" - Rotate arm around Y axis (pitch) down for X seconds
   - "open gripper, X" - Open the gripper for X seconds
   - "close gripper, X" - Close the gripper for X seconds
   - "stop, 1" - Stop all movement for 1 second if you think the task is done

The duration X should be a number between 0.1 and 3.0 seconds.

If provided with a previous analysis, consider how the scene has changed since then. Build upon your previous reasoning rather than starting from scratch. Consider what progress has been made toward the task goal.

You will also be provided with the current Cartesian position of the robot arm in the following format:
Position (X, Y, Z): [x, y, z] meters
Orientation (X, Y, Z): [x, y, z] radian/s

Use this position information to better understand the arm's current state and make more precise movement decisions.

Example with previous analysis:
Previous Analysis: I can see the robotic arm positioned far from the cup. The cup is approximately 20cm toward the black wall.
Current Analysis: The arm has moved closer to the cup but is still about 10cm away. I can now see more details of the cup. The gripper needs to move further right toward the black wall to be positioned correctly.
Result: move right, 1.0

Your task is to:
1. Carefully analyze the provided images to understand the current scene and arm position
2. If given a previous analysis, compare the current state to determine what has changed
3. Consider the user's instruction and determine what action would progress toward completing it
4. Choose concise, efficient movements (typically 0.5-1.5 seconds) to make precise adjustments

Understanding rotation axes:
- X axis runs left to right (roll rotates around the arm's forward axis)
- Y axis runs front to back (pitch rotates like nodding up and down)
- Z axis runs up and down (yaw rotates like turning left and right)

Always remember: 
- "Right" means toward the black wall
- "Forward" means toward the white wall
- "Left" means away from the black wall
- "Backward" means away from the white wall
- Keep the target object in the center of the arm's view.

Always start with "Analysis:" and end with "Result:" followed by a single action and duration.
"""
        
        # Track execution state
        self.task = None
        self.previous_analysis = None
        self.execution_log = []
        
        # Define cartesian velocity parameters (same as in record_demo.py)
        self.cartesian_linear_scale = 0.1  # Maximum Cartesian linear velocity in m/s
        self.cartesian_angular_scale = 40.0  # Maximum Cartesian angular velocity in degrees/s
    
    def capture_and_process_images(self):
        """
        Capture images from cameras and process them.
                
        Returns:
            dict: Dictionary of camera IDs to image paths.
        """
        if self.camera_interface is None:
            print("No camera interface available. Cannot capture images.")
            return {}
        
        try:
            # Capture frames from cameras
            frames = self.camera_interface.capture_frames()
            
            # Convert frames to local URLs
            image_paths = save_images(frames)
            
            return image_paths
        except Exception as e:
            print(f"Error capturing images: {e}")
            return {}
    
    def get_cartesian_position(self, robot_controller):
        """
        Get the current Cartesian position of the robot arm.
        
        Args:
            robot_controller: The robot controller instance.
            
        Returns:
            list: The Cartesian position as [x, y, z, qx, qy, qz]
            or None if the position couldn't be retrieved.
        """
        try:
            if robot_controller and hasattr(robot_controller, 'arm'):
                return robot_controller.arm.get_cartesian_position()
            return None
        except Exception as e:
            print(f"Error getting Cartesian position: {e}")
            return None
    
    def format_cartesian_position(self, cartesian_position):
        """
        Format the Cartesian position for inclusion in the prompt.
        
        Args:
            cartesian_position: The Cartesian position as returned by get_cartesian_position.
            
        Returns:
            str: Formatted string describing the position.
        """
        if cartesian_position is None:
            return ""
            
        try:
            # Extract position and orientation from the full pose
            # Typically cartesian_position is a list/tuple with 7 values:
            # [x, y, z, qx, qy, qz, qw]
            if len(cartesian_position) >= 7:
                position = cartesian_position[:3]
                orientation = cartesian_position[3:6]
                
                position_str = f"Position (X, Y, Z): [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}] meters\n"
                orientation_str = f"Orientation (X, Y, Z): [{orientation[0]:.3f}, {orientation[1]:.3f}, {orientation[2]:.3f}] radian/s"
                
                return f"Current arm cartesian position:\n{position_str}{orientation_str}"
            else:
                return f"Incomplete cartesian position data: {cartesian_position}"
        except Exception as e:
            print(f"Error formatting Cartesian position: {e}")
            return ""
    
    def upload_images(self, image_paths):
        """
        Upload images to a server or Cloudinary.
        
        Args:
            image_paths (dict): Dictionary of camera IDs to image file paths.
            
        Returns:
            dict: Dictionary of camera IDs to image URLs.
        """
        if self.server_url:
            # Use server upload
            uploaded_urls = {}
            for cam_id, file_path in image_paths.items():
                url = upload_image_to_server(self.server_url, file_path)
                if url:
                    uploaded_urls[cam_id] = url
            return uploaded_urls
        else:
            # Use Cloudinary upload
            return upload_images_to_cloudinary(image_paths)
    
    def parse_action_duration(self, response_str):
        """
        Parse the action and duration from the LLM response.
        
        Args:
            response_str (str): The LLM response string.
            
        Returns:
            tuple: (action, duration) where action is a string and duration is a float.
        """
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
    
    def extract_analysis(self, response_str):
        """
        Extract the analysis section from the LLM response.
        
        Args:
            response_str (str): The LLM response string.
            
        Returns:
            str: The analysis part of the response, or None if not found.
        """
        if "Analysis:" in response_str and "Result:" in response_str:
            analysis = response_str.split("Analysis:", 1)[1].split("Result:", 1)[0].strip()
            return analysis
        return None
    
    def set_task(self, task):
        """
        Set the high-level task to be executed.
        
        Args:
            task (str): The task description, e.g., "locate the cup on the table".
        """
        self.task = task
        self.previous_analysis = None
        self.execution_log = []
    
    def execute_action(self, action, duration, robot_controller):
        """
        Execute a specific robot action for the given duration.
        
        Args:
            action (str): The action to execute (e.g., "move left").
            duration (float): The duration in seconds.
            robot_controller: The robot controller to use for execution.
            
        Returns:
            bool: True if execution was successful, False otherwise.
        """
        try:
            # Linear movements
            if action == 'move left':
                robot_controller.arm.send_cartesian_velocity(
                    linear_velocity=(self.cartesian_linear_scale, 0.0, 0.0), 
                    angular_velocity=(0.0, 0.0, 0.0), 
                    fingers=(0.0, 0.0, 0.0), 
                    duration=duration, 
                    period=0.01
                )
            elif action == 'move right':
                robot_controller.arm.send_cartesian_velocity(
                    linear_velocity=(-self.cartesian_linear_scale, 0.0, 0.0), 
                    angular_velocity=(0.0, 0.0, 0.0), 
                    fingers=(0.0, 0.0, 0.0), 
                    duration=duration, 
                    period=0.01
                )
            elif action == 'move forward':
                robot_controller.arm.send_cartesian_velocity(
                    linear_velocity=(0.0, -self.cartesian_linear_scale, 0.0), 
                    angular_velocity=(0.0, 0.0, 0.0), 
                    fingers=(0.0, 0.0, 0.0), 
                    duration=duration, 
                    period=0.01
                )
            elif action == 'move backward':
                robot_controller.arm.send_cartesian_velocity(
                    linear_velocity=(0.0, self.cartesian_linear_scale, 0.0), 
                    angular_velocity=(0.0, 0.0, 0.0), 
                    fingers=(0.0, 0.0, 0.0), 
                    duration=duration, 
                    period=0.01
                )
            elif action == 'move up':
                robot_controller.arm.send_cartesian_velocity(
                    linear_velocity=(0.0, 0.0, self.cartesian_linear_scale), 
                    angular_velocity=(0.0, 0.0, 0.0), 
                    fingers=(0.0, 0.0, 0.0), 
                    duration=duration, 
                    period=0.01
                )
            elif action == 'move down':
                robot_controller.arm.send_cartesian_velocity(
                    linear_velocity=(0.0, 0.0, -self.cartesian_linear_scale), 
                    angular_velocity=(0.0, 0.0, 0.0), 
                    fingers=(0.0, 0.0, 0.0), 
                    duration=duration, 
                    period=0.01
                )
            
            # Angular movements - CORRECTED MAPPINGS
            # Turn (Y-axis rotation)
            elif action == 'turn left':
                robot_controller.arm.send_cartesian_velocity(
                    linear_velocity=(0.0, 0.0, 0.0),
                    angular_velocity=(0.0, -self.cartesian_angular_scale, 0.0),
                    fingers=(0.0, 0.0, 0.0),
                    duration=duration,
                    period=0.01
                )
            elif action == 'turn right':
                robot_controller.arm.send_cartesian_velocity(
                    linear_velocity=(0.0, 0.0, 0.0),
                    angular_velocity=(0.0, self.cartesian_angular_scale, 0.0),
                    fingers=(0.0, 0.0, 0.0),
                    duration=duration,
                    period=0.01
                )
            
            # Pitch (X-axis rotation)
            elif action == 'pitch up':
                robot_controller.arm.send_cartesian_velocity(
                    linear_velocity=(0.0, 0.0, 0.0),
                    angular_velocity=(-self.cartesian_angular_scale, 0.0, 0.0),
                    fingers=(0.0, 0.0, 0.0),
                    duration=duration,
                    period=0.01
                )
            elif action == 'pitch down':
                robot_controller.arm.send_cartesian_velocity(
                    linear_velocity=(0.0, 0.0, 0.0),
                    angular_velocity=(self.cartesian_angular_scale, 0.0, 0.0),
                    fingers=(0.0, 0.0, 0.0),
                    duration=duration,
                    period=0.01
                )
            
            # Roll (Z-axis rotation)
            elif action == 'roll left':
                robot_controller.arm.send_cartesian_velocity(
                    linear_velocity=(0.0, 0.0, 0.0),
                    angular_velocity=(0.0, 0.0, -self.cartesian_angular_scale),
                    fingers=(0.0, 0.0, 0.0),
                    duration=duration,
                    period=0.01
                )
            elif action == 'roll right':
                robot_controller.arm.send_cartesian_velocity(
                    linear_velocity=(0.0, 0.0, 0.0),
                    angular_velocity=(0.0, 0.0, self.cartesian_angular_scale),
                    fingers=(0.0, 0.0, 0.0),
                    duration=duration,
                    period=0.01
                )
            
            # Gripper commands
            elif action == 'open gripper':
                robot_controller.arm.send_angular_velocity(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                    hand_mode=1, 
                    fingers=(-3000.0, -3000.0, -3000.0), 
                    duration=duration, 
                    period=0.005
                )
            elif action == 'close gripper':
                robot_controller.arm.send_angular_velocity(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                    hand_mode=1, 
                    fingers=(3000.0, 3000.0, 3000.0), 
                    duration=duration, 
                    period=0.005
                )
            elif action == 'stop':
                # Just wait for the duration
                time.sleep(duration)
            else:
                print(f"Unknown action: {action}")
                return False
            return True
        except Exception as e:
            print(f"Error executing action {action}: {e}")
            return False
    
    def execute_task(self, max_steps=50):
        """
        Execute the task by repeatedly capturing images, getting LLM guidance,
        and executing robot actions until the task is complete or max_steps is reached.
        
        Args:
            max_steps (int): Maximum number of steps to execute before stopping.
            
        Returns:
            dict: Execution result containing status and details.
        """
        if not self.task:
            return {"status": "error", "message": "No task set. Call set_task() first."}
        
        step_count = 0
        
        while step_count < max_steps:
            step_count += 1
            print(f"\nExecuting step {step_count} for task: {self.task}")
            
            # Small delay to ensure we get a fresh frame
            time.sleep(0.1)
            
            # Capture and process images using the instance's camera interface
            image_paths = self.capture_and_process_images()
            
            if not image_paths:
                return {"status": "error", "message": "Failed to capture any images for execution."}
            
            # Upload images
            uploaded_image_urls = self.upload_images(image_paths)
            
            if not uploaded_image_urls:
                return {"status": "error", "message": "Failed to upload any images for execution."}
            
            # Get and format the current Cartesian position
            cartesian_position = None
            cartesian_position_str = ""
            if self.robot_controller:
                cartesian_position = self.get_cartesian_position(self.robot_controller)
                cartesian_position_str = self.format_cartesian_position(cartesian_position)
            
            # Create prompt with previous analysis and cartesian position if available
            if self.previous_analysis:
                task_prompt = f'task: {self.task}\n\nPrevious Analysis: {self.previous_analysis}'
            else:
                task_prompt = f'task: {self.task}'
                
            # Add cartesian position information if available
            if cartesian_position_str:
                task_prompt += f'\n\n{cartesian_position_str}'
            
            # Call LLM with images and prompt
            llm_response = call_llm_with_images(
                task_prompt, 
                uploaded_image_urls, 
                model_name=self.model_name, 
                system_prompt=self.action_prompt,
                debug=True
            )
            
            # Check if the response contains an error
            if isinstance(llm_response, str) and llm_response.startswith("error:"):
                print(f"LLM call failed: {llm_response}")
                continue
            
            try:
                # Extract and store the current analysis
                current_analysis = self.extract_analysis(llm_response)
                if current_analysis:
                    print(f"LLM Analysis:\n{current_analysis}\n")
                    self.previous_analysis = current_analysis  # Save for next iteration
                
                # Parse and execute the action
                action, duration = self.parse_action_duration(llm_response)
                print(f"Executing: {action}, {duration}")
                
                # Log the action along with cartesian position
                action_log = {
                    "step": step_count,
                    "action": action,
                    "duration": duration,
                    "analysis": current_analysis
                }
                
                # Add cartesian position to the log if available
                if cartesian_position:
                    action_log["cartesian_position"] = cartesian_position
                
                self.execution_log.append(action_log)
                
                # Execute the action if we have a robot controller
                if self.robot_controller:
                    success = self.execute_action(action, duration, self.robot_controller)
                    if not success:
                        return {"status": "error", "message": f"Failed to execute action: {action}"}
                else:
                    # Simulate execution
                    print(f"Simulating action: {action} for {duration} seconds")
                    time.sleep(min(duration, 1.0))  # Simulate shorter to make testing faster
                
                # Check if we're done
                if action == 'stop':
                    return {
                        "status": "success", 
                        "message": "Task completed successfully.", 
                        "steps_taken": step_count,
                        "execution_log": self.execution_log
                    }
                
            except ValueError as e:
                print(f"Error parsing LLM response: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error during execution: {e}")
                return {"status": "error", "message": f"Execution error: {str(e)}"}
        
        # If we reach here, we've hit the maximum number of steps
        return {
            "status": "incomplete", 
            "message": f"Reached maximum steps ({max_steps}) without completing the task.", 
            "steps_taken": step_count,
            "execution_log": self.execution_log
        }
    
    def get_execution_status(self):
        """
        Get the current execution status.
        
        Returns:
            dict: Dictionary containing the task and execution log.
        """
        return {
            "task": self.task,
            "steps_taken": len(self.execution_log),
            "execution_log": self.execution_log
        }
    
    def set_action_prompt(self, new_prompt):
        """
        Set a new action prompt.
        
        Args:
            new_prompt (str): The new prompt to use.
        """
        self.action_prompt = new_prompt


if __name__ == "__main__":
    # Initialize the camera interface
    try:
        from sim_env.Kinova_gen2.src.devices.camera_interface import MultiCameraInterface
        camera_interface = MultiCameraInterface(height=240, width=320)
        print("Camera interface initialized in main function.")
        time.sleep(3)  # Wait for cameras to initialize
    except ImportError:
        print("Warning: MultiCameraInterface not found. Using None instead.")
        camera_interface = None
    
    # Example usage with camera interface passed to constructor
    agent = ActionExecutionAgent(camera_interface=camera_interface)
    
    # Set a task
    task = "There is a user sitting next to the table, locate the user"
    agent.set_task(task)
    
    # Execute the task (in simulation mode without robot controller)
    result = agent.execute_task(max_steps=10)
    
    print(f"\nExecution Result: {result['status']}")
    print(f"Message: {result['message']}")
    print(f"Steps taken: {result['steps_taken']}")
    
    # Print the execution log
    print("\nExecution Log:")
    for step in agent.execution_log:
        print(f"Step {step['step']}: {step['action']}, {step['duration']} seconds") 
    
    # test actions
    # agent.execute_action(action="move left", duration=1, robot_controller=agent.robot_controller)
    # agent.execute_action(action="turn right", duration=1, robot_controller=agent.robot_controller)
    # agent.execute_action(action="turn left", duration=1, robot_controller=agent.robot_controller)
    # agent.execute_action(action="pitch up", duration=1, robot_controller=agent.robot_controller)
    # agent.execute_action(action="pitch down", duration=1, robot_controller=agent.robot_controller)