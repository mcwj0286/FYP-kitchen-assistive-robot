import os
import sys
import time
import cv2

from dotenv import load_dotenv
load_dotenv()  # Load .env variables

# Import functions from get_prompt.py
from .get_prompt import (
    upload_images_to_cloudinary,
    upload_image_to_server,
    call_llm_with_images,
    save_images
)

class ActionExecutionAgent:
    """
    A class that translates high-level action plans into specific robot commands
    and provides real-time guidance during execution.
    """
    
    def __init__(self, server_url=None):
        """
        Initialize the ActionExecutionAgent.
        
        Args:
            server_url (str, optional): URL of the server to upload images to.
                If None, images will be uploaded to Cloudinary.
        """
        self.server_url = server_url or os.getenv("IMAGE_SERVER_URL")
        self.model_name = os.getenv("MODEL_NAME")
        self.system_prompt = os.getenv("SYSTEM_PROMPT")
        
        # Initialize the base execution prompt
        self.execution_prompt_template = """
        You are an expert in robotic task execution. Your role is to translate high-level action steps 
        into specific, executable commands for a kitchen assistive robot.
        
        Current action step: {action_step}
        
        Based on the image of the current environment, provide detailed guidance on how to execute this step:
        1. Identify the exact objects involved in this step
        2. Specify the precise movements required (e.g., grasp coordinates, movement trajectory)
        3. Describe any precautions or adjustments needed based on the current scene
        4. Indicate when the step is complete and what the success criteria are
        
        Please be specific and focus only on the current step.
        """
        
        # Track execution state
        self.current_step_index = 0
        self.action_plan = []
        self.execution_status = []
    
    def capture_and_process_images(self, camera_interface=None):
        """
        Capture images from cameras and process them.
        
        Args:
            camera_interface: Instance of a camera interface to capture frames.
                If None, will try to import and use MultiCameraInterface.
                
        Returns:
            dict: Dictionary of camera IDs to image paths.
        """
        if camera_interface is None:
            try:
                # Try to import the MultiCameraInterface
                from sim_env.Kinova_gen2.src.devices.camera_interface import MultiCameraInterface
                camera_interface = MultiCameraInterface(height=240, width=320)
                time.sleep(3)  # Wait for cameras to initialize
            except ImportError:
                print("Warning: MultiCameraInterface not found. No images will be captured.")
                return {}
        
        # Capture frames from cameras
        frames = camera_interface.capture_frames()
        
        # Close camera interface if it was created in this method
        if camera_interface is not None:
            camera_interface.close()
        
        # Save the captured frames and return their paths
        return save_images(frames)
    
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
    
    def set_action_plan(self, action_plan):
        """
        Set the action plan to be executed.
        
        Args:
            action_plan (list): List of action steps to execute.
        """
        self.action_plan = action_plan
        self.current_step_index = 0
        self.execution_status = ["pending"] * len(action_plan)
    
    def get_current_step(self):
        """
        Get the current action step to be executed.
        
        Returns:
            str: The current action step, or None if all steps are completed.
        """
        if self.current_step_index < len(self.action_plan):
            return self.action_plan[self.current_step_index]
        return None
    
    def execute_current_step(self, camera_interface=None, robot_controller=None):
        """
        Execute the current action step.
        
        Args:
            camera_interface: Optional camera interface to use for capturing frames.
            robot_controller: Controller for the robot to execute commands.
            
        Returns:
            dict: Execution result containing status and details.
        """
        current_step = self.get_current_step()
        if not current_step:
            return {"status": "completed", "message": "All steps have been executed."}
        
        # Capture and save images
        image_paths = self.capture_and_process_images(camera_interface)
        
        if not image_paths:
            return {"status": "error", "message": "Failed to capture any images for execution."}
        
        # Upload images
        uploaded_urls = self.upload_images(image_paths)
        
        if not uploaded_urls:
            return {"status": "error", "message": "Failed to upload any images for execution."}
        
        # Construct the prompt with the current action step
        prompt = self.execution_prompt_template.format(action_step=current_step)
        
        # Call LLM with images to get execution guidance
        execution_guidance = call_llm_with_images(
            prompt, 
            uploaded_urls, 
            model_name=self.model_name, 
            system_prompt=self.system_prompt
        )
        
        # Here you would integrate with the robot controller to execute the step
        # For now, we'll just simulate execution
        execution_result = self._simulate_execution(execution_guidance, robot_controller)
        
        # Update execution status
        self.execution_status[self.current_step_index] = execution_result["status"]
        
        # Move to the next step if successful
        if execution_result["status"] == "success":
            self.current_step_index += 1
        
        return {
            "step": current_step,
            "guidance": execution_guidance,
            "result": execution_result
        }
    
    def _simulate_execution(self, execution_guidance, robot_controller=None):
        """
        Simulate the execution of a step based on the guidance.
        In a real implementation, this would interface with the robot controller.
        
        Args:
            execution_guidance (str): The guidance from the LLM.
            robot_controller: Controller for the robot to execute commands.
            
        Returns:
            dict: Simulated execution result.
        """
        # In a real implementation, this would parse the guidance and send commands to the robot
        # For now, we'll just simulate success
        if robot_controller:
            # Here you would use the robot_controller to execute the commands
            # For example: robot_controller.execute_commands(parsed_commands)
            pass
        
        # Simulate a successful execution
        return {
            "status": "success",
            "message": "Step executed successfully (simulated)."
        }
    
    def execute_all_steps(self, camera_interface=None, robot_controller=None):
        """
        Execute all steps in the action plan.
        
        Args:
            camera_interface: Optional camera interface to use for capturing frames.
            robot_controller: Controller for the robot to execute commands.
            
        Returns:
            list: List of execution results for each step.
        """
        results = []
        while self.get_current_step():
            result = self.execute_current_step(camera_interface, robot_controller)
            results.append(result)
            
            # Break if there was an error
            if result.get("result", {}).get("status") != "success":
                break
                
        return results
    
    def get_execution_status(self):
        """
        Get the current execution status of all steps.
        
        Returns:
            dict: Dictionary containing the action plan and execution status.
        """
        return {
            "total_steps": len(self.action_plan),
            "completed_steps": self.current_step_index,
            "action_plan": self.action_plan,
            "execution_status": self.execution_status
        }
    
    def set_execution_prompt(self, new_prompt):
        """
        Set a new execution prompt template.
        
        Args:
            new_prompt (str): The new prompt template to use.
        """
        self.execution_prompt_template = new_prompt


if __name__ == "__main__":
    # Example usage
    agent = ActionExecutionAgent()
    
    # Set an example action plan
    action_plan = [
        "Pick up the sponge from the counter",
        "Wet the sponge with water",
        "Wipe the table surface in circular motions"
    ]
    
    agent.set_action_plan(action_plan)
    
    # Execute the first step
    result = agent.execute_current_step()
    print(f"Execution Result:\n{result}")
    
    # Get execution status
    status = agent.get_execution_status()
    print(f"\nExecution Status:\n{status}") 