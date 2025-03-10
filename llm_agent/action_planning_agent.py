import os
import sys
import time
import cv2

from dotenv import load_dotenv
load_dotenv()  # Load .env variables

# Import functions from get_prompt.py
from get_prompt import (
    upload_images_to_cloudinary,
    upload_image_to_server,
    call_llm_with_images,
    save_images
)

class ActionPlanningAgent:
    """
    A class that generates a detailed action plan based on user prompts and camera images.
    """
    
    def __init__(self, server_url=None):
        """
        Initialize the ActionPlanningAgent.
        
        Args:
            server_url (str, optional): URL of the server to upload images to.
                If None, images will be uploaded to Cloudinary.
        """
        self.server_url = server_url or os.getenv("IMAGE_SERVER_URL")
        self.model_name = os.getenv("MODEL_NAME")
        
        # Initialize the base planning prompt
        self.system_prompt = """
        You are an expert in tactical planning. Using the provided user goal and the environmental 
        context from a live camera feed, devise a well-organized action plan in numbered steps. 
        Each step should describe a concrete action (e.g., "1. Pick up the bowl", "2. Place it in the cabinet", 
        "3. Clean the table with a towel") with no extraneous details.
        
        Consider the following:
        - Spatial arrangement of objects
        - Required tools and their locations
        - Physical constraints of a robotic arm
        - Safety considerations
        - Logical sequence of operations
        
        Please provide a clear, numbered list of actions to accomplish this goal based on what you see in the image.
        """
    
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
    
    def generate_action_plan(self, user_goal, camera_interface=None,image_paths=None):
        """
        Generate an action plan based on the user goal and images from cameras.
        
        Args:
            user_goal (str): The goal specified by the user.
            camera_interface: Optional camera interface to use for capturing frames.
            
        Returns:
            str: The generated action plan.
        """
        # Capture and save images
        if image_paths is None:
            image_paths = self.capture_and_process_images(camera_interface)
        
        if not image_paths:
            return "Failed to capture any images for planning."
        
        # Upload images
        uploaded_urls = self.upload_images(image_paths)
        
        if not uploaded_urls:
            return "Failed to upload any images for planning."
        
      
    
        
        # Call LLM with images
        llm_response = call_llm_with_images(
            user_goal, 
            uploaded_urls, 
            model_name=self.model_name, 
            system_prompt=self.system_prompt
        )
        
        return llm_response
    
    def parse_action_plan(self, action_plan_text):
        """
        Parse the action plan text into a structured format.
        
        Args:
            action_plan_text (str): The raw action plan text from the LLM.
            
        Returns:
            list: A list of action steps.
        """
        # Split the text by newlines and filter out empty lines
        lines = [line.strip() for line in action_plan_text.split('\n') if line.strip()]
        
        # Extract steps (assuming they are numbered)
        steps = []
        for line in lines:
            # Look for lines that start with numbers followed by a period or parenthesis
            if line and (line[0].isdigit() or (len(line) > 1 and line[0:2].isdigit())):
                # Extract the step content (remove the number and any delimiter)
                step_content = line.split('.', 1)[-1].strip()
                if not step_content and len(line.split(')', 1)) > 1:
                    step_content = line.split(')', 1)[-1].strip()
                
                if step_content:
                    steps.append(step_content)
        
        # If no steps were extracted, try a different approach
        if not steps and lines:
            # Just return all non-empty lines
            steps = lines
        
        return steps
    
    def set_planning_prompt(self, new_prompt):
        """
        Set a new planning prompt template.
        
        Args:
            new_prompt (str): The new prompt template to use.
        """
        self.system_prompt = new_prompt


if __name__ == "__main__":
    # Example usage
    agent = ActionPlanningAgent()
    image_paths = {'cam0':'/Users/johnmok/Documents/GitHub/FYP-kitchen-assistive-robot/llm_agent/assortment-delicious-healthy-food_23-2149043057.jpg'}
    
    action_plan = agent.generate_action_plan("Clean the kitchen table",image_paths=image_paths)
    print(f"Action Plan:\n{action_plan}")
    
    # Parse the action plan into steps
    steps = agent.parse_action_plan(action_plan)
    print("\nParsed Steps:")
    for i, step in enumerate(steps, 1):
        print(f"{i}. {step}") 