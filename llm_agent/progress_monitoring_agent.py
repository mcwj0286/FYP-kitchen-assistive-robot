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

class ProgressMonitoringAgent:
    """
    A class that monitors the progress of a task by capturing images and 
    sending them to an LLM for analysis and feedback.
    """
    
    def __init__(self, server_url=None):
        """
        Initialize the ProgressMonitoringAgent.
        
        Args:
            server_url (str, optional): URL of the server to upload images to.
                If None, images will be uploaded to Cloudinary.
        """
        self.server_url = server_url or os.getenv("IMAGE_SERVER_URL")
        self.model_name = os.getenv("MODEL_NAME")
        
        # Updated strict monitoring prompt
        self.system_prompt = """
        You are a highly skilled AI progress monitoring assistant analyzing images of a robotic arm performing tasks. Your output must strictly adhere to the following structure:

        Analysis: Provide a succinct summary of your observations from the image. Focus only on details directly related to the task's progress or anomalies.

        Classification: Classify the task status as exactly one of the following:
            - "executing" (if the task is actively in progress without issues),
            - "finished" (if the task has been successfully completed),
            - "failed" (if the task has encountered insurmountable issues),
            - "collision" (if you detect the robotic arm colliding with a non-target object).
        
        Operation: Based on your classification, issue an operational command as follows:
            - Output "continue" only if the classification is "executing".
            - Output "stop" if the classification is "finished", "failed", or "collision".

        Your response must include only these three sections labeled **Analysis**, **Classification**, and **Operation** (each followed by a colon) with no additional commentary or text.
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
    
    def monitor_progress(self, task_name="", camera_interface=None, image_paths=None):
        """
        Monitor the progress of a task by capturing images, uploading them,
        and getting feedback from an LLM.
        
        Args:
            task_context (str): Additional context about the task being monitored.
            camera_interface: Optional camera interface to use for capturing frames.
            
        Returns:
            dict: The LLM's feedback on the task progress.
        """
        # Capture and save images
        if image_paths is None:
            image_paths = self.capture_and_process_images(camera_interface)
        
        if not image_paths:
            return "Failed to capture any images for monitoring."
        
        # Upload images
        uploaded_urls = self.upload_images(image_paths)
        
        if not uploaded_urls:
            return "Failed to upload any images for monitoring."
        
        
        prompt = f"Current executing task: {task_name}"
        
        # Call LLM with images
        llm_response = call_llm_with_images(
            prompt, 
            uploaded_urls, 
            model_name=self.model_name, 
            system_prompt=self.system_prompt
        )
        
        # Parse the response
        analysis, classification, operation = self.parse_monitoring_response(llm_response)
        
        # You can now use these parsed components as needed
        # For example, return just the operation for other agents to use
        return {
            "raw_response": llm_response,
            "analysis": analysis,
            "classification": classification,
            "operation": operation
        }
    
    def set_monitoring_prompt(self, new_prompt):
        """
        Set a new monitoring prompt template.
        
        Args:
            new_prompt (str): The new prompt template to use.
        """
        self.monitoring_prompt_template = new_prompt

    def parse_monitoring_response(self, response):
        """
        Parse the LLM's monitoring response into analysis, classification, and operation.
        
        Args:
            response (str): The raw response from the LLM.
            
        Returns:
            tuple: (analysis, classification, operation) as strings.
                If any section is not found, its value will be an empty string.
        """
        # Initialize default values
        analysis = ""
        classification = ""
        operation = ""
        
        # Split the response into lines for processing
        lines = response.strip().split('\n')
        
        # Track which section we're currently parsing
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Check for section headers
            if line.lower().startswith("**analysis"):
                current_section = "analysis"
                # Extract content if it's on the same line (e.g., "**Analysis**: content")
                colon_pos = line.find(':')
                if colon_pos != -1:
                    analysis = line[colon_pos + 1:].strip()
                continue
                
            elif line.lower().startswith("**classification"):
                current_section = "classification"
                colon_pos = line.find(':')
                if colon_pos != -1:
                    classification = line[colon_pos + 1:].strip()
                continue
                
            elif line.lower().startswith("**operation"):
                current_section = "operation"
                colon_pos = line.find(':')
                if colon_pos != -1:
                    operation = line[colon_pos + 1:].strip()
                continue
            
            # Add content to the current section
            if current_section == "analysis" and line:
                if analysis and not line.startswith("**"):
                    analysis += " " + line
                elif not analysis and not line.startswith("**"):
                    analysis = line
                    
            elif current_section == "classification" and line:
                if not classification and not line.startswith("**"):
                    classification = line
                    
            elif current_section == "operation" and line:
                if not operation and not line.startswith("**"):
                    operation = line
        
        # Clean up the outputs - remove quotes if present
        classification = classification.strip('"\'')
        operation = operation.strip('"\'')
        
        # Validate classification is one of the expected values
        valid_classifications = ["executing", "finished", "failed", "collision"]
        if classification.lower() not in valid_classifications:
            classification = ""
        
        # Validate operation is one of the expected values
        valid_operations = ["continue", "stop"]
        if operation.lower() not in valid_operations:
            operation = ""
        
        return analysis, classification, operation


if __name__ == "__main__":
    # Example usage
    agent = ProgressMonitoringAgent()
    image_paths = {'cam0':'/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robot/workflow/captured_image_cam_0.jpg'}
    feedback = agent.monitor_progress("grasp the cup and put it on the plate",image_paths=image_paths)
    print(f"Progress Feedback: {feedback}")
