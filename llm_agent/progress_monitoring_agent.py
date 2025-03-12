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
        
        # Initialize the base monitoring prompt
        self.system_prompt = """
        You are an AI assistant helping to monitor the progress of a kitchen task.
        
        Please analyze the provided image(s) and answer the following questions:
        1. What is the current state of the task?
        2. Has there been progress since the last update?
        3. Are there any issues or obstacles that might prevent successful completion?
        
        Please be specific and descriptive in your analysis.
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
    
    def monitor_progress(self, task_name="", camera_interface=None,image_paths=None):
        """
        Monitor the progress of a task by capturing images, uploading them,
        and getting feedback from an LLM.
        
        Args:
            task_context (str): Additional context about the task being monitored.
            camera_interface: Optional camera interface to use for capturing frames.
            
        Returns:
            str: The LLM's feedback on the task progress.
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
        
        return llm_response
    
    def set_monitoring_prompt(self, new_prompt):
        """
        Set a new monitoring prompt template.
        
        Args:
            new_prompt (str): The new prompt template to use.
        """
        self.monitoring_prompt_template = new_prompt


if __name__ == "__main__":
    # Example usage
    agent = ProgressMonitoringAgent()
    image_paths = {'cam0':'/Users/johnmok/Documents/GitHub/FYP-kitchen-assistive-robot/llm_agent/assortment-delicious-healthy-food_23-2149043057.jpg'}
    feedback = agent.monitor_progress("Preparing a sandwich",image_paths=image_paths)
    print(f"Progress Feedback: {feedback}")
