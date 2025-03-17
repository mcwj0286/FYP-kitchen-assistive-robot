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

class KitchenCompanionAgent:
    """
    A class that continuously monitors user activities through camera feeds,
    identifies when assistance might be beneficial, and offers appropriate help.
    """
    
    def __init__(self, server_url=None):
        """
        Initialize the KitchenCompanionAgent.
        
        Args:
            server_url (str, optional): URL of the server to upload images to.
                If None, images will be uploaded to Cloudinary.
        """
        self.server_url = server_url or os.getenv("IMAGE_SERVER_URL")
        self.model_name = os.getenv("MODEL_NAME")
        
        # Initialize system prompt for the companion agent
        self.system_prompt = """
        You are an AI assistant for a kitchen assistive robot. Your role is to analyze images 
        of a person in a kitchen environment, identify what they're doing, determine if they 
        might need assistance, and suggest appropriate help.
        
        Your response must follow this exact format:
        
        **Activity**: [Describe what the person is doing in the kitchen]
        
        **Needs Assistance**: [Yes/No]
        
        **Assistance Type**: [Only if assistance is needed: describe what kind of help might be useful]
        
        **Suggested Dialog**: [Only if assistance is needed: provide a natural, conversational 
        offer of assistance that the robot could speak to the person]
        
        For example, if someone is struggling to open a jar, you might respond:
        **Activity**: Person is trying to open a jar but appears to be struggling.
        **Needs Assistance**: Yes
        **Assistance Type**: Help opening the jar lid
        **Suggested Dialog**: "I notice you're trying to open that jar. Would you like me to help you with that?"
        
        Focus on identifying genuine needs for assistance based on:
        - Tasks that require physical strength the person may lack
        - Repetitive or inefficient movements suggesting difficulty
        - Awkward postures or positions that could be ergonomically improved
        - Safety concerns where the robot could reduce risk
        
        Be conservative in offering help - don't suggest assistance unless it's clearly warranted.
        """
        
        # Store user interaction history
        self.interaction_history = []
        self.user_preferences = {}
        
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
    
    def monitor_user_activity(self, camera_interface=None, image_paths=None):
        """
        Monitor user activity by capturing images, analyzing them,
        and determining if assistance is needed.
        
        Args:
            camera_interface: Optional camera interface to use for capturing frames.
            image_paths: Optional dictionary of pre-captured images to use.
            
        Returns:
            dict: Analysis of user activity and potential assistance.
        """
        # Capture and save images if not provided
        if image_paths is None:
            image_paths = self.capture_and_process_images(camera_interface)
        
        if not image_paths:
            return {"error": "Failed to capture any images for monitoring."}
        
        # Upload images
        uploaded_urls = self.upload_images(image_paths)
        
        if not uploaded_urls:
            return {"error": "Failed to upload any images for monitoring."}
        
        # Prompt for the LLM to analyze the image and determine if assistance is needed
        prompt = "Please analyze the current kitchen scene and identify if the person needs any assistance."
        
        # Call LLM with images
        llm_response = call_llm_with_images(
            prompt, 
            uploaded_urls, 
            model_name=self.model_name, 
            system_prompt=self.system_prompt
        )
        
        # Parse the response
        analysis = self.parse_monitoring_response(llm_response)
        
        # Add to interaction history
        self.interaction_history.append({
            "timestamp": time.time(),
            "analysis": analysis
        })
        
        return analysis
    
    def parse_monitoring_response(self, response):
        """
        Parse the LLM's response into structured components.
        
        Args:
            response (str): The raw response from the LLM.
            
        Returns:
            dict: Structured analysis with activity, assistance_needed, assistance_type, and dialog.
        """
        result = {
            "activity": "",
            "assistance_needed": False,
            "assistance_type": "",
            "suggested_dialog": ""
        }
        
        # Split the response into lines
        lines = response.strip().split('\n')
        
        current_field = None
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check for field markers
            if "**Activity**:" in line:
                current_field = "activity"
                result["activity"] = line.split("**Activity**:", 1)[1].strip()
            elif "**Needs Assistance**:" in line:
                current_field = "assistance_needed"
                value = line.split("**Needs Assistance**:", 1)[1].strip().lower()
                result["assistance_needed"] = value == "yes"
            elif "**Assistance Type**:" in line:
                current_field = "assistance_type"
                result["assistance_type"] = line.split("**Assistance Type**:", 1)[1].strip()
            elif "**Suggested Dialog**:" in line:
                current_field = "suggested_dialog"
                result["suggested_dialog"] = line.split("**Suggested Dialog**:", 1)[1].strip()
            # Continue adding to the current field if it's a continuation line
            elif current_field and not line.startswith("**"):
                result[current_field] += " " + line
        
        return result
    
    def offer_assistance(self, assistance_info, tts_interface=None):
        """
        Present the assistance offer to the user.
        
        Args:
            assistance_info (dict): The analysis containing suggested dialog.
            tts_interface: Optional text-to-speech interface for speaking the dialog.
            
        Returns:
            str: The dialog presented to the user.
        """
        if not assistance_info.get("assistance_needed", False):
            return "No assistance needed at this time."
        
        dialog = assistance_info.get("suggested_dialog", "")
        
        if not dialog:
            dialog = f"Would you like help with {assistance_info.get('assistance_type', 'your task')}?"
        
        # Use text-to-speech if available
        if tts_interface:
            tts_interface.speak(dialog)
        
        # Print the dialog as well (for debugging or fallback)
        print(f"Robot: {dialog}")
        
        return dialog
    
    def handle_user_response(self, user_response, assistance_info):
        """
        Process the user's response to the offer of assistance.
        
        Args:
            user_response (str): The user's verbal response.
            assistance_info (dict): The original analysis that prompted the offer.
            
        Returns:
            dict: Action to take based on user response.
        """
        # Convert to lowercase for easier processing
        response_lower = user_response.lower()
        
        # Check for positive responses
        positive_keywords = ["yes", "yeah", "sure", "please", "help", "okay", "ok"]
        negative_keywords = ["no", "nope", "don't", "not", "later"]
        
        # Determine if this is a positive or negative response
        if any(keyword in response_lower for keyword in positive_keywords):
            # User accepted help - prepare to coordinate with other agents
            return {
                "accepted": True,
                "assistance_type": assistance_info.get("assistance_type", ""),
                "action": "initiate_assistance"
            }
        elif any(keyword in response_lower for keyword in negative_keywords):
            # User declined help
            return {
                "accepted": False,
                "action": "resume_monitoring"
            }
        else:
            # Unclear response - ask for clarification
            return {
                "accepted": None,
                "action": "clarify",
                "clarification_message": "I didn't understand. Would you like me to help you?"
            }
    
    def initiate_assistance(self, assistance_info, action_planning_agent=None, tts_interface=None):
        """
        Initiate the assistance by generating a plan and connecting to action agents.
        
        Args:
            assistance_info (dict): Information about the required assistance.
            action_planning_agent: The action planning agent to use for planning.
            tts_interface: Optional text-to-speech interface for speaking to the user.
            
        Returns:
            dict: The action plan for providing assistance.
        """
        # Step 1: Confirm with the user and inform them we're preparing to help
        confirmation_message = f"I'll help you with {assistance_info.get('assistance_type')}. Just a moment while I prepare."
        
        if tts_interface:
            tts_interface.speak(confirmation_message)
        print(f"Robot: {confirmation_message}")
        
        # Step 2: Generate assistance plan using the action planning agent
        if action_planning_agent:
            # Formulate a prompt for the action planning agent
            task_prompt = f"Help the user with {assistance_info.get('assistance_type')}. The user is currently {assistance_info.get('activity')}"
            
            # Generate plan
            action_plan_text = action_planning_agent.generate_action_plan(task_prompt)
            action_plan = action_planning_agent.parse_action_plan(action_plan_text)
            
            return {
                "status": "plan_generated",
                "action_plan": action_plan,
                "original_assistance": assistance_info
            }
        else:
            # No action planning agent available
            return {
                "status": "error",
                "message": "Action planning agent not available",
                "original_assistance": assistance_info
            }
    
    def set_companion_prompt(self, new_prompt):
        """
        Set a new system prompt for the companion agent.
        
        Args:
            new_prompt (str): The new prompt to use.
        """
        self.system_prompt = new_prompt
    
    def update_user_preferences(self, assistance_type, accepted):
        """
        Update user preferences based on interaction outcomes.
        
        Args:
            assistance_type (str): The type of assistance offered.
            accepted (bool): Whether the user accepted the assistance.
        """
        if assistance_type not in self.user_preferences:
            self.user_preferences[assistance_type] = {
                "offered": 0,
                "accepted": 0
            }
        
        self.user_preferences[assistance_type]["offered"] += 1
        if accepted:
            self.user_preferences[assistance_type]["accepted"] += 1


if __name__ == "__main__":
    # Example usage
    agent = KitchenCompanionAgent()
    
    # Example image path for testing
    test_image_path = {'cam0': '/Users/johnmok/Documents/GitHub/FYP-kitchen-assistive-robot/llm_agent/opening_jars-e1295217385283.jpg'}
    
    # Monitor user activity
    analysis = agent.monitor_user_activity(image_paths=test_image_path)
    print(f"Activity Analysis: {analysis}")
    
    # If assistance is needed, offer it
    if analysis.get("assistance_needed", False):
        agent.offer_assistance(analysis)
        
        # Simulate user response (in a real scenario, this would come from speech recognition)
        user_response = input("User response: ")
        
        # Handle the response
        response_action = agent.handle_user_response(user_response, analysis)
        print(f"Response Action: {response_action}")
        
        # If the user accepted assistance, initiate it
        if response_action.get("accepted", False):
            # In a real scenario, you would have the action planning agent available
            # For this example, we'll just print what would happen
            print("Initiating assistance process...") 