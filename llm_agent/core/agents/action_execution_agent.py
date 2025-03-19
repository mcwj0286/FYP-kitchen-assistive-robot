import os
from typing import Dict, List, Any, Optional

from ..base_agent import BaseAgent
from ..tools.camera_tool import CameraTool
from ..parsers.base_parser import BaseParser

class ActionExecutionAgent(BaseAgent):
    """
    An agent that executes an action plan by providing step-by-step instructions.
    """
    
    def __init__(
        self,
        name: str = "action_execution_agent",
        description: str = "Executes an action plan by providing detailed instructions for each step",
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        server_url: Optional[str] = None,
        enable_short_term_memory: bool = True,
        enable_long_term_memory: bool = True,
        memory_storage_path: Optional[str] = "memory/action_execution",
        camera_interface=None
    ):
        """
        Initialize the ActionExecutionAgent.
        
        Args:
            name: The name of the agent.
            description: A description of the agent's purpose.
            model_name: The name of the LLM model to use.
            system_prompt: The system prompt to use for LLM interactions.
            server_url: URL of the server to upload images to.
            enable_short_term_memory: Whether to enable short-term memory.
            enable_long_term_memory: Whether to enable long-term memory.
            memory_storage_path: Path to store long-term memory (if enabled).
            camera_interface: Optional camera interface to use for capturing frames.
        """
        super().__init__(
            name=name,
            description=description,
            model_name=model_name,
            system_prompt=system_prompt,
            server_url=server_url,
            enable_short_term_memory=enable_short_term_memory,
            enable_long_term_memory=enable_long_term_memory,
            memory_storage_path=memory_storage_path
        )
        
        # Register camera tool
        self.register_tool(CameraTool(camera_interface=camera_interface))
        
        # Custom parser could be added here if needed
        
        # Set default execution prompt if none is provided
        if not self.system_prompt:
            self.system_prompt = self._get_default_execution_prompt()
    
    def _get_default_execution_prompt(self) -> str:
        """
        Get the default execution prompt for the agent.
        
        Returns:
            str: The default execution prompt.
        """
        return """
        You are an expert in providing detailed instructions for executing specific actions.
        For the given action step, provide clear, concise instructions that would help someone 
        (or a robot) complete the task correctly.
        
        For each action:
        1. Break it down into small, executable sub-steps if needed
        2. Explain precisely what movements or manipulations are required
        3. Identify key objects and their spatial relationships
        4. Note any safety considerations or potential issues
        5. Describe what success looks like for this step
        
        Use the camera feed to understand the current state of the environment and adapt your 
        instructions accordingly. Be specific about object locations, orientations, and the 
        exact motions needed.
        """
    
    def process_step(self, action_step: str, use_camera: bool = True) -> Dict[str, Any]:
        """
        Process a single action step and provide detailed execution instructions.
        
        Args:
            action_step: The action step to process.
            use_camera: Whether to use the camera to capture images.
            
        Returns:
            Dict[str, Any]: The generated instructions and processing details.
        """
        self.logger.info(f"Processing action step: {action_step}")
        
        # Capture images if requested
        image_paths = {}
        if use_camera:
            self.logger.info("Capturing images...")
            camera_tool = self.get_tool("camera")
            if camera_tool:
                frames = self.execute_tool("camera", "capture")
                if frames:
                    image_paths = self.execute_tool("camera", "process", frames=frames)
                    self.logger.info(f"Captured images: {list(image_paths.keys())}")
        
        # Upload images
        image_urls = {}
        if image_paths:
            self.logger.info("Uploading images...")
            image_urls = self.upload_images(image_paths)
            self.logger.info(f"Uploaded image URLs: {list(image_urls.keys())}")
        
        # Call LLM with prompt and images
        self.logger.info("Calling LLM for detailed instructions...")
        detailed_instructions = self.call_llm(action_step, image_urls)
        
        # Store the instructions in memory
        memory_data = {
            "action_step": action_step,
            "detailed_instructions": detailed_instructions,
            "images": list(image_paths.keys())
        }
        
        # Store in both short-term and long-term memory if enabled
        self.memory.store(memory_data, "short_term")
        if self.memory.enable_long_term_memory:
            self.memory.store(memory_data, "long_term")
        
        # Return the result
        return {
            "action_step": action_step,
            "detailed_instructions": detailed_instructions,
            "image_paths": image_paths,
            "image_urls": image_urls
        }
    
    def execute_plan(self, plan_steps: List[Dict[str, Any]], use_camera: bool = True) -> List[Dict[str, Any]]:
        """
        Execute a complete action plan step by step.
        
        Args:
            plan_steps: List of plan steps to execute.
            use_camera: Whether to use the camera for each step.
            
        Returns:
            List[Dict[str, Any]]: Results for each executed step.
        """
        self.logger.info(f"Executing plan with {len(plan_steps)} steps")
        
        results = []
        for i, step in enumerate(plan_steps):
            step_description = step.get("description", "")
            if not step_description:
                self.logger.warning(f"Empty step description for step {i+1}")
                continue
                
            self.logger.info(f"Executing step {i+1}/{len(plan_steps)}: {step_description}")
            
            # Process the current step
            step_result = self.process_step(step_description, use_camera)
            
            # Add step number to the result
            step_result["step_num"] = step.get("step_num", i+1)
            
            # Append to results
            results.append(step_result)
            
            # Log completion
            self.logger.info(f"Step {i+1} execution complete")
        
        self.logger.info("Plan execution complete")
        return results
    
    def verify_step_completion(self, step_description: str, use_camera: bool = True) -> Dict[str, Any]:
        """
        Verify whether a specific step has been completed successfully.
        
        Args:
            step_description: The description of the step to verify.
            use_camera: Whether to use the camera to capture images.
            
        Returns:
            Dict[str, Any]: Verification result with success status and details.
        """
        self.logger.info(f"Verifying completion of step: {step_description}")
        
        # Capture verification images
        image_paths = {}
        if use_camera:
            self.logger.info("Capturing verification images...")
            camera_tool = self.get_tool("camera")
            if camera_tool:
                frames = self.execute_tool("camera", "capture")
                if frames:
                    image_paths = self.execute_tool("camera", "process", frames=frames)
                    self.logger.info(f"Captured verification images: {list(image_paths.keys())}")
        
        # Upload images
        image_urls = {}
        if image_paths:
            self.logger.info("Uploading verification images...")
            image_urls = self.upload_images(image_paths)
        
        # Create verification prompt
        verification_prompt = f"""
        Verify if the following action has been completed successfully:
        
        "{step_description}"
        
        Based on the image(s), determine if the action has been completed correctly.
        Respond with:
        - YES if the action is clearly completed
        - NO if the action is clearly not completed
        - UNCERTAIN if you cannot determine from the image
        
        Follow your response with a brief explanation.
        """
        
        # Call LLM with verification prompt and images
        self.logger.info("Calling LLM for step verification...")
        verification_response = self.call_llm(verification_prompt, image_urls)
        
        # Parse verification result (simple string matching for now)
        is_completed = False
        is_uncertain = True
        
        if "YES" in verification_response.upper():
            is_completed = True
            is_uncertain = False
        elif "NO" in verification_response.upper():
            is_completed = False
            is_uncertain = False
        
        # Store verification result in memory
        memory_data = {
            "step_description": step_description,
            "verification_response": verification_response,
            "is_completed": is_completed,
            "is_uncertain": is_uncertain,
            "images": list(image_paths.keys())
        }
        
        self.memory.store(memory_data, "short_term")
        if self.memory.enable_long_term_memory:
            self.memory.store(memory_data, "long_term")
        
        # Return verification result
        return {
            "step_description": step_description,
            "verification_response": verification_response,
            "is_completed": is_completed,
            "is_uncertain": is_uncertain,
            "image_paths": image_paths,
            "image_urls": image_urls
        } 