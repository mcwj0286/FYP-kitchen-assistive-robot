import os
from typing import Dict, List, Any, Optional

from ..base_agent import BaseAgent
from ..tools.camera_tool import CameraTool
from ..parsers.base_parser import PlanParser

class ActionPlanningAgent(BaseAgent):
    """
    An agent that generates a detailed action plan based on user prompts and camera images.
    """
    
    def __init__(
        self,
        name: str = "action_planning_agent",
        description: str = "Generates detailed action plans based on user prompts and images",
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        server_url: Optional[str] = None,
        enable_short_term_memory: bool = True,
        enable_long_term_memory: bool = True,
        memory_storage_path: Optional[str] = "memory/action_planning",
        camera_interface=None
    ):
        """
        Initialize the ActionPlanningAgent.
        
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
        
        # Create parser for plan responses
        self.plan_parser = PlanParser()
        
        # Set default planning prompt if none is provided
        if not self.system_prompt:
            self.system_prompt = self._get_default_planning_prompt()
    
    def _get_default_planning_prompt(self) -> str:
        """
        Get the default planning prompt for the agent.
        
        Returns:
            str: The default planning prompt.
        """
        return """
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
    
    def process(self, user_goal: str, use_camera: bool = True) -> Dict[str, Any]:
        """
        Generate an action plan based on the user goal and camera images.
        
        Args:
            user_goal: The goal specified by the user.
            use_camera: Whether to use the camera to capture images.
            
        Returns:
            Dict[str, Any]: The generated plan and processing details.
        """
        self.logger.info(f"Processing goal: {user_goal}")
        
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
        self.logger.info("Calling LLM...")
        llm_response = self.call_llm(user_goal, image_urls)
        
        # Parse the plan
        parsed_plan = self.plan_parser.parse(llm_response)
        
        # Store the plan in memory
        memory_data = {
            "goal": user_goal,
            "raw_plan": llm_response,
            "steps": parsed_plan.get("steps", [])
        }
        
        # Store in both short-term and long-term memory if enabled
        self.memory.store(memory_data, "short_term")
        if self.memory.enable_long_term_memory:
            self.memory.store(memory_data, "long_term")
        
        # Return the result
        return {
            "goal": user_goal,
            "raw_plan": llm_response,
            "steps": parsed_plan.get("steps", []),
            "error": parsed_plan.get("error"),
            "image_paths": image_paths,
            "image_urls": image_urls
        }
    
    def get_similar_plans(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get similar plans from memory based on a query.
        
        Args:
            query: The query to search for.
            limit: The maximum number of results to return.
            
        Returns:
            List[Dict[str, Any]]: List of similar plans.
        """
        if not self.memory.enable_long_term_memory:
            self.logger.warning("Long-term memory is not enabled. Cannot retrieve similar plans.")
            return []
        
        # For now, we'll just do a simple search for goals containing any word from the query
        query_words = set(query.lower().split())
        results = []
        
        # Get all plans from long-term memory
        plans = self.memory.retrieve(memory_type="long_term", limit=100)
        
        # Filter plans based on query words
        for plan_entry in plans:
            plan_data = plan_entry.get("data", {})
            goal = plan_data.get("goal", "").lower()
            
            # Check if any query word appears in the goal
            if any(word in goal for word in query_words):
                results.append(plan_data)
                
                # Stop when we reach the limit
                if len(results) >= limit:
                    break
        
        return results
    
    def set_planning_prompt(self, new_prompt: str) -> None:
        """
        Set a new planning prompt.
        
        Args:
            new_prompt (str): The new prompt to use.
        """
        self.set_system_prompt(new_prompt)
        self.logger.info("Planning prompt updated.") 