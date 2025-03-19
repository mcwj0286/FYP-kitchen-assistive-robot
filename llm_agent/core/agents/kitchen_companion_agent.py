import os
from typing import Dict, List, Any, Optional

from ..base_agent import BaseAgent
from .action_planning_agent import ActionPlanningAgent
from .action_execution_agent import ActionExecutionAgent
from ..tools.camera_tool import CameraTool
from ..parsers.base_parser import PlanParser, DialogParser

class KitchenCompanionAgent(BaseAgent):
    """
    A coordinator agent that manages the workflow between planning and execution agents
    for assisting in kitchen tasks.
    """
    
    def __init__(
        self,
        name: str = "kitchen_companion_agent",
        description: str = "Kitchen companion agent that assists with meal preparation and cooking tasks",
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        server_url: Optional[str] = None,
        enable_short_term_memory: bool = True,
        enable_long_term_memory: bool = True,
        memory_storage_path: Optional[str] = "memory/kitchen_companion",
        camera_interface=None,
        planning_agent: Optional[ActionPlanningAgent] = None,
        execution_agent: Optional[ActionExecutionAgent] = None
    ):
        """
        Initialize the KitchenCompanionAgent.
        
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
            planning_agent: Optional ActionPlanningAgent instance.
            execution_agent: Optional ActionExecutionAgent instance.
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
        
        # Create parsers
        self.dialog_parser = DialogParser()
        
        # Set default companion prompt if none is provided
        if not self.system_prompt:
            self.system_prompt = self._get_default_companion_prompt()
        
        # Initialize sub-agents
        self.planning_agent = planning_agent or ActionPlanningAgent(
            camera_interface=camera_interface,
            server_url=server_url,
            model_name=model_name
        )
        
        self.execution_agent = execution_agent or ActionExecutionAgent(
            camera_interface=camera_interface,
            server_url=server_url,
            model_name=model_name
        )
    
    def _get_default_companion_prompt(self) -> str:
        """
        Get the default companion prompt for the agent.
        
        Returns:
            str: The default companion prompt.
        """
        return """
        You are a helpful kitchen companion robot assistant. Your role is to help users with meal 
        preparation, cooking, and other kitchen tasks. You can provide guidance, answer questions,
        and assist with practical tasks using your robotic capabilities.
        
        When interacting with users, be friendly, supportive, and focused on being helpful.
        
        If a user asks for help with a specific cooking task or recipe, you should:
        1. Acknowledge their request positively
        2. Ask clarifying questions if needed
        3. Consider the best way to assist (instructions, guidance, or physical help)
        4. Provide clear, concise information
        
        For any task that requires physical action, break it down into clear steps.
        
        Remember:
        - Safety is the top priority
        - Be conversational but efficient
        - Be encouraging and supportive
        - Focus on practical solutions
        """
    
    def process_user_request(self, user_request: str, use_camera: bool = True) -> Dict[str, Any]:
        """
        Process a user request by determining if it requires planning, execution, or just conversation.
        
        Args:
            user_request: The user's request or question.
            use_camera: Whether to use the camera to capture images.
            
        Returns:
            Dict[str, Any]: The response to the user's request.
        """
        self.logger.info(f"Processing user request: {user_request}")
        
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
        
        # First, check if this is a task that requires planning and execution
        task_classification_prompt = f"""
        I need to classify the following user request in a kitchen context:
        
        "{user_request}"
        
        Please classify this request into one of these categories:
        1. TASK - If it's a physical task that requires creating a plan and taking actions (e.g., "help me prepare coffee", "can you assist with cutting vegetables")
        2. QUESTION - If it's an informational question that just needs a verbal answer (e.g., "how do I cook pasta", "what's the best way to store leftovers")
        3. CONVERSATION - If it's just casual conversation or a greeting (e.g., "hello", "how are you", "thank you")
        
        Respond with JUST ONE of these category names: TASK, QUESTION, or CONVERSATION.
        """
        
        # Call LLM to classify the request
        self.logger.info("Classifying user request...")
        classification = self.call_llm(task_classification_prompt, {})
        classification = classification.strip().upper()
        
        # Store the request in memory
        memory_data = {
            "user_request": user_request,
            "request_type": classification,
            "images": list(image_paths.keys())
        }
        
        self.memory.store(memory_data, "short_term")
        if self.memory.enable_long_term_memory:
            self.memory.store(memory_data, "long_term")
        
        # Process based on classification
        if "TASK" in classification:
            return self._handle_task_request(user_request, image_paths, image_urls)
        elif "QUESTION" in classification:
            return self._handle_question(user_request, image_paths, image_urls)
        else:  # CONVERSATION
            return self._handle_conversation(user_request, image_paths, image_urls)
    
    def _handle_task_request(self, user_request: str, image_paths: Dict[str, str], image_urls: Dict[str, str]) -> Dict[str, Any]:
        """
        Handle a request that requires planning and execution.
        
        Args:
            user_request: The user's request.
            image_paths: Dictionary of image paths.
            image_urls: Dictionary of image URLs.
            
        Returns:
            Dict[str, Any]: The response to the user's request.
        """
        self.logger.info("Handling as a TASK request")
        
        # First, generate a plan
        plan_result = self.planning_agent.process(user_request, use_camera=True)
        
        # Check if plan was generated successfully
        if not plan_result.get("steps"):
            # If planning failed, provide a conversational response about the issue
            response_prompt = f"""
            I tried to help with:
            
            "{user_request}"
            
            But I couldn't create a clear plan. Please explain to the user in a conversational way 
            why this might be difficult, and ask for more specific instructions or context.
            Keep your response friendly and helpful. Suggest alternatives if appropriate.
            """
            
            response = self.call_llm(response_prompt, image_urls)
            return {
                "request_type": "TASK",
                "success": False,
                "response": response,
                "image_paths": image_paths,
                "image_urls": image_urls,
                "plan": None,
                "execution_results": None
            }
        
        # We have a plan, now execute it
        execution_results = self.execution_agent.execute_plan(plan_result.get("steps", []))
        
        # Generate a summary of what was done
        plan_steps_text = "\n".join([f"{step.get('step_num', i+1)}. {step.get('description', '')}" 
                                   for i, step in enumerate(plan_result.get("steps", []))])
        
        summary_prompt = f"""
        I've helped with the following request:
        
        "{user_request}"
        
        By following this plan:
        {plan_steps_text}
        
        Please provide a friendly, conversational summary of what was done. Keep it concise but informative,
        highlighting what was accomplished and any important points about the process.
        """
        
        summary = self.call_llm(summary_prompt, {})
        
        return {
            "request_type": "TASK",
            "success": True,
            "response": summary,
            "image_paths": image_paths,
            "image_urls": image_urls,
            "plan": plan_result,
            "execution_results": execution_results
        }
    
    def _handle_question(self, user_request: str, image_paths: Dict[str, str], image_urls: Dict[str, str]) -> Dict[str, Any]:
        """
        Handle an informational question.
        
        Args:
            user_request: The user's question.
            image_paths: Dictionary of image paths.
            image_urls: Dictionary of image URLs.
            
        Returns:
            Dict[str, Any]: The response to the user's question.
        """
        self.logger.info("Handling as a QUESTION request")
        
        # Generate a helpful response to the question
        question_prompt = f"""
        Please provide a helpful, informative answer to this kitchen-related question:
        
        "{user_request}"
        
        Your answer should be:
        - Accurate and factual
        - Concise but thorough
        - Conversational and friendly
        - Practical and actionable
        
        If the question relates to what you can see in the provided image, refer to relevant details.
        """
        
        response = self.call_llm(question_prompt, image_urls)
        
        return {
            "request_type": "QUESTION",
            "response": response,
            "image_paths": image_paths,
            "image_urls": image_urls
        }
    
    def _handle_conversation(self, user_message: str, image_paths: Dict[str, str], image_urls: Dict[str, str]) -> Dict[str, Any]:
        """
        Handle casual conversation.
        
        Args:
            user_message: The user's message.
            image_paths: Dictionary of image paths.
            image_urls: Dictionary of image URLs.
            
        Returns:
            Dict[str, Any]: The response to the user's message.
        """
        self.logger.info("Handling as a CONVERSATION request")
        
        # Get recent conversation history
        conversation_history = self.memory.retrieve(memory_type="short_term", limit=5)
        history_text = ""
        
        for item in conversation_history:
            data = item.get("data", {})
            if "user_request" in data and "response" in data:
                history_text += f"User: {data['user_request']}\nAssistant: {data['response']}\n\n"
        
        # Generate a conversational response
        conversation_prompt = f"""
        {history_text}
        User: {user_message}
        
        Please respond to the user in a friendly, helpful way. Keep your response conversational and natural.
        If the user's message relates to the kitchen or cooking, try to be helpful while staying on topic.
        """
        
        response = self.call_llm(conversation_prompt, image_urls)
        
        # Parse and store the dialog
        dialog_data = {
            "user_request": user_message,
            "response": response
        }
        
        self.memory.store(dialog_data, "short_term")
        if self.memory.enable_long_term_memory:
            self.memory.store(dialog_data, "long_term")
        
        return {
            "request_type": "CONVERSATION",
            "response": response,
            "image_paths": image_paths,
            "image_urls": image_urls
        }
    
    def get_recipe_suggestions(self, ingredients: List[str], dietary_restrictions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get recipe suggestions based on available ingredients and dietary restrictions.
        
        Args:
            ingredients: List of available ingredients.
            dietary_restrictions: Optional list of dietary restrictions.
            
        Returns:
            List[Dict[str, Any]]: List of recipe suggestions.
        """
        # Format ingredients and restrictions for the prompt
        ingredients_text = ", ".join(ingredients)
        restrictions_text = ", ".join(dietary_restrictions) if dietary_restrictions else "None"
        
        # Create the recipe suggestion prompt
        suggestion_prompt = f"""
        Please suggest 3 recipes that can be made with some or all of these ingredients:
        {ingredients_text}
        
        Dietary restrictions: {restrictions_text}
        
        For each recipe, provide:
        1. A name
        2. A brief description (1-2 sentences)
        3. A list of ingredients needed (indicating which ones are from the provided list)
        4. Approximate preparation time
        
        Format each recipe clearly and number them 1-3.
        """
        
        # Call LLM to get suggestions
        response = self.call_llm(suggestion_prompt, {})
        
        # For now, return the raw response - in a real implementation, you would parse this into structured data
        return [{"suggestions": response}]
    
    def get_meal_plan(self, days: int = 7, dietary_preferences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a meal plan for a specified number of days.
        
        Args:
            days: Number of days to plan for.
            dietary_preferences: Optional list of dietary preferences.
            
        Returns:
            Dict[str, Any]: The generated meal plan.
        """
        # Format preferences for the prompt
        preferences_text = ", ".join(dietary_preferences) if dietary_preferences else "None"
        
        # Create the meal plan prompt
        plan_prompt = f"""
        Please create a {days}-day meal plan with the following parameters:
        
        Dietary preferences: {preferences_text}
        Meals per day: 3 (breakfast, lunch, dinner)
        
        For each day and meal, provide:
        1. Name of the dish
        2. A very brief description (1 sentence)
        3. Key ingredients (3-5 main items)
        
        Format the plan clearly by day and meal.
        """
        
        # Call LLM to get meal plan
        response = self.call_llm(plan_prompt, {})
        
        # For now, return the raw response - in a real implementation, you would parse this into structured data
        return {"meal_plan": response, "days": days, "dietary_preferences": dietary_preferences}
    
    def set_companion_prompt(self, new_prompt: str) -> None:
        """
        Set a new companion prompt.
        
        Args:
            new_prompt: The new prompt to use.
        """
        self.set_system_prompt(new_prompt)
        self.logger.info("Companion prompt updated.") 