#!/usr/bin/env python3

import os
import time
import yaml
import logging
from typing import Dict, List, Tuple, Optional, Any
import cv2

logger = logging.getLogger("llm_agent")

class ActionPlanExecutor:
    """
    An agent that executes predefined action plans from YAML files.
    It coordinates robot arm movements, camera captures, and speech
    based on the steps in the action plan.
    """
    
    def __init__(self, agent, tool_manager):
        """
        Initialize the ActionPlanExecutor.
        
        Args:
            agent: The AI agent with tool capabilities
            tool_manager: Tool manager for hardware tools
        """
        self.agent = agent
        self.tool_manager = tool_manager
        
        # Load plans and locations
        self.plans = self.load_action_plans()
        self.predefined_locations = self.load_predefined_locations()
        
        # Current plan state
        self.current_plan = None
        self.current_step_index = 0
        self.execution_history = []
    
    def load_predefined_locations(self) -> Dict[str, List[str]]:
        """
        Load predefined locations from YAML file.
        
        Returns:
            Dict[str, List[str]]: Dictionary of predefined locations
        """
        locations_file = os.path.join(os.path.dirname(__file__), 'actions_config', 'predefined_loaction.yaml')
        if not os.path.exists(locations_file):
            logger.warning(f"Predefined locations file not found: {locations_file}")
            return {}
            
        try:
            with open(locations_file, 'r') as file:
                self.predefined_locations = yaml.safe_load(file) or {}
            return self.predefined_locations
        except Exception as e:
            logger.error(f"Error loading predefined locations: {e}")
            return {}
    
    def load_action_plans(self) -> Dict[str, Any]:
        """Load all action plans from YAML file."""
        try:
            with open("llm_agent/actions_config/action_plan.yaml", "r") as f:
                plans = yaml.safe_load(f)
            return plans
        except Exception as e:
            logger.error(f"Error loading action plans: {e}")
            return {}
    
    def list_available_plans(self) -> List[str]:
        """List all available action plans."""
        return list(self.plans.keys())
    
    def load_plan(self, plan_name: str) -> bool:
        """
        Load a specific action plan by name.
        
        Args:
            plan_name: Name of the action plan to load
            
        Returns:
            bool: True if plan was loaded successfully, False otherwise
        """
        if plan_name not in self.plans:
            logger.error(f"Action plan '{plan_name}' not found")
            return False
        
        self.current_plan = self.plans[plan_name]
        self.current_step_index = 0
        self.execution_history = []
        logger.info(f"Loaded action plan: {plan_name}")
        logger.info(f"Goal: {self.current_plan['goal']}")
        return True
    
    def get_current_step(self) -> Optional[Dict[str, Any]]:
        """Get the current step in the action plan."""
        if not self.current_plan:
            return None
        
        steps = self.current_plan.get('steps', [])
        if not steps or self.current_step_index >= len(steps):
            return None
            
        return steps[self.current_step_index]
    
    def execute_current_step(self, environment_image: Optional[str] = None) -> Tuple[bool, str]:
        """
        Execute the current step in the action plan.
        
        Args:
            environment_image: Optional path to an environment image for context
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        current_step = self.get_current_step()
        if not current_step:
            return False, "No steps to execute or plan not loaded"
        
        step_num = current_step['step_num']
        description = current_step['description']
        
        logger.info(f"Executing step {step_num}: {description}")
        
        # Capture environment image if not provided
        if not environment_image and self.tool_manager.camera_tool:
            result = self.tool_manager.camera_tool("capture")
            if "Successfully captured" not in result:
                logger.warning(f"Failed to capture environment image: {result}")
        
        # Execute step based on description content
        success, result = False, "Step execution not implemented"
        
        # Step 1: Move to open jar position
        if "move to open jar position" in description.lower():
            success, result = self._execute_move_to_location("open_jar_position")
        
        # Step 2: Announce user to put the jar on the gripper
        elif "announce" in description.lower() and "jar" in description.lower():
            message = "Please put the jar on the gripper."
            success, result = self._execute_announcement(message)
        
        # Step 3: Wait for user to put the jar on the gripper
        elif "wait for user" in description.lower() and "jar" in description.lower():
            success, result = self._wait_for_jar_placement()
        
        # Step 4: Close the gripper
        elif "close the gripper" in description.lower():
            success, result = self._execute_gripper_action("close")
        
        # Step 5: Roll left to open the jar
        elif "roll left" in description.lower() and "open the jar" in description.lower():
            success, result = self._execute_gripper_action("roll_left")
        
        # Log the result
        if success:
            logger.info(f"Step {step_num} executed successfully: {result}")
        else:
            logger.error(f"Step {step_num} execution failed: {result}")
        
        # Record in history
        self.execution_history.append({
            "step_num": step_num,
            "description": description,
            "success": success,
            "result": result,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Move to next step if successful
        if success:
            self.current_step_index += 1
        
        return success, result
    
    def _execute_move_to_location(self, location_name: str) -> Tuple[bool, str]:
        """
        Move the robot arm to a predefined location.
        
        Args:
            location_name (str): The name of the location in the predefined locations file
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        if not self.tool_manager.robot_arm_tool:
            return False, "Robot arm tool not available"
            
        # Get the location coordinates
        location = self.predefined_locations.get(location_name)
        if not location:
            return False, f"Location '{location_name}' not found in predefined locations"
            
        logger.info(f"Moving to location: {location_name} - {location}")
        
        # Capture initial image for reference
        self.tool_manager.camera_tool("capture")
        
        # Determine if the location has a gripper position
        if len(location) >= 7:
            position = location[:6]
            gripper_position = location[6]
        else:
            position = location
            gripper_position = None
        
        # Convert all position values to float (they may be strings from YAML)
        position = [float(x) for x in position]
        
        # Move to position
        position_cmd = f"move_position {' '.join(map(str, position))}"
        response = self.tool_manager.robot_arm_tool(position_cmd)
        success = "successfully" in response.lower()
        
        if not success:
            return False, f"Failed to move to position: {response}"
            
        # Set gripper position if available
        if gripper_position is not None:
            try:
                # Convert gripper position to int via float (it may be a string from YAML)
                gripper_position = int(float(gripper_position))
                gripper_cmd = f"gripper position {gripper_position}"
                gripper_response = self.tool_manager.robot_arm_tool(gripper_cmd)
                gripper_success = "successfully" in gripper_response.lower() or "position set" in gripper_response.lower()
                
                if not gripper_success:
                    return False, f"Failed to set gripper position: {gripper_response}"
            except Exception as e:
                return False, f"Failed to set gripper position: {str(e)}"
        
        return True, f"Successfully moved to location: {location_name}"
    
    def _execute_announcement(self, message: str) -> Tuple[bool, str]:
        """
        Make an announcement using the speaker.
        
        Args:
            message: The message to announce
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        if not self.tool_manager.speaker_tool:
            return False, "Speaker tool not available"
        
        result = self.tool_manager.speaker_tool(f"speak {message}")
        
        # Check if successful
        if "error" in result.lower():
            return False, f"Failed to make announcement: {result}"
            
        return True, f"Announcement made: {message}"
    
    def _wait_for_jar_placement(self, timeout: int = 60, check_interval: int = 3) -> Tuple[bool, str]:
        """
        Wait for the user to place the jar on the gripper.
        Uses both image analysis and optional user confirmation.
        
        Args:
            timeout: Maximum time to wait in seconds
            check_interval: How often to check in seconds
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        if not self.tool_manager.camera_tool:
            return False, "Camera tool not available"
        
        # Announce instructions
        if self.tool_manager.speaker_tool:
            self.tool_manager.speaker_tool("speak Please place the jar on the gripper and say ready when done.")
        
        start_time = time.time()
        
        # Check if we're running in mock mode
        is_mock_mode = os.getenv("MOCK_CAMERA_SCENARIO") is not None
        
        # Wait loop
        while time.time() - start_time < timeout:
            # Capture image
            capture_result = self.tool_manager.camera_tool("capture")
            if "Successfully captured" not in capture_result:
                logger.warning(f"Failed to capture image: {capture_result}")
                time.sleep(check_interval)
                continue
            
            # If we're in mock mode, can optionally ask for user confirmation to advance mock scenario
            if is_mock_mode:
                print("\n🔧 MOCK MODE: Would you like to simulate the jar being placed on the gripper?")
                user_input = input("Type 'y' to simulate jar placement or anything else to continue waiting: ")
                if user_input.lower() == 'y':
                    # Change the mock scenario to show jar on gripper
                    os.environ["MOCK_CAMERA_SCENARIO"] = "jar_on_gripper"
                    logger.info("Mock scenario changed to 'jar_on_gripper'")
                    print("🔧 Mock scenario changed to show jar on gripper")
                    
                    # Take a new capture with the updated scenario
                    self.tool_manager.camera_tool("capture")
                    return True, "Jar detected on gripper (mock simulation)"
            
            # Analyze image with LLM
            analysis = self.agent.analyze_captured_image(
                camera_tool=self.tool_manager.camera_tool,
                prompt="Is there a jar placed on the robot gripper? Answer only 'yes' or 'no' and explain briefly why.",
                max_tokens=100
            )
            
            logger.info(f"Jar detection analysis: {analysis}")
            
            # Check if jar is detected
            if "yes" in analysis.lower():
                # If in mock mode, make sure to update the scenario
                if is_mock_mode:
                    os.environ["MOCK_CAMERA_SCENARIO"] = "jar_on_gripper"
                return True, "Jar detected on gripper"
            
            # Wait before checking again
            time.sleep(check_interval)
        
        return False, f"Timeout waiting for jar placement after {timeout} seconds"
    
    def _execute_gripper_action(self, action: str) -> Tuple[bool, str]:
        """
        Execute a gripper action.
        
        Args:
            action (str): The action to execute (open, close, etc.)
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        if not self.tool_manager.robot_arm_tool:
            return False, "Robot arm tool not available"
            
        # Check if we're in mock mode
        is_mock_mode = os.getenv("MOCK_CAMERA_SCENARIO") is not None
        
        # Handle different actions
        if "close" in action.lower():
            # Close the gripper
            response = self.tool_manager.robot_arm_tool("gripper close")
            success = "successfully" in response.lower()
            
            # Announce the action
            if success and self.tool_manager.speaker_tool:
                self.tool_manager.speaker_tool("speak Gripper closed. Holding jar firmly.")
                
            return success, response
            
        elif "open" in action.lower():
            # Open the gripper
            response = self.tool_manager.robot_arm_tool("gripper open")
            success = "successfully" in response.lower()
            
            # Announce the action
            if success and self.tool_manager.speaker_tool:
                self.tool_manager.speaker_tool("speak Gripper opened.")
                
            return success, response
            
        elif "roll left" in action.lower() or "roll" in action.lower() or "twist" in action.lower():
            # This is a special action for opening the jar by rolling the arm
            
            # Get current joints
            joint_response = self.tool_manager.robot_arm_tool("get_joints")
            logger.info(f"Current joints before roll: {joint_response}")
            
            # Execute roll movement for opening jar
            # Here we would typically move a specific joint to twist the jar
            # For now, we'll simply use a preset movement
            
            # Announce start of twist
            if self.tool_manager.speaker_tool:
                self.tool_manager.speaker_tool("speak Opening the jar now. Hold steady please.")
            
            # Start with a slight movement (adjust based on robot capabilities)
            response1 = self.tool_manager.robot_arm_tool("move_joint 0.0 0.0 0.0 0.0 0.0 -30.0")
            success1 = "successfully" in response1.lower()
            
            if not success1:
                return False, f"Failed to execute first twist: {response1}"
                
            # Short pause
            time.sleep(1)
            
            # More pronounced twist
            response2 = self.tool_manager.robot_arm_tool("move_joint 0.0 0.0 0.0 0.0 0.0 -60.0")
            success2 = "successfully" in response2.lower()
            
            if not success2:
                return False, f"Failed to execute second twist: {response2}"
                
            # If in mock mode, update the scenario to show jar opened
            if is_mock_mode:
                os.environ["MOCK_CAMERA_SCENARIO"] = "jar_opened"
                logger.info("Mock scenario changed to 'jar_opened'")
                print("🔧 Mock scenario changed to show jar opened")
                
                if self.tool_manager.camera_tool:
                    # Take a new capture with the updated scenario
                    self.tool_manager.camera_tool("capture")
            
            # Announce success
            if self.tool_manager.speaker_tool:
                self.tool_manager.speaker_tool("speak Jar opened successfully!")
                
            return True, "Successfully executed roll left movement to open jar"
            
        else:
            return False, f"Unknown gripper action: {action}"
    
    def execute_plan(self, plan_name: str) -> List[Dict[str, Any]]:
        """
        Execute an entire action plan from start to finish.
        
        Args:
            plan_name: Name of the action plan to execute
            
        Returns:
            List[Dict[str, Any]]: Execution history with results for each step
        """
        if not self.load_plan(plan_name):
            return self.execution_history
        
        logger.info(f"Starting execution of plan: {plan_name}")
        logger.info(f"Goal: {self.current_plan['goal']}")
        
        # Execute each step
        while self.get_current_step() is not None:
            current_step = self.get_current_step()
            step_num = current_step['step_num']
            description = current_step['description']
            
            print(f"\nExecuting Step {step_num}: {description}")
            
            success, result = self.execute_current_step()
            
            if success:
                print(f"✅ Success: {result}")
                # Small pause between steps
                time.sleep(2)
            else:
                print(f"❌ Failed: {result}")
                # Ask whether to retry, skip, or abort
                response = input("Retry (r), Skip (s), or Abort (a)? ").lower()
                if response == 'r':
                    # Retry the same step
                    continue
                elif response == 's':
                    # Skip to next step
                    self.current_step_index += 1
                else:
                    # Abort plan execution
                    print("Aborting plan execution.")
                    break
        
        # Summarize execution
        if self.current_step_index >= len(self.current_plan.get('steps', [])):
            print(f"\n✅ Plan '{plan_name}' executed successfully!")
        else:
            print(f"\n⚠️ Plan '{plan_name}' execution incomplete.")
        
        return self.execution_history
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get the current execution status."""
        total_steps = len(self.current_plan.get('steps', [])) if self.current_plan else 0
        completed_steps = self.current_step_index
        
        return {
            "plan_name": self.current_plan.get('goal', "No plan loaded") if self.current_plan else "No plan loaded",
            "current_step": self.current_step_index + 1 if self.current_plan else 0,
            "total_steps": total_steps,
            "progress": f"{completed_steps}/{total_steps}",
            "completed": completed_steps >= total_steps,
            "history": self.execution_history
        }


# For testing the module directly
if __name__ == "__main__":
    from dotenv import load_dotenv
    from ai_agent import BaseAgent
    from hardware_tools import HardwareToolManager
    
    # Load environment variables
    load_dotenv()
    
    # Initialize components
    tool_manager = HardwareToolManager()
    agent = BaseAgent(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model_name=os.getenv("MODEL_NAME", "anthropic/claude-3-opus-20240229"),
    )
    
    # Add tools to agent
    tools = tool_manager.get_all_tools()
    for tool in tools:
        agent.add_tool(tool)
    
    # Create executor
    executor = ActionPlanExecutor(agent, tool_manager)
    
    # List available plans
    plans = executor.list_available_plans()
    print(f"Available plans: {plans}")
    
    if plans:
        # Execute first available plan
        executor.execute_plan(plans[0]) 