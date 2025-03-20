#!/usr/bin/env python3

import os
import time
import yaml
import logging
import threading
import re
import json
from typing import Dict, List, Tuple, Optional, Any
import cv2

from llm_agent.monitoring_agent import MonitoringAgent

logger = logging.getLogger("llm_agent")

class ActionPlanExecutor:
    """
    An agent that executes predefined action plans from YAML files.
    It uses AI to interpret and execute steps dynamically based on their descriptions.
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
        
        # Initialize the monitoring agent
        self.monitoring_agent = MonitoringAgent(agent, tool_manager.camera_tool)
        
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
    
    def interpret_step(self, step_description: str) -> Dict[str, Any]:
        """
        Use the AI agent to interpret a step description and determine how to execute it.
        
        Args:
            step_description: The description of the step to interpret
            
        Returns:
            Dict with instructions for executing the step
        """
        # Special case for step 1 which is consistently problematic
        current_step = self.get_current_step()
        if current_step and current_step.get('step_num') == 1 and "move to open jar position" in step_description.lower():
            logger.info("Using special fallback for step 1 (move to open jar position)")
            return {
                "action_type": "movement",
                "location_name": "open_jar_position",  # Use exact location name as in YAML file
                "parameters": ["open_jar_position"]
            }
            
        # Capture current environment to provide context
        if self.tool_manager.camera_tool:
            self.tool_manager.camera_tool("capture")
        
        # Create a prompt for the AI to interpret the step
        prompt = f"""
        As an AI controlling a kitchen robot, interpret the following action step and provide 
        execution instructions in a structured format:
        
        Step description: "{step_description}"
        
        Based on this description, determine:
        1. What type of action is needed (movement, announcement, waiting for condition, gripper action, etc.)
        2. The specific parameters or context needed to execute this action
        
        Return your interpretation in the following JSON format without any markdown formatting or backticks:
        
        {{
            "ACTION_TYPE": "movement|announcement|wait_for_condition|gripper_action|other",
            "PARAMETERS": ["specific parameters needed for this action"],
            "DETECTION_CRITERIA": "if waiting for a condition, describe what to detect",
            "LOCATION_NAME": "if movement, the name of the location",
            "MESSAGE": "if announcement, the message to speak",
            "GRIPPER_ACTION": "if gripper action, what to do: open, close, roll_left, etc.",
            "SYSTEM_PROMPT": "if waiting for a condition, specialized prompt for the monitoring agent",
            "TIMEOUT": "if waiting for a condition, how long to wait in seconds"
        }}
        
        IMPORTANT: For location names, use underscores instead of spaces (for example, use "open_jar_position" not "open jar position").
        
        Only include fields that are relevant to the action type. Return ONLY the JSON object without any markdown formatting or backticks.
        """
        
        # Ask the AI to interpret the step
        interpretation = self.agent.process(
            prompt=prompt,
            system_prompt="You are a helpful assistant specialized in robotics task planning. When asked to return JSON, provide only the raw JSON without any markdown formatting or backticks.",
            max_tokens=500,
            use_tools=False
        )
        
        logger.info(f"Raw interpretation from LLM: {interpretation}")
        
        # Parse the JSON response
        try:
            # Clean up the response to ensure it's valid JSON
            interpretation = interpretation.strip()
            if interpretation.startswith('```json'):
                interpretation = interpretation[7:]
            elif interpretation.startswith('```'):
                interpretation = interpretation[3:]
            if interpretation.endswith('```'):
                interpretation = interpretation[:-3]
            
            # Special fallback for very short responses that are likely erroneous
            if len(interpretation) < 10:
                # Try to determine action type from step description
                if "move to" in step_description.lower() or "move the" in step_description.lower():
                    location_name = step_description.lower().replace("move to", "").replace("move the", "").strip()
                    location_name = location_name.replace(" ", "_")  # Convert spaces to underscores
                    logger.info(f"LLM response too short, using fallback with location: {location_name}")
                    return {
                        "action_type": "movement",
                        "location_name": location_name,
                        "parameters": [location_name]
                    }
                elif "announce" in step_description.lower() or "speak" in step_description.lower():
                    message = step_description.lower().replace("announce", "").replace("speak", "").strip()
                    logger.info(f"LLM response too short, using fallback with message: {message}")
                    return {
                        "action_type": "announcement",
                        "message": message
                    }
                elif "wait" in step_description.lower():
                    criteria = step_description.lower().replace("wait for", "").replace("wait until", "").strip()
                    logger.info(f"LLM response too short, using fallback with criteria: {criteria}")
                    return {
                        "action_type": "wait_for_condition",
                        "detection_criteria": criteria,
                        "system_prompt": f"You are monitoring a kitchen environment. Check if: {criteria}",
                        "timeout": 60
                    }
                elif "gripper" in step_description.lower():
                    action = "open" if "open" in step_description.lower() else "close"
                    if "roll" in step_description.lower() or "twist" in step_description.lower():
                        action = "roll_left" if "left" in step_description.lower() else "roll_right"
                    logger.info(f"LLM response too short, using fallback with gripper action: {action}")
                    return {
                        "action_type": "gripper_action",
                        "gripper_action": action
                    }
            
            # Parse the JSON
            result = json.loads(interpretation)
            logger.info(f"Parsed JSON result: {result}")
            
            # Convert keys to lowercase for consistency
            result = {k.lower(): v for k, v in result.items()}
            
            # Ensure action_type is present and lowercase
            if "action_type" not in result:
                logger.error("No action_type found in parsed result")
                return {"action_type": "unknown"}
            
            # Normalize location name for movements (replace spaces with underscores)
            if result["action_type"].lower() == "movement" and "location_name" in result:
                original_name = result["location_name"]
                result["location_name"] = original_name.replace(" ", "_")
                if original_name != result["location_name"]:
                    logger.info(f"Normalized location name from '{original_name}' to '{result['location_name']}'")
                    
                    # Also update the parameters if they contain the location
                    if "parameters" in result and isinstance(result["parameters"], list):
                        for i, param in enumerate(result["parameters"]):
                            if isinstance(param, str) and param == original_name:
                                result["parameters"][i] = result["location_name"]
            
            result["action_type"] = result["action_type"].lower()
            logger.info(f"Final interpreted result: {result}")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {interpretation}")
            
            # Fallback pattern matching
            current_step_num = current_step.get('step_num') if current_step else None
            logger.info(f"Attempting fallback for step {current_step_num}: {step_description}")
            
            # Try to determine action type from step description
            if "move to" in step_description.lower() or "move the" in step_description.lower():
                location_name = step_description.lower().replace("move to", "").replace("move the", "").strip()
                # Convert spaces to underscores in location name
                location_name = location_name.replace(" ", "_")
                logger.info(f"Using fallback with location: {location_name}")
                return {
                    "action_type": "movement",
                    "location_name": location_name,
                    "parameters": [location_name]
                }
            elif "announce" in step_description.lower() or "speak" in step_description.lower():
                message = step_description.lower().replace("announce", "").replace("speak", "").strip()
                logger.info(f"Using fallback with message: {message}")
                return {
                    "action_type": "announcement",
                    "message": message
                }
            elif "wait for" in step_description.lower():
                criteria = step_description.lower().replace("wait for", "").replace("wait until", "").strip()
                logger.info(f"Using fallback with criteria: {criteria}")
                return {
                    "action_type": "wait_for_condition",
                    "detection_criteria": criteria,
                    "system_prompt": f"You are monitoring a kitchen environment. Check if: {criteria}",
                    "timeout": 60
                }
            elif "gripper" in step_description.lower():
                action = "open" if "open" in step_description.lower() else "close"
                if "roll" in step_description.lower() or "twist" in step_description.lower():
                    action = "roll_left" if "left" in step_description.lower() else "roll_right"
                logger.info(f"Using fallback with gripper action: {action}")
                return {
                    "action_type": "gripper_action",
                    "gripper_action": action
                }
            
            return {"action_type": "unknown"}
            
        except Exception as e:
            logger.error(f"Unexpected error in interpret_step: {e}")
            return {"action_type": "unknown"}
    
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
        
        # Use AI to interpret the step
        interpretation = self.interpret_step(description)
        action_type = interpretation.get("action_type", "unknown")
        
        logger.info(f"Step {step_num} interpreted as action type: {action_type}")
        
        # Execute based on the interpreted action type
        success, result = False, "Action type not supported"
        
        if action_type == "movement":
            location_name = interpretation.get("location_name")
            if location_name:
                success, result = self._execute_movement(location_name)
            else:
                result = "Movement action missing location name"
                
        elif action_type == "announcement":
            message = interpretation.get("message")
            if message:
                success, result = self._execute_announcement(message)
            else:
                result = "Announcement action missing message"
                
        elif action_type == "wait_for_condition":
            criteria = interpretation.get("detection_criteria")
            system_prompt = interpretation.get("system_prompt")
            timeout = interpretation.get("timeout", 60)
            
            if criteria and system_prompt:
                success, result = self._wait_for_condition(
                    detection_criteria=criteria,
                    system_prompt=system_prompt,
                    timeout=timeout
                )
            else:
                result = "Wait for condition action missing criteria or system prompt"
                
        elif action_type == "gripper_action":
            gripper_action = interpretation.get("gripper_action")
            if gripper_action:
                success, result = self._execute_gripper_action(gripper_action)
            else:
                result = "Gripper action missing specific action"
        
        else:
            result = f"Unsupported action type: {action_type}"
        
        # Log the result
        if success:
            logger.info(f"Step {step_num} executed successfully: {result}")
        else:
            logger.error(f"Step {step_num} execution failed: {result}")
        
        # Record in history
        self.execution_history.append({
            "step_num": step_num,
            "description": description,
            "action_type": action_type,
            "success": success,
            "result": result,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Move to next step if successful
        if success:
            self.current_step_index += 1
        
        return success, result
    
    def _execute_movement(self, location_name: str) -> Tuple[bool, str]:
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
        if self.tool_manager.camera_tool:
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
    
    def _wait_for_condition(self, detection_criteria: str, system_prompt: str, timeout: int = 60) -> Tuple[bool, str]:
        """
        Generic method to wait for any condition to be met.
        This uses the monitoring agent to continuously check for the condition.
        
        Args:
            detection_criteria: Description of what to detect
            system_prompt: System instructions for the monitoring LLM
            timeout: Maximum time to wait in seconds
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        if not self.tool_manager.camera_tool:
            return False, "Camera tool not available"
        
        try:
            # Ensure timeout is an integer
            try:
                timeout = int(float(timeout))  # Handle both string and float inputs
            except (ValueError, TypeError):
                logger.warning(f"Invalid timeout value '{timeout}', using default of 60 seconds")
                timeout = 60
            
            # Start the monitoring agent
            self.monitoring_agent.start_monitoring(
                detection_criteria=detection_criteria,
                system_prompt=system_prompt,
                confidence_threshold=75  # Require 75% confidence
            )
            
            # Initial announcement
            condition_summary = detection_criteria.lower()
            if self.tool_manager.speaker_tool:
                self.tool_manager.speaker_tool(f"speak I am waiting for: {condition_summary}")
            
            # Variables for status tracking
            start_time = time.time()
            status_interval = 10  # Give status updates every 10 seconds
            last_status_time = 0
            check_interval = 1  # How often to check for condition (seconds)
            
            # Check if we're running in mock mode
            is_mock_mode = os.getenv("MOCK_CAMERA_SCENARIO") is not None
            
            # Wait loop
            while time.time() - start_time < timeout:
                current_time = time.time()
                
                # Check if condition has been detected
                if self.monitoring_agent.is_detection_successful():
                    analysis = self.monitoring_agent.get_latest_analysis()
                    
                    # Announce success
                    if self.tool_manager.speaker_tool:
                        self.tool_manager.speaker_tool("speak Thank you, condition detected successfully.")
                    
                    # Stop monitoring and return success
                    try:
                        logger.info("Condition detected successfully! Stopping monitoring.")
                        self.monitoring_agent.stop_monitoring()
                    except Exception as e:
                        logger.error(f"Error stopping monitoring agent: {e}")
                    
                    return True, f"Condition detected with {analysis.get('confidence', 0)}% confidence after {int(current_time - start_time)} seconds"
                
                # In mock mode, offer option to manually advance - but only once every few seconds to prevent UI spam
                if is_mock_mode and (current_time - last_status_time >= 5):
                    last_status_time = current_time
                    print(f"\n🔧 MOCK MODE: Would you like to simulate the condition being met? ({detection_criteria})")
                    print("Type 'y' to simulate condition being met or press Enter to continue waiting.")
                    
                    # Use select to wait for input with timeout
                    import select
                    import sys
                    
                    # Wait for input with a timeout
                    ready, _, _ = select.select([sys.stdin], [], [], 1)
                    if ready:
                        user_input = sys.stdin.readline().strip().lower()
                        if user_input == 'y':
                            # Acknowledge mock success
                            if self.tool_manager.speaker_tool:
                                self.tool_manager.speaker_tool("speak Condition successfully detected (mock mode).")
                            # Stop monitoring and return success
                            self.monitoring_agent.stop_monitoring()
                            return True, f"Condition detected (mock simulation): {detection_criteria}"
                
                # Provide periodic status updates
                if current_time - last_status_time > status_interval:
                    latest = self.monitoring_agent.get_latest_analysis()
                    
                    if latest and "confidence" in latest:
                        confidence = latest["confidence"]
                        
                        if confidence > 50:
                            msg = f"I might see what I'm looking for. Confidence is {confidence}%. Please adjust if needed."
                        elif confidence > 30:
                            msg = f"I see something, but I'm not confident yet. Confidence is {confidence}%."
                        else:
                            msg = f"Still waiting for the condition to be met: {condition_summary}"
                        
                        if self.tool_manager.speaker_tool:
                            self.tool_manager.speaker_tool(f"speak {msg}")
                    
                    last_status_time = current_time
                
                # Brief sleep to prevent CPU hogging
                time.sleep(check_interval)
            
            # Timeout occurred
            if self.tool_manager.speaker_tool:
                self.tool_manager.speaker_tool("speak I've waited too long and the condition wasn't met. Let's try again.")
            
            # Stop monitoring and return failure
            try:
                self.monitoring_agent.stop_monitoring()
            except Exception as e:
                logger.error(f"Error stopping monitoring agent on timeout: {e}")
                
            return False, f"Timeout after {timeout} seconds waiting for condition: {detection_criteria}"
            
        except Exception as e:
            logger.error(f"Error in _wait_for_condition: {e}")
            # Ensure monitoring is stopped even if there's an error
            try:
                self.monitoring_agent.stop_monitoring()
            except Exception as stop_error:
                logger.error(f"Error stopping monitoring agent: {stop_error}")
                
            return False, f"Error waiting for condition: {str(e)}"
    
    def _execute_gripper_action(self, action: str) -> Tuple[bool, str]:
        """
        Execute a gripper action.
        
        Args:
            action (str): The action to execute (open, close, roll_left, etc.)
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        if not self.tool_manager.robot_arm_tool:
            return False, "Robot arm tool not available"
            
        # Check if we're in mock mode
        is_mock_mode = os.getenv("MOCK_CAMERA_SCENARIO") is not None
        
        # Handle different actions
        if action == "close":
            # Close the gripper
            response = self.tool_manager.robot_arm_tool("gripper close")
            success = "successfully" in response.lower()
            
            # Announce the action
            if success and self.tool_manager.speaker_tool:
                self.tool_manager.speaker_tool("speak Gripper closed.")
                
            return success, response
            
        elif action == "open":
            # Open the gripper
            response = self.tool_manager.robot_arm_tool("gripper open")
            success = "successfully" in response.lower()
            
            # Announce the action
            if success and self.tool_manager.speaker_tool:
                self.tool_manager.speaker_tool("speak Gripper opened.")
                
            return success, response
            
        elif action == "roll_left" or action == "twist_left":
            # Execute roll movement
            
            # Announce action
            if self.tool_manager.speaker_tool:
                self.tool_manager.speaker_tool("speak Executing twist motion.")
            
            # Start with a slight movement
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
                
            # Update mock scenario if in mock mode
            if is_mock_mode and self.tool_manager.camera_tool:
                # Update scenario for testing
                if "jar" in os.environ.get("MOCK_CAMERA_SCENARIO", ""):
                    os.environ["MOCK_CAMERA_SCENARIO"] = "jar_opened"
                    logger.info("Mock scenario changed to 'jar_opened'")
                    print("🔧 Mock scenario changed to show jar opened")
                    
                    # Take a new capture with the updated scenario
                    self.tool_manager.camera_tool("capture")
            
            return True, f"Successfully executed {action} movement"
            
        elif action == "roll_right" or action == "twist_right":
            # Execute opposite roll movement
            
            # Announce action
            if self.tool_manager.speaker_tool:
                self.tool_manager.speaker_tool("speak Executing twist motion in opposite direction.")
            
            # Start with a slight movement
            response1 = self.tool_manager.robot_arm_tool("move_joint 0.0 0.0 0.0 0.0 0.0 30.0")
            success1 = "successfully" in response1.lower()
            
            if not success1:
                return False, f"Failed to execute first twist: {response1}"
                
            # Short pause
            time.sleep(1)
            
            # More pronounced twist
            response2 = self.tool_manager.robot_arm_tool("move_joint 0.0 0.0 0.0 0.0 0.0 60.0")
            success2 = "successfully" in response2.lower()
            
            if not success2:
                return False, f"Failed to execute second twist: {response2}"
                
            return True, f"Successfully executed {action} movement"
        
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