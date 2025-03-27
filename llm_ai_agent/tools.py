#!/usr/bin/env python3
"""
Tools for the Kitchen Assistive Robot AI.
This module provides a collection of tools that can be used by agents.
"""

import re
import logging
import time
import threading
from typing import Dict, Any, Optional, List
import os
import sys
import yaml
from dotenv import load_dotenv
import torch
import numpy as np
import cv2
# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import model definition (add absolute imports to avoid potential issues)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from models.bc_transformer_policy import bc_transformer_policy
    from models.bc_act_policy import bc_act_policy
    from utils import encode_task
    MODELS_AVAILABLE = True
    logger.info("Successfully imported model modules")
except ImportError as e:
    logger.warning(f"Could not import model modules: {e}")
    MODELS_AVAILABLE = False

# Environment variables to control hardware components
ENABLE_CAMERA = os.getenv("ENABLE_CAMERA", "true").lower() == "true"
ENABLE_SPEAKER = os.getenv("ENABLE_SPEAKER", "true").lower() == "true"
ENABLE_ARM = os.getenv("ENABLE_ARM", "true").lower() == "true"

# Model configuration variables
MODEL_TYPE = os.getenv("MODEL_TYPE", "bc_transformer")  # Default to bc_act if not specified
MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "")  # Path to the model weights
INFERENCE_DURATION = int(os.getenv("INFERENCE_DURATION", "10"))  # Default 10 seconds

logger.info(f"Hardware components enabled via env vars: Camera={ENABLE_CAMERA}, Speaker={ENABLE_SPEAKER}, Arm={ENABLE_ARM}")
logger.info(f"Model configuration: Type={MODEL_TYPE}, Weights={MODEL_WEIGHTS_PATH}, Duration={INFERENCE_DURATION}")

# Import hardware tools
try:
    from .hardware_tools import CameraTools, SpeakerTools, RoboticArmTools
    HARDWARE_AVAILABLE = True
    logger.info("Successfully imported hardware tools")
except ImportError as e:
    logger.warning(f"Could not import hardware tools: {e}")
    HARDWARE_AVAILABLE = False

def calculator_tool(input_str: str) -> str:
    """
    Calculate the result of a mathematical expression.
    
    Args:
        input_str: A mathematical expression as a string (e.g., "2 + 2", "3 * 4")
        
    Returns:
        The result of the calculation as a string
    """
    try:
        # Using Python's eval but only for mathematical operations
        # This is safe as long as we're careful about what we allow
        # Convert ^ to ** for exponentiation
        input_str = input_str.replace("^", "**")
        
        # Clean up the input by removing any non-mathematical characters
        allowed_chars = set("0123456789+-*/() .**")
        cleaned_input = ''.join(c for c in input_str if c in allowed_chars)
        
        # Evaluate the expression
        result = eval(cleaned_input)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

def text_processor(text: str) -> str:
    """
    Process text by counting words, characters, and performing other text operations.
    
    Args:
        text: The text to process
        
    Returns:
        Information about the text
    """
    try:
        # Count words
        words = text.split()
        word_count = len(words)
        
        # Count characters
        char_count = len(text)
        char_count_no_spaces = len(text.replace(" ", ""))
        
        # Count sentences (simple approximation)
        sentence_count = len(re.findall(r'[.!?]+', text)) or 1
        
        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / max(1, word_count)
        
        # Calculate average sentence length
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # Generate result
        result = (
            f"Text Analysis:\n"
            f"- Words: {word_count}\n"
            f"- Characters (with spaces): {char_count}\n"
            f"- Characters (without spaces): {char_count_no_spaces}\n"
            f"- Sentences: {sentence_count}\n"
            f"- Average word length: {avg_word_length:.2f} characters\n"
            f"- Average sentence length: {avg_sentence_length:.2f} words"
        )
        
        return result
    except Exception as e:
        return f"Error processing text: {str(e)}"

def echo_tool(text: str) -> str:
    """
    Simply echoes back the input text.
    
    Args:
        text: The text to echo
        
    Returns:
        The input text prefixed with "Echo: "
    """
    return f"Echo: {text}"

# Memory access tools

def get_memory_path() -> str:
    """Helper function to get the path to the memory directory."""
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # The memory directory is a subdirectory of the current directory
    memory_dir = os.path.join(current_dir, "memory")
    return memory_dir

def get_action_plans() -> str:
    """
    Retrieve all action plans stored in memory.
    
    This tool provides access to predefined robot action sequences, including
    step-by-step procedures for tasks like opening jars or retrieving items.
    
    Returns:
        A formatted string containing all available action plans
    """
    try:
        memory_dir = get_memory_path()
        action_plans_file = os.path.join(memory_dir, "action_plan.yaml")
        
        if not os.path.exists(action_plans_file):
            return "Error: Action plans file not found in memory directory."
        
        with open(action_plans_file, 'r') as file:
            action_plans = yaml.safe_load(file)
        
        if not action_plans:
            return "No action plans found in memory."
        
        # Format the output for readability
        result = "Available Action Plans:\n\n"
        for plan_name, plan_details in action_plans.items():
            result += f"Plan: {plan_name}\n"
            result += f"Goal: {plan_details.get('goal', 'No goal specified')}\n"
            
            steps = plan_details.get('steps', [])
            if steps:
                result += "Steps:\n"
                for step in steps:
                    step_num = step.get('step_num', '?')
                    description = step.get('description', 'No description')
                    result += f"  {step_num}. {description}\n"
            else:
                result += "No steps defined for this plan.\n"
            
            result += "\n"
        
        return result
    except yaml.YAMLError as e:
        return f"Error parsing action plans YAML: {e}"
    except Exception as e:
        return f"Error retrieving action plans: {e}"

def get_action_positions() -> str:
    """
    Retrieve all stored robot arm positions from memory.
    
    This tool provides access to predefined robot arm coordinates for specific actions,
    such as positions for grasping objects or performing tasks.
    
    Returns:
        A formatted string containing all stored action positions
    """
    try:
        memory_dir = get_memory_path()
        positions_file = os.path.join(memory_dir, "action_position.yaml")
        
        if not os.path.exists(positions_file):
            return "Error: Action positions file not found in memory directory."
        
        with open(positions_file, 'r') as file:
            positions = yaml.safe_load(file)
        
        if not positions:
            return "No action positions found in memory."
        
        # Format the output for readability
        result = "Available Action Positions:\n\n"
        for action_name, coordinates in positions.items():
            result += f"Action: {action_name}\n"
            result += f"Cartesian position: {coordinates}\n\n"
        
        return result
    except yaml.YAMLError as e:
        return f"Error parsing action positions YAML: {e}"
    except Exception as e:
        return f"Error retrieving action positions: {e}"

def get_item_locations() -> str:
    """
    Retrieve the locations of all known items in the environment.
    
    This tool provides access to the stored coordinates of items that the
    robot knows about, such as objects on tables or in the workspace.
    
    Returns:
        A formatted string containing all known item locations
    """
    try:
        memory_dir = get_memory_path()
        locations_file = os.path.join(memory_dir, "item_location.yaml")
        
        if not os.path.exists(locations_file):
            return "Error: Item locations file not found in memory directory."
        
        with open(locations_file, 'r') as file:
            locations_data = yaml.safe_load(file)
        
        if not locations_data or 'items' not in locations_data:
            return "No item locations found in memory."
        
        items = locations_data.get('items', {})
        
        # Format the output for readability
        result = "Known Item Locations:\n\n"
        for item_name, item_data in items.items():
            result += f"Item: {item_name}\n"
            coordinates = item_data.get('coordinates', 'Unknown')
            result += f"Coordinates: {coordinates}\n\n"
        
        return result
    except yaml.YAMLError as e:
        return f"Error parsing item locations YAML: {e}"
    except Exception as e:
        return f"Error retrieving item locations: {e}"

def save_item_location(item_name: str, coordinates: list) -> str:
    """
    Save or update an item's location in memory.
    
    This tool adds or updates the coordinates of an item in the item_location.yaml file.
    
    Args:
        item_name: Name of the item to save
        coordinates: List of [x, y, z] coordinates in meters
        
    Returns:
        A string indicating success or failure
    """
    try:
        # Validate inputs
        if not item_name or not isinstance(item_name, str):
            return "Error: Item name must be a non-empty string"
        
        if not coordinates or not isinstance(coordinates, list) or len(coordinates) != 3:
            return "Error: Coordinates must be a list of 3 numbers [x, y, z]"
        
        # Ensure all coordinates are numbers
        try:
            coordinates = [float(c) for c in coordinates]
        except (ValueError, TypeError):
            return "Error: Coordinates must be numerical values"
        
        # Get memory directory path
        memory_dir = get_memory_path()
        locations_file = os.path.join(memory_dir, "item_location.yaml")
        
        # Create directory if it doesn't exist
        if not os.path.exists(memory_dir):
            os.makedirs(memory_dir)
        
        # Load existing data or create new structure
        if os.path.exists(locations_file):
            with open(locations_file, 'r') as file:
                locations_data = yaml.safe_load(file) or {}
        else:
            locations_data = {}
        
        # Ensure 'items' key exists
        if 'items' not in locations_data:
            locations_data['items'] = {}
        
        # Add or update item
        locations_data['items'][item_name] = {
            'coordinates': coordinates
        }
        
        # Write back to file
        with open(locations_file, 'w') as file:
            yaml.dump(locations_data, file, default_flow_style=False)
        
        return f"Successfully saved location for item '{item_name}' at coordinates {coordinates}"
        
    except yaml.YAMLError as e:
        return f"Error saving item location YAML: {e}"
    except Exception as e:
        return f"Error saving item location: {e}"

def save_action_position(action_name: str, position: list) -> str:
    """
    Save or update a robot arm position in memory.
    
    This tool adds or updates a named position in the action_position.yaml file.
    Positions are used for predefined robot movements or task-specific locations.
    
    Args:
        action_name: Name of the action or position (e.g., "grasp_cup_position")
        position: List of position values [x, y, z, theta_x, theta_y, theta_z, fingers]
        
    Returns:
        A string indicating success or failure
    """
    try:
        # Validate inputs
        if not action_name or not isinstance(action_name, str):
            return "Error: Action name must be a non-empty string"
        
        if not position or not isinstance(position, list) or len(position) != 7:
            return "Error: Position must be a list of 7 values [x, y, z, theta_x, theta_y, theta_z, fingers]"
        
        # Ensure all position values are numbers
        try:
            position = [float(p) for p in position]
        except (ValueError, TypeError):
            return "Error: Position values must be numerical"
        
        # Get memory directory path
        memory_dir = get_memory_path()
        positions_file = os.path.join(memory_dir, "action_position.yaml")
        
        # Create directory if it doesn't exist
        if not os.path.exists(memory_dir):
            os.makedirs(memory_dir)
        
        # Load existing data or create new structure
        if os.path.exists(positions_file):
            with open(positions_file, 'r') as file:
                positions_data = yaml.safe_load(file) or {}
        else:
            positions_data = {}
        
        # Add or update position
        positions_data[action_name] = position
        
        # Write back to file
        with open(positions_file, 'w') as file:
            yaml.dump(positions_data, file, default_flow_style=False)
        
        return f"Successfully saved position '{action_name}': {position}"
        
    except yaml.YAMLError as e:
        return f"Error saving action position YAML: {e}"
    except Exception as e:
        return f"Error saving action position: {e}"

# Model loading and inference functions

def load_model():
    """
    Load the model based on environment variables.
    
    Returns:
        Tuple of (model, device) or (None, None) if loading fails
    """
    if not MODELS_AVAILABLE:
        logger.error("Model modules not available. Cannot load model.")
        return None, None
    
    if not MODEL_WEIGHTS_PATH:
        logger.error("MODEL_WEIGHTS_PATH environment variable not set")
        return None, None
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Determine proprioceptive shape based on model type
        # We're using combined (15,) as default for maximum flexibility
        proprioceptive_type = "combined"
        proprio_shape = (15,)
        
        # Configure model based on model type
        if MODEL_TYPE.lower() == "bc_act":
            logger.info("Initializing BC-ACT model")
            config = {
                "obs_shape": {
                    "pixels": (3, 128, 128),
                    "pixels_egocentric": (3, 128, 128),
                    "proprioceptive": proprio_shape
                },
                "act_dim": 7,
                "policy_head": "deterministic",
                "hidden_dim": 256,
                "device": device,
                "num_queries": 10,
                "max_episode_len": 1000,
                "use_proprio": True,
                "n_layer": 4,
                "use_mpi": True,
                "mpi_root_dir": os.path.join(os.path.expanduser("~"), "Documents/GitHub/FYP-kitchen-assistive-robot/models/networks/utils/MPI/mpi/checkpoints/mpi-small")
            }
            model = bc_act_policy(**config)
        elif MODEL_TYPE.lower() == "bc_transformer":
            logger.info("Initializing BC-Transformer model")
            config = {
                "obs_shape": {
                    "pixels": (3, 128, 128),
                    "pixels_egocentric": (3, 128, 128),
                    "proprioceptive": proprio_shape
                },
                "history": False,
                "max_episode_len": 1000,
                "use_mpi_pixels_egocentric": False,
                "device": device
            }
            model = bc_transformer_policy(**config)
        else:
            logger.error(f"Unknown model type: {MODEL_TYPE}")
            return None, None
        
        # Load model weights
        logger.info(f"Loading model weights from {MODEL_WEIGHTS_PATH}")
        checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
        
        # Extract model state dict from the checkpoint
        if "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
        else:
            model_state = checkpoint
        
        # Load state dict with strict=False to be more tolerant of mismatches
        model.load_state_dict(model_state, strict=False)
        logger.info("Model loaded successfully")
        
        # Switch to evaluation mode
        model.eval()
        
        return model, device
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

def preprocess_image(frames_dict, device):
    """Preprocess multiple camera frames for model input
    
    Args:
        frames_dict: Dictionary of camera frames {cam_id: frame} (only valid frames)
        device: PyTorch device
        
    Returns:
        Dictionary mapping camera IDs to tensors of shape (B=1, T=1, C=3, H, W)
    """
    try:
        processed_frames = {}
        for cam_id, frame in frames_dict.items():
            if frame is None:
                continue
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize using cv2 (target size 128x128)
            frame_resized = cv2.resize(image_rgb, (128, 128))
            # Convert to float32, normalize to [0,1] range  
            frame_normalized = frame_resized.astype(np.float32) / 255.0
            # Convert to tensor and rearrange dimensions to (C,H,W)
            tensor_img = torch.from_numpy(frame_normalized).permute(2, 0, 1)
            # Add batch and time dimensions: (B=1, T=1, C, H, W)
            tensor_img = tensor_img.unsqueeze(0).unsqueeze(0).to(device)
            processed_frames[cam_id] = tensor_img
        return processed_frames
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        return None

def preprocess_robot_state(robot_state, device):
    """
    Preprocess robot state for model input, using combined format.
    
    Args:
        robot_state: Dictionary with 'joint_angles' and 'cartesian_pose' keys
        device: PyTorch device
        
    Returns:
        Tensor of shape (B=1, T=1, D) where D depends on proprioceptive type
    """
    try:
        if not robot_state or 'joint_angles' not in robot_state or 'cartesian_pose' not in robot_state:
            logger.error("Robot state missing required information")
            return None
        
        joint_angles = robot_state['joint_angles']
        cartesian_pose = robot_state['cartesian_pose']
        
        # Ensure joint_angles has 9 values (6 joints + 3 fingers)
        if len(joint_angles) < 9:
            joint_angles = joint_angles + [0.0] * (9 - len(joint_angles))
        
        # Ensure cartesian_pose has 6 values (x,y,z + rx,ry,rz)
        if len(cartesian_pose) < 6:
            cartesian_pose = cartesian_pose + [0.0] * (6 - len(cartesian_pose))
        
        # Convert to numpy arrays
        joint_np = np.array(joint_angles[:9], dtype=np.float32)
        cartesian_np = np.array(cartesian_pose[:6], dtype=np.float32)
        
        # Apply joint normalization
        joint_np[:6] = (joint_np[:6] - 180.0) / 180.0  # Map joint angles from [0,360] to [-1,1]
        gripper_min, gripper_max = 0.0, 10000.0
        joint_np[6:] = 2.0 * (joint_np[6:] - gripper_min) / (gripper_max - gripper_min) - 1.0
        
        # Apply cartesian normalization
        cartesian_np[:3] = cartesian_np[:3] / 1.0  # Position
        cartesian_np[3:6] = cartesian_np[3:6] / np.pi  # Orientation
        
        # Concatenate to create the combined representation
        combined_np = np.concatenate([joint_np, cartesian_np])
        
        # Convert to tensor with batch and time dimensions
        robot_tensor = torch.from_numpy(combined_np).unsqueeze(0).unsqueeze(0).to(device)
        
        return robot_tensor
    except Exception as e:
        logger.error(f"Error preprocessing robot state: {e}")
        return None

def object_manipulation(task: str) -> str:
    """
    Manipulate objects using the robot arm based on the task description.
    
    This tool uses the machine learning model to generate robot arm actions
    based on a text description of the task and visual/proprioceptive feedback.
    
    Args:
        task: Textual description of the task to perform (e.g., "pick up the cup")
        
    Returns:
        Status message indicating the result of the operation
    """
    # Check if required hardware and models are available
    if not HARDWARE_AVAILABLE:
        return "Error: Hardware tools are not available"
    
    if not MODELS_AVAILABLE:
        return "Error: Model modules are not available"
    
    if not (ENABLE_CAMERA and ENABLE_ARM):
        return "Error: Both camera and robot arm must be enabled to use this tool"
    
    # Access hardware interfaces directly
    camera_interface = None
    arm_interface = None
    
    try:
        # Get references to the hardware tool instances first
        camera_tools = None
        arm_tools = None
        
        for tool_name, tool_func in hardware_tools.items():
            if tool_name == "capture_both" and camera_tools is None:
                # Extract the CameraTools instance
                camera_tools = tool_func.__self__
            elif tool_name == "get_position" and arm_tools is None:
                # Extract the RoboticArmTools instance
                arm_tools = tool_func.__self__
        
        if not camera_tools or not arm_tools:
            return "Error: Could not access required hardware tools"
        
        # Now get direct access to the underlying interfaces
        camera_interface = camera_tools.cameras  # MultiCameraInterface
        arm_interface = arm_tools.arm  # KinovaArmInterface
        
        if not camera_interface or not arm_interface:
            return "Error: Could not access camera or arm interface directly"
        
        logger.info(f"Successfully accessed camera interface: {type(camera_interface).__name__}")
        logger.info(f"Successfully accessed arm interface: {type(arm_interface).__name__}")
        
        # Load the model
        model, device = load_model()
        if model is None:
            return "Error: Failed to load model"
        
        # Encode the task
        logger.info(f"Encoding task: {task}")
        task_emb = encode_task(task)
        
        # Set up timer for inference loop
        start_time = time.time()
        end_time = start_time + INFERENCE_DURATION
        
        # Status tracking
        success_count = 0
        failure_count = 0
        total_frames = 0
        
        logger.info(f"Starting object manipulation inference loop for task: '{task}'")
        logger.info(f"Inference will run for {INFERENCE_DURATION} seconds")
        
        # Start a separate thread to report status
        stop_flag = threading.Event()
        
        def status_reporter():
            while not stop_flag.is_set():
                elapsed = time.time() - start_time
                remaining = max(0, end_time - time.time())
                logger.info(f"Inference status: {elapsed:.1f}s elapsed, {remaining:.1f}s remaining, {success_count} actions sent")
                time.sleep(1.0)
        
        status_thread = threading.Thread(target=status_reporter)
        status_thread.daemon = True
        status_thread.start()
        
        # Get camera IDs
        env_camera_id = camera_tools.env_camera_id  # Environment camera ID
        wrist_camera_id = camera_tools.wrist_camera_id  # Wrist camera ID
        
        logger.info(f"Using environment camera ID: {env_camera_id}, wrist camera ID: {wrist_camera_id}")
        
        # Main inference loop
        try:
            while time.time() < end_time:
                # 1. Capture camera frames DIRECTLY from interface (not using tool)
                frames = camera_interface.capture_frames()
                
                if not frames:
                    logger.warning("Failed to capture camera frames, skipping iteration")
                    failure_count += 1
                    time.sleep(0.1)
                    continue
                
                # Extract frames for each camera
                if env_camera_id in frames and wrist_camera_id in frames:
                    success_env, env_frame = frames[env_camera_id]
                    success_wrist, wrist_frame = frames[wrist_camera_id]
                    
                    if not (success_env and success_wrist):
                        logger.warning("Failed to capture good frames from both cameras")
                        failure_count += 1
                        time.sleep(0.1)
                        continue
                else:
                    logger.warning(f"Missing camera frames. Available cameras: {list(frames.keys())}")
                    failure_count += 1
                    time.sleep(0.1)
                    continue
                
                total_frames += 1
                
                # 2. Get robot state DIRECTLY from interface
                joint_angles = arm_interface.get_joint_angles()
                cartesian_pose = arm_interface.get_cartesian_position()
                
                if not joint_angles or not cartesian_pose:
                    logger.warning("Failed to get robot state from interface, skipping iteration")
                    failure_count += 1
                    time.sleep(0.1)
                    continue
                
                robot_state = {
                    'joint_angles': joint_angles,
                    'cartesian_pose': cartesian_pose
                }
                
                # 3. Preprocess camera frames - NO NEED FOR BASE64 DECODING!
                frames_dict = {
                    'agent_view': env_frame,   # Direct numpy arrays
                    'ego_view': wrist_frame    # Direct numpy arrays
                }
                
                image_tensor_dict = preprocess_image(frames_dict, device)
                
                if not image_tensor_dict or 'agent_view' not in image_tensor_dict or 'ego_view' not in image_tensor_dict:
                    logger.warning("Image preprocessing failed, skipping iteration")
                    failure_count += 1
                    time.sleep(0.1)
                    continue
                
                # 4. Preprocess robot state
                robot_state_tensor = preprocess_robot_state(robot_state, device)
                if robot_state_tensor is None:
                    logger.warning("Robot state preprocessing failed, skipping iteration")
                    failure_count += 1
                    time.sleep(0.1)
                    continue
                
                # 5. Prepare input for model
                data = {
                    "pixels": image_tensor_dict['agent_view'],
                    "pixels_egocentric": image_tensor_dict['ego_view'],
                    "proprioceptive": robot_state_tensor,
                    "task_emb": task_emb
                }
                
                # 6. Run inference
                with torch.no_grad():
                    action = model.get_action(data)
                
                # 7. Send action DIRECTLY to arm interface
                # Check if action is already a numpy array or still a tensor
                if isinstance(action, torch.Tensor):
                    action = action.squeeze(0).cpu().numpy()  # Convert tensor to numpy
                else:
                    # It's already a numpy array
                    action = np.squeeze(action, axis=0) if action.ndim > 1 else action
                    
                if len(action) != 7:
                    logger.warning(f"Invalid action dimension: {len(action)}, expected 7")
                    failure_count += 1
                    continue
                
                # Extract linear, angular velocities and gripper control
                linear_velocity = action[:3] * 0.2  # Scale to appropriate range (m/s)
                angular_velocity = action[3:6] * 20.0  # Scale to appropriate range (deg/s)
                gripper_velocity = action[6] * 6000.0  # Scale to appropriate range
                
                # Send velocity commands directly to arm interface
                arm_interface.send_cartesian_velocity(
                    linear_velocity, 
                    angular_velocity,
                    fingers=(gripper_velocity, gripper_velocity, gripper_velocity),
                    hand_mode=1,
                    duration=0.03333,
                    period=0.005
                )
                success_count += 1
                
                # Control loop rate (30Hz)
                time.sleep(0.033)
        
        except Exception as e:
            logger.error(f"Error in inference loop: {e}")
            return f"Error during object manipulation: {str(e)}"
        
        finally:
            # Stop the status reporter thread
            stop_flag.set()
            status_thread.join(timeout=1.0)
            
            # Stop any ongoing robot motion
            try:
                # Send zero velocity command to stop motion
                arm_interface.send_cartesian_velocity(
                    [0.0, 0.0, 0.0],  # Zero linear velocity
                    [0.0, 0.0, 0.0],  # Zero angular velocity
                    fingers=(0.0, 0.0, 0.0),  # No finger movement
                    hand_mode=1,
                    duration=0.1,
                    period=0.005
                )
                logger.info("Sent stop command to robot arm")
            except Exception as e:
                logger.error(f"Error stopping robot motion: {e}")
        
        # Return summary
        elapsed_time = time.time() - start_time
        success_rate = success_count / max(1, total_frames) * 100
        
        result = (
            f"Object manipulation completed:\n"
            f"- Task: {task}\n"
            f"- Duration: {elapsed_time:.2f} seconds\n"
            f"- Actions sent: {success_count}\n"
            f"- Success rate: {success_rate:.1f}%"
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Error in object_manipulation: {e}")
        return f"Error: {str(e)}"

# Add more tools as needed

# Initialize hardware tools
hardware_tools = {}

if HARDWARE_AVAILABLE:
    # Initialize camera tools if enabled
    if ENABLE_CAMERA:
        try:
            camera_tools = CameraTools()
            # Add camera tools
            hardware_tools["capture"] = camera_tools.capture_environment
            hardware_tools["capture_environment"] = camera_tools.capture_environment
            hardware_tools["capture_wrist"] = camera_tools.capture_wrist
            hardware_tools["capture_both"] = camera_tools.capture_both
            logger.info("Camera tools initialized successfully")
        except Exception as e:
            logger.warning(f"Error initializing camera tools: {e}")
    else:
        logger.info("Camera tools disabled via ENABLE_CAMERA environment variable")

    # Initialize speaker tools if enabled
    if ENABLE_SPEAKER:
        try:
            speaker_tools = SpeakerTools()
            # Add speaker tools
            hardware_tools["speak"] = speaker_tools.speak
            hardware_tools["is_speaking"] = speaker_tools.is_speaking
            hardware_tools["stop_speaking"] = speaker_tools.stop_speaking
            hardware_tools["play_audio"] = speaker_tools.play_audio
            logger.info("Speaker tools initialized successfully")
        except Exception as e:
            logger.warning(f"Error initializing speaker tools: {e}")
    else:
        logger.info("Speaker tools disabled via ENABLE_SPEAKER environment variable")

    # Initialize robotic arm tools if enabled
    if ENABLE_ARM:
        try:
            arm_tools = RoboticArmTools()
            # Add robotic arm tools
            hardware_tools["move_home"] = arm_tools.move_home
            hardware_tools["move_default"] = arm_tools.move_default
            hardware_tools["move_position"] = arm_tools.move_position
            hardware_tools["grasp"] = arm_tools.grasp
            hardware_tools["release"] = arm_tools.release
            hardware_tools["get_position"] = arm_tools.get_position
            hardware_tools["send_cartesian_velocity"] = arm_tools.send_cartesian_velocity
            hardware_tools["turn_left"] = arm_tools.turn_left
            hardware_tools["turn_right"] = arm_tools.turn_right
            hardware_tools["turn_down"] = arm_tools.turn_down
            hardware_tools["turn_up"] = arm_tools.turn_up
            hardware_tools["rotate_left"] = arm_tools.rotate_left
            hardware_tools["rotate_right"] = arm_tools.rotate_right
            logger.info("Robotic arm tools initialized successfully")
        except Exception as e:
            logger.warning(f"Error initializing robotic arm tools: {e}")
    else:
        logger.info("Robotic arm tools disabled via ENABLE_ARM environment variable")

# Dictionary of all available tools
TOOLS = {
    "calculator": calculator_tool,
    "text_processor": text_processor,
    "echo": echo_tool,
    "get_action_plans": get_action_plans,
    "get_action_positions": get_action_positions,
    "get_item_locations": get_item_locations,
    "save_item_location": save_item_location,
    "save_action_position": save_action_position,
    "object_manipulation": object_manipulation,  # Add the new tool
}

# Add hardware tools to TOOLS dictionary
TOOLS.update(hardware_tools)

def get_tool(name: str):
    """
    Get a tool by name.
    
    Args:
        name: Name of the tool
        
    Returns:
        Tool function or None if not found
    """
    return TOOLS.get(name)

def get_all_tools() -> Dict[str, callable]:
    """
    Get all available tools.
    
    Returns:
        Dictionary of all tools
    """
    return TOOLS.copy()

def get_tool_names() -> List[str]:
    """
    Get the names of all available tools.
    
    Returns:
        List of tool names
    """
    return list(TOOLS.keys())

# Example usage
if __name__ == "__main__":
    # Test the tools
    print("Calculator: ", calculator_tool("2 + 2 * 3"))
    print("Text Processor: ", text_processor("This is a test. This is only a test."))
    print("Echo: ", echo_tool("Hello, world!"))
    
    # Test memory tools
    print("\nAction Plans:")
    print(get_action_plans())
    
    print("\nAction Positions:")
    print(get_action_positions())
    
    print("\nItem Locations:")
    print(get_item_locations())
    
    # Print all available tools
    print("\nAvailable tools:")
    for name in get_tool_names():
        print(f"- {name}") 

    object_manipulation("grasp cup ")