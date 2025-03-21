import argparse
import time
import torch
import cv2
import numpy as np
import os
import logging

# Import model definition
from models.bc_transformer_policy import bc_transformer_policy  # Keep this for backward compatibility
from models.bc_act_policy import bc_act_policy  # Add import for BC-ACT policy

# Fixing import paths by replacing '/' with '.' for Python module import
from sim_env.Kinova_gen2.src.devices.camera_interface import MultiCameraInterface, CameraInterface
from sim_env.Kinova_gen2.src.robot_controller import RobotController

import torchvision.transforms as T
from PIL import Image
from utils import encode_task

# Helper function to attempt retrieving joint angles with retries
def get_joint_angles_with_retry(arm, max_retries=3, wait_time=0.1):
    """Attempt to get joint angles with retry logic in case of transient errors."""
    for attempt in range(max_retries):
        joint_angles = arm.get_joint_angles()
        if joint_angles is not None:
            return joint_angles
        print(f"Warning: Failed to get joint angles on attempt {attempt+1}/{max_retries}, retrying...")
        time.sleep(wait_time)
    return None

def list_experiment_dirs(base_dir):
    """List all experiment directories in the base directory"""
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist!")
        return []
    
    exp_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not exp_dirs:
        print("No experiment directories found!")
        return []
    
    print("\nAvailable experiment directories:")
    print("-" * 50)
    for idx, dir_name in enumerate(exp_dirs, 1):
        print(f"{idx}. {dir_name}")
    print("-" * 50)
    return exp_dirs
def list_model_weights(exp_dir):
    """List all model weight files in the experiment directory"""
    def get_epoch_num(filename):
        # Extract epoch number from filename, return -1 for 'final' to sort it at the end
        if 'final' in filename:
            return float('inf')  # This will make 'final' appear last
        try:
            return int(filename.split('_')[-1].replace('.pth', ''))
        except:
            return float('inf')  # Any other special cases will also appear at the end
    
    weight_files = [f for f in os.listdir(exp_dir) if f.endswith('.pth')]
    if not weight_files:
        print("No model weights found in this directory!")
        return []
    
    # Sort weight files by epoch number
    weight_files.sort(key=get_epoch_num)
    
    print("\nAvailable model weights:")
    print("-" * 50)
    for idx, weight_file in enumerate(weight_files, 1):
        print(f"{idx}. {weight_file}")
    print("-" * 50)
    return weight_files


def select_model_weights():
    """Interactive function to select model weights"""
    base_dir = "kinova_experiments"
    
    # List and select experiment directory
    exp_dirs = list_experiment_dirs(base_dir)
    if not exp_dirs:
        return None, None
    
    while True:
        try:
            choice = int(input("\nEnter the number of the experiment directory (1-{}): ".format(len(exp_dirs))))
            if 1 <= choice <= len(exp_dirs):
                selected_exp_dir = exp_dirs[choice - 1]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    exp_path = os.path.join(base_dir, selected_exp_dir)
    
    # List and select weight file
    weight_files = list_model_weights(exp_path)
    if not weight_files:
        return None, None
    
    while True:
        try:
            choice = int(input("\nEnter the number of the weight file (1-{}): ".format(len(weight_files))))
            if 1 <= choice <= len(weight_files):
                selected_weight = weight_files[choice - 1]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Determine policy type from directory name
    policy_class, config = get_policy_class(exp_path)
    
    return os.path.join(exp_path, selected_weight), (policy_class, config)

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

def preprocess_robot_state(robot_state, control_mode='joint', proprioceptive_type='joint', device='cpu', robot_controller=None):
    """Preprocess robot state based on proprioceptive_type and normalize using dataset.py methods
    
    Args:
        robot_state: Robot state data (joint angles, cartesian pose, or both)
        control_mode: 'joint' or 'cartesian'
        proprioceptive_type: 'joint', 'cartesian', or 'combined'
        device: PyTorch device to use
        robot_controller: Robot controller instance (needed for combined mode)
    
    Returns:
        Properly normalized tensor of robot state
    """
    try:
        if proprioceptive_type == 'joint':
            # Ensure there are 9 values (pad with zeros if necessary)
            if len(robot_state) < 9:
                robot_state = robot_state + [0.0] * (9 - len(robot_state))
            
            # Convert to numpy array
            robot_state_np = np.array(robot_state, dtype=np.float32)
            
            # Apply joint normalization as in dataset.py
            # Joint angles (first 6): (angle - 180)/180, mapping [0,360] to [-1,1]
            # Gripper positions (last 3): min-max scaling
            robot_state_np[:6] = (robot_state_np[:6] - 180.0) / 180.0
            gripper_min, gripper_max = 0.0, 10000.0
            robot_state_np[6:] = 2.0 * (robot_state_np[6:] - gripper_min) / (gripper_max - gripper_min) - 1.0
            
            # Convert to tensor with batch and time dimensions
            robot_tensor = torch.from_numpy(robot_state_np).unsqueeze(0).unsqueeze(0).to(device)
            
        elif proprioceptive_type == 'cartesian':
            # Ensure there are 7 values (pad with zeros if necessary)
            if len(robot_state) < 7:
                robot_state = robot_state + [0.0] * (7 - len(robot_state))
            
            # Convert to numpy array
            robot_state_np = np.array(robot_state, dtype=np.float32)
            
            # Apply cartesian normalization as in dataset.py
            # Position (XYZ): first 3 values normalized by 1.0
            robot_state_np[:3] = robot_state_np[:3] / 1.0
            
            # Orientation (XYZ): next 3 values normalized by dividing by pi
            robot_state_np[3:6] = robot_state_np[3:6] / np.pi
            
            # Gripper: last value normalized by min-max scaling
            gripper_min, gripper_max = -6.0, 7020.0
            robot_state_np[6] = 2.0 * (robot_state_np[6] - gripper_min) / (gripper_max - gripper_min) - 1.0
            
            # Convert to tensor with batch and time dimensions
            robot_tensor = torch.from_numpy(robot_state_np).unsqueeze(0).unsqueeze(0).to(device)
            
        elif proprioceptive_type == 'combined':
            # For combined, we need both joint angles and cartesian pose
            if robot_controller is None:
                raise ValueError("Robot controller is required for combined proprioceptive mode")
                
            # Get joint angles state
            joint_angles = get_robot_state_with_retry(robot_controller.arm, control_mode='joint')
            # Get cartesian pose state
            cartesian_pose = get_robot_state_with_retry(robot_controller.arm, control_mode='cartesian')
            
            if joint_angles is None or cartesian_pose is None:
                raise ValueError("Failed to retrieve both joint and cartesian data for combined mode")
            
            # Ensure joint_angles has 9 values
            if len(joint_angles) < 9:
                joint_angles = joint_angles + [0.0] * (9 - len(joint_angles))
            
            # Ensure cartesian_pose has at least 6 values (excluding gripper)
            if len(cartesian_pose) < 6:
                cartesian_pose = cartesian_pose + [0.0] * (6 - len(cartesian_pose))
            
            # Convert to numpy arrays
            joint_np = np.array(joint_angles[:9], dtype=np.float32)
            cartesian_np = np.array(cartesian_pose[:6], dtype=np.float32)  # Only get position and orientation
            
            # Apply joint normalization
            joint_np[:6] = (joint_np[:6] - 180.0) / 180.0
            gripper_min, gripper_max = 0.0, 10000.0
            joint_np[6:] = 2.0 * (joint_np[6:] - gripper_min) / (gripper_max - gripper_min) - 1.0
            
            # Apply cartesian normalization
            cartesian_np[:3] = cartesian_np[:3] / 1.0  # Position
            cartesian_np[3:6] = cartesian_np[3:6] / np.pi  # Orientation
            
            # Concatenate to create the combined representation
            combined_np = np.concatenate([joint_np, cartesian_np])
            
            # Convert to tensor with batch and time dimensions
            robot_tensor = torch.from_numpy(combined_np).unsqueeze(0).unsqueeze(0).to(device)
        else:
            raise ValueError(f"Unsupported proprioceptive_type: {proprioceptive_type}")
            
        return robot_tensor
        
    except Exception as e:
        print(f"Error in robot state preprocessing: {e}")
        return None

def get_robot_state_with_retry(arm, control_mode='joint', max_retries=3, wait_time=0.1):
    """Attempt to get robot state with retry logic in case of transient errors."""
    for attempt in range(max_retries):
        if control_mode == 'joint':
            state = arm.get_joint_angles()
        else:  # cartesian mode
            state = arm.get_cartesian_position()
        if state is not None:
            return state
        print(f"Warning: Failed to get robot state on attempt {attempt+1}/{max_retries}, retrying...")
        time.sleep(wait_time)
    return None

def parse_args_file(args_file_path):
    """Parse the args.txt file to extract hyperparameters"""
    config = {}
    if not os.path.exists(args_file_path):
        print(f"Warning: args.txt not found at {args_file_path}")
        return None
    
    try:
        with open(args_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or ': ' not in line:
                    continue
                key, value = line.split(': ', 1)
                # Convert values to appropriate types
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit():
                    value = float(value)
                config[key] = value
        return config
    except Exception as e:
        print(f"Error parsing args.txt: {e}")
        return None

def get_policy_class(experiment_dir):
    """Determine which policy class to use based on experiment directory and args.txt"""
    args_file_path = os.path.join(experiment_dir, 'args.txt')
    args_config = parse_args_file(args_file_path)
    
    # Default device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Default values for proprioceptive dimensions
    proprio_shape = (9,)  # Default to joint angles shape
    
    if args_config and 'model_type' in args_config:
        model_type = args_config['model_type'].lower()
        print(f"Using {model_type} policy based on args.txt")
        
        # Determine proprioceptive shape based on type in args.txt
        proprioceptive_type = args_config.get("proprioceptive_type", "joint")
        if proprioceptive_type == 'joint':
            proprio_shape = (9,)
            print("Using joint angles proprioceptive shape: (9,)")
        elif proprioceptive_type == 'cartesian':
            proprio_shape = (7,)
            print("Using cartesian pose proprioceptive shape: (7,)")
        elif proprioceptive_type == 'combined':
            proprio_shape = (15,)  # 9 from joint angles + 6 from cartesian
            print("Using combined proprioceptive shape: (15,)")
        
        if model_type == 'bc_act':
            # Configure BC-ACT policy based on args.txt
            config = {
                "obs_shape": {
                    "pixels": (3, 128, 128),
                    "pixels_egocentric": (3, 128, 128),
                    "proprioceptive": proprio_shape
                },
                "act_dim": 7,
                "policy_head": args_config.get("policy_head", "deterministic"),
                "hidden_dim": args_config.get("hidden_dim", 256),
                "device": device,
                "num_queries": args_config.get("num_queries", 10),
                "max_episode_len": 1000,
                "use_proprio": True,
                "n_layer": args_config.get("n_layer", 4),
                "use_mpi": args_config.get("use_mpi", False),
                "mpi_root_dir": os.path.join(os.path.expanduser("~"), "Documents/GitHub/FYP-kitchen-assistive-robot/models/networks/utils/MPI/mpi/checkpoints/mpi-small")
            }
            # Add additional parameters if present in args.txt
            if "repr_dim" in args_config:
                config["repr_dim"] = args_config["repr_dim"]
                
            return bc_act_policy, config
        
        elif model_type == 'bc_transformer':
            config = {
                "obs_shape": {
                    "pixels": (3, 128, 128),
                    "pixels_egocentric": (3, 128, 128),
                    "proprioceptive": proprio_shape
                },
                "history": args_config.get("history", False),
                "max_episode_len": 1000,
                "use_mpi_pixels_egocentric": args_config.get("use_mpi", False),
                "device": device
            }
            return bc_transformer_policy, config
    
    # Fallback to directory name-based detection if args.txt parsing failed
    dir_name = os.path.basename(experiment_dir).lower()
    # Default to cartesian shape for fallback
    if 'bc_act' in dir_name:
        print("Using BC-ACT policy (fallback method)")
        config = {
            "obs_shape": {
                "pixels": (3, 128, 128),
                "pixels_egocentric": (3, 128, 128),
                "proprioceptive": (7,)  # Default to cartesian shape for fallback
            },
            "act_dim": 7,
            "policy_head": "deterministic",
            "hidden_dim": 256,
            "device": device,
            "num_queries": 10,
            "max_episode_len": 1000,
            "use_proprio": True,
            "n_layer": 4
        }
        return bc_act_policy, config
    elif 'bc_transformer' in dir_name:
        print("Using BC-Transformer policy (fallback method)")
        config = {
            "obs_shape": {
                "pixels": (3, 128, 128),
                "pixels_egocentric": (3, 128, 128),
                "proprioceptive": (7,)  # Default to cartesian shape for fallback
            },
            "history": False,
            "max_episode_len": 1000,
            "use_mpi_pixels_egocentric": False,
            "device": device
        }
        return bc_transformer_policy, config
    else:
        print("Warning: Could not determine policy type. Defaulting to BC-Transformer")
        config = {
            "obs_shape": {
                "pixels": (3, 128, 128),
                "pixels_egocentric": (3, 128, 128),
                "proprioceptive": (7,)  # Default to cartesian shape for fallback
            },
            "history": False,
            "max_episode_len": 1000,
            "use_mpi_pixels_egocentric": False,
            "device": device
        }
        return bc_transformer_policy, config

def main():
    parser = argparse.ArgumentParser(description="Real-time evaluation using camera and robot controller")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the inference on")
    parser.add_argument("--agent-view-cam", type=int, help="Camera ID to use for agent view")
    parser.add_argument("--ego-view-cam", type=int, help="Camera ID to use for egocentric view")
    parser.add_argument("--base-dir", type=str, default="kinova_experiments", help="Base directory for experiment directories")
    parser.add_argument("--control-mode", type=str, choices=['joint', 'cartesian'], default='cartesian',
                        help="Control mode (joint or cartesian)")
    parser.add_argument("--proprioceptive-type", type=str, choices=['joint', 'cartesian', 'combined'], default='combined',
                        help="Type of proprioceptive data to use (joint, cartesian, or combined)")
    parser.add_argument("--position", type=str, help="Custom cartesian position in format '[x, y, z, rx, ry, rz, rw]'")
    args = parser.parse_args()
    device = args.device
    control_mode = args.control_mode
    proprioceptive_type = args.proprioceptive_type

    # Parse custom cartesian position if provided
    custom_position = None
    if args.position:
        try:
            # Convert string to list of floats
            position_list = eval(args.position)
            if isinstance(position_list, list) and len(position_list) >= 7:
                # Convert string values to float if necessary
                custom_position = [float(val) for val in position_list]
                print(f"Using custom cartesian position: {custom_position}")
            else:
                print("Invalid position format. Expected 7 values. Using default home position.")
        except Exception as e:
            print(f"Error parsing position: {e}. Using default home position.")

    # Initialize variables to None for cleanup
    cameras = None
    robot_controller = None

    try:
        # Select model weights and get policy class
        print("\nPlease select the model weights to use:")
        checkpoint_path, (policy_class, config) = select_model_weights()
        if checkpoint_path is None or policy_class is None:
            print("Error: Failed to select valid model weights")
            return
        
        # Initialize the model with proper configuration
        print("\nInitializing model...")
        
        # Update device in config
        config["device"] = device
        
        # Determine proper proprioceptive shape based on command-line argument
        # This overrides any config from args.txt
        if proprioceptive_type == 'joint':
            proprio_shape = (9,)
            print(f"Using joint angles as proprioceptive data (9 dimensions) based on command-line argument")
        elif proprioceptive_type == 'cartesian':
            proprio_shape = (7,)
            print(f"Using cartesian poses as proprioceptive data (7 dimensions) based on command-line argument")
        elif proprioceptive_type == 'combined':
            proprio_shape = (15,)
            print(f"Using combined joint angles and cartesian poses as proprioceptive data (15 dimensions) based on command-line argument")
        
        # Update the proprioceptive shape in the config
        if "obs_shape" in config:
            config["obs_shape"]["proprioceptive"] = proprio_shape
        
        # Check if we're using bc_act policy with MPI
        is_bc_act_mpi = (policy_class == bc_act_policy) and config.get("use_mpi", False)
        
        if is_bc_act_mpi:
            # For bc_act_policy with MPI, use the special loading method
            print("Using special loading method for BC-ACT policy with MPI")
            try:
                model = bc_act_policy.load_from_checkpoint(
                    checkpoint_path=checkpoint_path,
                    map_location=device,
                    **config  # Pass all configuration parameters
                )
                print("Successfully loaded model with MPI weights")
            except Exception as e:
                print(f"Error loading BC-ACT model with MPI: {e}")
                print("Attempting fallback to standard loading method...")
                try:
                    # Try standard loading as fallback
                    model = policy_class(**config)
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    if "model_state_dict" in checkpoint:
                        model_state = checkpoint["model_state_dict"]
                    else:
                        model_state = checkpoint
                    model.load_state_dict(model_state, strict=False)
                    print("Successfully loaded model weights using fallback method (non-strict loading)")
                except Exception as nested_e:
                    print(f"Fallback loading also failed: {nested_e}")
                    return
        else:
            # Standard loading for other models
            # Initialize model
            model = policy_class(**config)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Extract model state dict from the checkpoint
            if "model_state_dict" in checkpoint:
                model_state = checkpoint["model_state_dict"]
            else:
                model_state = checkpoint  # In case it's already just the state dict
            
            try:
                model.load_state_dict(model_state)
                print("Successfully loaded model weights")
            except Exception as e:
                print(f"Error loading model weights: {e}")
                return
                
        model.eval()

        # Log model info for debugging
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model loaded with {num_params/1e6:.2f}M trainable parameters")
        
        # Log if MPI encoder is being used
        if hasattr(model, 'use_mpi') and model.use_mpi:
            print("Model is using MPI vision encoder")
            
        # Get task string from user and encode it
        task = input('\nType the task you want to eval: ')
        task_emb = encode_task(task)

        # Initialize camera interface with valid camera IDs
        try:
            available_cams = sorted(CameraInterface.list_available_cameras())
            print(f"Available cameras: {available_cams}")
            if not available_cams:
                print("No cameras found!")
                return
            
            # Use command line args if provided, otherwise use first two cameras
            AGENT_VIEW_CAM = args.agent_view_cam if args.agent_view_cam is not None else available_cams[0]
            EGO_VIEW_CAM = args.ego_view_cam if args.ego_view_cam is not None else available_cams[1]
            
            if AGENT_VIEW_CAM not in available_cams or EGO_VIEW_CAM not in available_cams:
                print(f"Specified cameras not available. Available cameras: {available_cams}")
                return
            
            cameras = MultiCameraInterface(camera_ids=[AGENT_VIEW_CAM, EGO_VIEW_CAM], width=320, height=240)
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return

        # Initialize robot controller (which connects and initializes the Kinova arm)
        try:
            robot_controller = RobotController(debug_mode=False, enable_controller=False, control_mode=control_mode)
            if not robot_controller.initialize_devices(move_home=False):
                print("Error initializing robot controller")
                return
            # Send zero velocity command after initialization
            zero_velocities = [0.0] * 7
            robot_controller.arm.send_angular_velocity(zero_velocities, hand_mode=1, 
                fingers=(0.0, 0.0, 0.0), duration=0.1, period=0.005)
                
            # Move to specified position if provided, otherwise stay in current position
            if custom_position:
                print("\nMoving to specified cartesian position before starting evaluation...")
                # Extract position and rotation from the 7D vector
                position = custom_position[:3]  # First 3 values (x, y, z)
                rotation = custom_position[3:6]  # Next 3 values (rx, ry, rz)
                # Send cartesian position command
                robot_controller.arm.send_cartesian_position(
                    position=position,
                    rotation=rotation,
                    fingers=(0.0, 0.0, 0.0),  # Open fingers
                    duration=5.0
                )
                # Wait for movement to complete
                print("Waiting for position movement to complete...")
                time.sleep(5)
                print("Position movement completed.")
        except Exception as e:
            print(f"Error initializing robot controller: {e}")
            return

        print("Starting inference loop... Press 'q' in any camera window to exit.")

        while True:
            # Capture frames from all cameras
            frames = cameras.capture_frames()
            # Build a dictionary of valid frames (using cam_id as key)
            frames_dict = {}
            for cam_id, (success, frame) in frames.items():
                if success and frame is not None:
                    cv2.imshow(f"Camera {cam_id}", frame)
                    frames_dict[cam_id] = frame

            # Check if user pressed 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting inference loop.")
                break

            # If no valid frames were captured, wait and continue
            if not frames_dict:
                print("No valid frames captured. Skipping iteration.")
                time.sleep(0.1)
                continue

            # Preprocess the captured images
            image_tensor_dict = preprocess_image(frames_dict, device)
            if image_tensor_dict is None or len(image_tensor_dict) == 0:
                print("Image preprocessing failed. Skipping iteration.")
                time.sleep(0.1)
                continue

            # Use mapped camera IDs for views
            agentview_tensor = image_tensor_dict[AGENT_VIEW_CAM]
            egocentric_tensor = image_tensor_dict[EGO_VIEW_CAM]

            # Get robot state from the arm using retry logic
            robot_state = get_robot_state_with_retry(robot_controller.arm, control_mode=control_mode, max_retries=3, wait_time=0.1)
            if robot_state is None:
                print("Failed to get robot state after retries. Skipping iteration.")
                time.sleep(0.03333)
                continue

            robot_state_tensor = preprocess_robot_state(
                robot_state, 
                control_mode=control_mode,
                proprioceptive_type=proprioceptive_type, 
                device=device,
                robot_controller=robot_controller
            )
            if robot_state_tensor is None:
                print("Robot state preprocessing failed. Skipping iteration.")
                time.sleep(0.03333)
                continue

            # Prepare the data dictionary for the transformer-based model
            data = {
                "pixels": agentview_tensor,             # Expected shape: (1, 1, 3, 128, 128)
                "pixels_egocentric": egocentric_tensor,   # Expected shape: (1, 1, 3, 128, 128)
                "proprioceptive": robot_state_tensor,     # Expected shape: (1, 1, 9) for joint mode or (1, 1, 7) for cartesian mode
                "task_emb": task_emb
            }

            # Run the model to predict the action
            try:
                action = model.get_action(data)  # Expecting shape: (1, 7)
                action = action.squeeze(0) 
                gripper_velocity = action[6]
                print("Predicted action:", action)
                # Send the predicted action to the robot controller/arm
                robot_controller.send_action(action, gripper_velocity)
            except Exception as e:
                print(f"Error during inference: {e}")

            # Sleep to aim for a ~30Hz control loop
            time.sleep(0.03333)

    except KeyboardInterrupt:
        print("\nExiting inference loop (keyboard interrupt).")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        print("\nCleaning up resources...")
        
        # First stop any ongoing motion
        if robot_controller and robot_controller.arm:
            try:
                print("Stopping arm motion...")
                # Send multiple zero velocity commands to ensure the arm stops
                for _ in range(3):  # Send stop command multiple times
                    zero_velocities = [0.0] * 7
                    robot_controller.arm.send_angular_velocity(zero_velocities, hand_mode=1, 
                        fingers=(0.0, 0.0, 0.0), duration=0.1, period=0.005)
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error stopping arm motion: {e}")

            try:
                # Try to move to home position before closing
                # print("Moving to home position...")
                # robot_controller.arm.move_home()
                time.sleep(2.0)  # Give some time for home movement
            except Exception as e:
                print(f"Error moving to home: {e}")

        # Then close cameras
        if cameras:
            try:
                print("Closing cameras...")
                cameras.close()
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"Error closing cameras: {e}")

        # Finally close the robot controller and arm
        if robot_controller:
            try:
                print("Closing robot controller...")
                # First set flags to stop any threads
                robot_controller.running = False
                robot_controller.emergency_stop = True
                
                # Wait for control thread
                if robot_controller.control_thread:
                    robot_controller.control_thread.join(timeout=2.0)
                
                # Explicitly close the arm API
                if robot_controller.arm:
                    print("Closing Kinova arm API...")
                    robot_controller.arm.close()
                    robot_controller.arm = None
            except Exception as e:
                print(f"Error closing robot controller: {e}")

        print("Cleanup complete.")
        # Use sys.exit instead of os._exit to allow for proper cleanup
        import sys
        sys.exit(0)


if __name__ == "__main__":
    main()
