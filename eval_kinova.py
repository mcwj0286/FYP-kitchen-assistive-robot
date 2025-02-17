import argparse
import time
import torch
import cv2
import numpy as np

# Import model definition
from models.bc_transformer_policy import bc_transformer_policy

# Fixing import paths by replacing '/' with '.' for Python module import
from sim_env.Kinova_gen2.src.devices.camera_interface import MultiCameraInterface, CameraInterface
from sim_env.Kinova_gen2.src.robot_controller import RobotController

import torchvision.transforms as T
from PIL import Image
from utils import encode_task

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
            # Resize using cv2 (target size 128x128)
            frame_resized = cv2.resize(frame, (128, 128))
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

def preprocess_robot_state(joint_angles, device):
    """Preprocess robot state (joint angles) into a tensor and normalize"""
    try:
        # Ensure there are 9 values (pad with zeros if necessary)
        if len(joint_angles) < 9:
            joint_angles = joint_angles + [0.0] * (9 - len(joint_angles))
        robot_tensor = torch.tensor(joint_angles, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        # Normalize using linear normalization: (angle - 180) / 180, mapping [0,360] to [-1,1]
        robot_tensor = (robot_tensor - 180.0) / 180.0
        return robot_tensor
    except Exception as e:
        print(f"Error in robot state preprocessing: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Real-time evaluation using camera and robot controller")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the inference on")
    parser.add_argument("--agent-view-cam", type=int, help="Camera ID to use for agent view")
    parser.add_argument("--ego-view-cam", type=int, help="Camera ID to use for egocentric view")
    args = parser.parse_args()
    device = args.device

    # Set checkpoint path (adjust as needed)
    args.checkpoint = '/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robot/kinova_experiments/model_epoch_20.pth'

    # Initialize the model with the same configuration used during training:
    model = bc_transformer_policy(
        history=False,
        max_episode_len=1000,
        use_mpi_pixels_egocentric=False,
        device=device
    ).to(device)

    # Get task string from user and encode it
    task = input('Type the task you want to eval: ')
    task_emb = encode_task(task)

    # Load checkpoint
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"Loaded checkpoint from {args.checkpoint}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

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
        robot_controller = RobotController(debug_mode=False, enable_controller=False)
        if not robot_controller.initialize_devices():
            print("Error initializing robot controller")
            return
    except Exception as e:
        print(f"Error initializing robot controller: {e}")
        return

    print("Starting inference loop... Press 'q' in any camera window to exit.")

    try:
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

            # Get robot joint angles from the arm
            joint_angles = robot_controller.arm.get_joint_angles()
            if joint_angles is None:
                print("Failed to get robot state. Skipping iteration.")
                time.sleep(0.03333)
                continue

            robot_state_tensor = preprocess_robot_state(joint_angles, device)
            if robot_state_tensor is None:
                print("Robot state preprocessing failed. Skipping iteration.")
                time.sleep(0.03333)
                continue

            # Prepare the data dictionary for the transformer-based model
            data = {
                "pixels": agentview_tensor,             # Expected shape: (1, 1, 3, 128, 128)
                "pixels_egocentric": egocentric_tensor,   # Expected shape: (1, 1, 3, 128, 128)
                "proprioceptive": robot_state_tensor,     # Expected shape: (1, 1, 9)
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
            # time.sleep(0.03333)

    except KeyboardInterrupt:
        print("Exiting inference loop (keyboard interrupt).")
    finally:
        cameras.close()
        robot_controller.arm.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
