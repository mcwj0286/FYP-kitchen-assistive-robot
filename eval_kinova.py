import argparse
import time
import torch

# Import model definition
from models.bc_transformer_policy import bc_transformer_policy

# Fixing import paths by replacing '/' with '.' for Python module import
from sim_env.Kinova_gen2.src.devices.camera_interface import CameraInterface
from sim_env.Kinova_gen2.src.robot_controller import RobotController

import torchvision.transforms as T
from PIL import Image
import numpy as np


def preprocess_image(frame, device):
    # Pseudocode: Convert the captured frame (BGR from OpenCV) to RGB, resize, normalize, and convert to tensor
    try:
        # Convert frame from BGR to RGB
        frame_rgb = frame[:, :, ::-1]
        # Convert numpy array to PIL Image
        pil_img = Image.fromarray(frame_rgb)
        # Define transformation: resize to (128,128) and convert to tensor
        transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
            # Optionally add normalization if required
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor_img = transform(pil_img).to(device)  # shape: (3, 128, 128)
        # Add batch and time dimensions: (B, T, C, H, W) where T=1
        tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)
        return tensor_img
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        return None


def preprocess_robot_state(joint_angles, device):
    # Pseudocode: Convert joint angles list to a tensor normalized as in Kinova_Dataset
    # Assuming we need a tensor of shape (1, 1, 9). If joint_angles has fewer than 9 values, pad zeros.
    try:
        if len(joint_angles) < 9:
            joint_angles = joint_angles + [0.0]*(9 - len(joint_angles))
        robot_tensor = torch.tensor(joint_angles, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        # Normalize using linear normalization: (angle - 180) / 180, mapping [0,360] to [-1,1]
        robot_tensor = (robot_tensor - 180.0) / 180.0
        return robot_tensor
    except Exception as e:
        print(f"Error in robot state preprocessing: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Real-time evaluation using camera and robot controller")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint (.pth) file to load")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the inference on")
    args = parser.parse_args()

    device = args.device

    # Initialize the model with the same config as used in training
    model = bc_transformer_policy(
        history=False,
        max_episode_len=1000,
        use_mpi_pixels_egocentric=False,
        device=device
    ).to(device)

    # Load checkpoint
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"Loaded checkpoint from {args.checkpoint}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Initialize camera interface
    try:
        camera = CameraInterface(camera_id=0, width=320, height=240, fps=30)
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return

    # Initialize robot controller (which initializes and connects Kinova arm internally)
    try:
        robot_controller = RobotController(debug_mode=False)
        if not robot_controller.initialize_devices():
            print("Error initializing robot controller")
            return
    except Exception as e:
        print(f"Error initializing robot controller: {e}")
        return

    print("Starting evaluation loop... Press Ctrl+C to exit.")

    try:
        while True:
            # Capture frame from camera
            ret, frame = camera.capture_frame()
            if not ret or frame is None:
                print("Failed to capture frame from camera")
                time.sleep(0.1)
                continue

            # Preprocess the captured frame
            image_tensor = preprocess_image(frame, device)
            if image_tensor is None:
                print("Image preprocessing failed")
                time.sleep(0.1)
                continue

            # Collect robot state from the robot controller's arm
            joint_angles = robot_controller.arm.get_joint_angles()
            if joint_angles is None:
                print("Failed to get robot state")
                time.sleep(0.1)
                continue

            robot_state_tensor = preprocess_robot_state(joint_angles, device)
            if robot_state_tensor is None:
                print("Robot state preprocessing failed")
                time.sleep(0.1)
                continue

            # Prepare dummy task embedding (could be replaced with actual task embedding if available)
            task_emb = torch.zeros((1, 768), device=device)

            # Prepare data dictionary as expected by bc_transformer_policy
            data = {
                "pixels": image_tensor,             # shape: (1, 1, 3, 128, 128)
                "pixels_egocentric": image_tensor,    # using same image for now
                "proprioceptive": robot_state_tensor, # shape: (1, 1, 9)
                "task_emb": task_emb
            }

            # Perform inference
            try:
                action = model.get_action(data)  # Expected shape: (1, 7)
                print("Predicted action:", action)
            except Exception as e:
                print(f"Error during inference: {e}")
                time.sleep(0.1)
                continue

            # Use the predicted action to control the robot via the robot controller
            try:
                action = action.squeeze(0).tolist()
                gripper_velocity = action[6]
        
                robot_controller.send_action(action, gripper_velocity)
            except Exception as e:
                print(f"Error sending action to robot controller: {e}")

            time.sleep(0.1)  # Adjust loop timing as needed
    except KeyboardInterrupt:
        print("Exiting evaluation loop.")
    finally:
        camera.close()
        # Close the robot controller's arm
        robot_controller.arm.close()


if __name__ == "__main__":
    main()
