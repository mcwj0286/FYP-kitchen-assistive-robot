#!/usr/bin/env python3

import h5py
import cv2
import numpy as np
import os

def visualize_demo(h5file_path, demo_group_name, delay_ms=33, print_data=False):
    """
    Visualize a recorded demo from the given HDF5 file and demo group.

    Parameters:
        h5file_path (str): Path to the HDF5 file.
        demo_group_name (str): The name of the demo group to visualize (e.g., "demo_0").
        delay_ms (int): Delay between frames in milliseconds (default ~33 ms for 30fps).
        print_data (bool): Whether to print joint angles and actions to the console.
    """
    with h5py.File(h5file_path, 'r') as h5file:
        if demo_group_name not in h5file:
            print(f"Demo group '{demo_group_name}' not found in file!")
            return
        
        demo_group = h5file[demo_group_name]

        # Look for the images subgroup (where camera datasets are stored)
        if "images" not in demo_group:
            print("No 'images' subgroup found in the selected demo group!")
            return

        images_group = demo_group["images"]
        camera_keys = sorted(list(images_group.keys()))
        if not camera_keys:
            print("No camera datasets found in the images group!")
            return
        
        # Use the first camera dataset to figure out the number of frames
        num_frames = images_group[camera_keys[0]].shape[0]

        # Retrieve joint angles and actions datasets if they exist
        joint_angles_dataset = demo_group.get("joint_angles", None)
        actions_dataset = demo_group.get("actions", None)

        print(f"Visualizing demo '{demo_group_name}' with {len(camera_keys)} cameras and {num_frames} frames.")

        for frame_idx in range(num_frames):
            images = []
            for key in camera_keys:
                frame = images_group[key][frame_idx]  # Each frame shape: (height, width, channels)
                if frame.dtype != np.uint8:
                    # Assuming frames were normalized between 0 and 1 if not uint8.
                    frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
                images.append(frame)
            
            if len(images) == 1:
                combined = images[0]
            else:
                # Resize images to have the same height before concatenating horizontally
                heights = [img.shape[0] for img in images]
                min_height = min(heights)
                resized_images = [cv2.resize(img, (int(img.shape[1] * min_height / img.shape[0]), min_height))
                                  for img in images]
                combined = cv2.hconcat(resized_images)
            
            # Optionally print joint angles and actions to the console
            if print_data:
                if joint_angles_dataset is not None:
                    joint_angles = joint_angles_dataset[frame_idx]
                    print(f"Frame {frame_idx}: Joint Angles: " + ", ".join(f"{v:.2f}" for v in joint_angles))
                if actions_dataset is not None:
                    actions = actions_dataset[frame_idx]
                    print(f"Frame {frame_idx}: Actions: " + ", ".join(f"{v:.2f}" for v in actions))
            
            cv2.imshow("Demo Playback", combined)
            key = cv2.waitKey(delay_ms)
            # Exit if 'q' or ESC is pressed
            if key in [ord('q'), 27]:
                break
        
        cv2.destroyAllWindows()
        # Print the resolution of the last displayed image
        if 'combined' in locals():
            height, width = combined.shape[:2]
            print(f"The resolution of the image is {height} x {width}.")

def main():
    # h5file_path = input("Enter path to HDF5 demo file: ").strip()
    h5file_path = "/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robot/sim_env/Kinova_gen2/data/test.hdf5"
    if not os.path.exists(h5file_path):
        print("File does not exist!")
        return

    # Open HDF5 file to list demo groups
    with h5py.File(h5file_path, 'r') as h5file:
        demo_groups = [key for key in h5file.keys() if key.startswith("demo_")]
        if not demo_groups:
            print("No demo groups found in the file.")
            return
        print("Available demo groups:")
        for demo in demo_groups:
            print(f" - {demo}")
    
    demo_group_name = input("Enter demo group name to visualize (e.g., demo_0): ").strip()
    visualize_demo(h5file_path, demo_group_name,print_data=True)

if __name__ == "__main__":
    main() 