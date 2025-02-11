#!/usr/bin/env python3

from devices.kinova_arm_interface import KinovaArmInterface
from devices.ps4controller_interface import PS4Interface
import time
import threading
import cv2
import h5py
import numpy as np
import os

class DataRecorder:
    def __init__(self, task_name):
        self.task_name = task_name
        self.file_path = f"data/{task_name}.hdf5"
        self.current_demo = None
        self.demo_idx = 0
        self.frame_idx = 0
        self.h5_file = None
        
        # Initialize or open the HDF5 file
        self._initialize_hdf5()
    
    def _initialize_hdf5(self):
        """Initialize HDF5 file and determine next demo index"""
        # Check if file exists
        file_exists = os.path.exists(self.file_path)
        
        # Open file in append mode if it exists, create it if it doesn't
        self.h5_file = h5py.File(self.file_path, 'a')
        
        # If file exists, find the next demo index
        if file_exists:
            existing_demos = [k for k in self.h5_file.keys() if k.startswith('demo_')]
            if existing_demos:
                last_demo = max(existing_demos)
                self.demo_idx = int(last_demo.split('_')[1]) + 1
            print(f"Appending to existing file. Next demo index: {self.demo_idx}")
        else:
            print(f"Created new HDF5 file: {self.file_path}")
    
    def start_new_demo(self):
        """Start recording a new demonstration"""
        demo_name = f'demo_{self.demo_idx:02d}'
        print(f"Starting new demo: {demo_name}")
        
        # Create a new group for this demo
        self.current_demo = self.h5_file.create_group(demo_name)
        
        # Create images subgroup
        images_group = self.current_demo.create_group('images')
        
        # Create resizable datasets for joint angles and actions
        # Joint angles: 6 joints + 3 fingers = 9 values
        self.current_demo.create_dataset('joint_angles', 
                                       shape=(0, 9),
                                       maxshape=(None, 9),
                                       dtype='float32')
        
        # Actions: 7 values (6 joint velocities + 1 gripper)
        self.current_demo.create_dataset('actions',
                                       shape=(0, 7),
                                       maxshape=(None, 7),
                                       dtype='float32')
        
        self.frame_idx = 0
        return demo_name
    
    def add_frame(self, frames_dict, joint_angles, action):
        """Add a new frame of data to the current demo
        
        Args:
            frames_dict: Dictionary of camera frames {cam_id: frame}
            joint_angles: Array of current joint angles (9 values)
            action: Array of joint velocities (7 values)
        """
        if self.current_demo is None:
            raise RuntimeError("No active demo. Call start_new_demo() first.")
        
        # Add frames for each camera
        for cam_id, frame in frames_dict.items():
            ds_name = f'cam_{cam_id}'
            if ds_name not in self.current_demo['images']:
                # Create new dataset for this camera
                self.current_demo['images'].create_dataset(
                    ds_name,
                    shape=(0, *frame.shape),
                    maxshape=(None, *frame.shape),
                    dtype=frame.dtype)
            
            # Resize dataset and add new frame
            dataset = self.current_demo['images'][ds_name]
            dataset.resize(self.frame_idx + 1, axis=0)
            dataset[self.frame_idx] = frame
        
        # Add joint angles
        joint_angles_dataset = self.current_demo['joint_angles']
        joint_angles_dataset.resize(self.frame_idx + 1, axis=0)
        joint_angles_dataset[self.frame_idx] = joint_angles
        
        # Add action
        actions_dataset = self.current_demo['actions']
        actions_dataset.resize(self.frame_idx + 1, axis=0)
        actions_dataset[self.frame_idx] = action
        
        self.frame_idx += 1
    
    def end_demo(self):
        """End the current demo and prepare for the next one"""
        if self.current_demo is not None:
            self.h5_file.flush()  # Ensure all data is written
            self.demo_idx += 1
            self.current_demo = None
            self.frame_idx = 0
    
    def close(self):
        """Close the HDF5 file"""
        if self.h5_file is not None:
            self.h5_file.close()

class record_demo:
    def __init__(self, debug_mode=False, camera_width=224, camera_height=224):
        self.velocity_scale = 30.0  # Maximum joint velocity in degrees/second (reduced for safety)
        self.gripper_scale = 3000  # Scale factor for gripper control
        self.running = False
        self.emergency_stop = False
        self.controller = None
        self.arm = None
        self.control_thread = None
        self.debug_mode = debug_mode
        self.cameras = None
        self.data_recorder = None
        self.recording = False
        # Set desired camera resolution
        self.camera_width = camera_width
        self.camera_height = camera_height

    def initialize_devices(self):
        """Initialize PS4 controller and Kinova arm"""
        try:
            # Initialize Kinova arm first
            print("Initializing Kinova arm...")
            self.arm = KinovaArmInterface()
            self.arm.connect()
            print("Kinova arm initialized successfully")
            
            # Move to home position first and wait for completion
            print("\nMoving to home position before starting control...")
            self.arm.move_home()
                
            # Wait for movement to complete (5 seconds should be enough)
            print("Waiting for home position movement to complete...")
            time.sleep(5)
            
            # Initialize PS4 controller
            try:
                print("Initializing PS4 controller...")
                self.controller = PS4Interface(interface="/dev/input/js0", connecting_using_ds4drv=False, debug_mode=self.debug_mode)
                print("PS4 Controller initialized")
                
                # Start controller in a separate thread
                self.controller_thread = threading.Thread(target=self.controller.listen)
                self.controller_thread.daemon = True
                self.controller_thread.start()
            except Exception as e:
                print(f"PS4 controller not available: {e}")
                return False
            
            # Initialize Camera interface
            try:
                print("Initializing Multi-Camera interface...")
                from devices.camera_interface import MultiCameraInterface
                self.cameras = MultiCameraInterface(width=self.camera_width, height=self.camera_height)
                if not self.cameras.cameras:
                    print("No cameras found!")
                # else:
                #     print(f"Successfully initialized {len(self.cameras.cameras)} cameras with resolution {self.camera_width}x{self.camera_height}")
            except Exception as e:
                print(f"Camera initialization failed: {e}")
            
            print("\nInitialization complete - ready for velocity control")
            return True
        except Exception as e:
            print(f"Error initializing devices: {e}")
            return False

    def record_demo_loop(self):
        """Main control loop for the robot using velocity control. Captures camera frames synchronously at the start of each iteration."""
        print("\nPS4 Controller Mapping:")
        print("Left Stick X/Y: Joint 1 & 2")
        print("Right Stick X/Y: Joint 3 & 4")
        print("L2/R2 Triggers: Joint 5")
        print("L1/R1 Buttons: Joint 6")
        print("Square/Circle: Gripper Open/Close")
        print("Triangle: Move to Home Position")
        print("Share: Emergency Stop")
        print("Options: Start/Stop Recording")
        print("Options: Exit")
        
        print("\nStarting velocity control loop...")
        print("Note: The arm should now be in home position and ready for velocity control")
        
        while self.running and not self.emergency_stop:
            try:
                frames_dict = {}
                # Synchronously capture frames from all cameras at the start of each iteration
                if self.cameras is not None and self.cameras.cameras:
                    frames = self.cameras.capture_frames()
                    for cam_id, (success, frame) in frames.items():
                        if success and frame is not None:
                            cv2.imshow(f"Camera {cam_id}", frame)
                            if self.recording:
                                frames_dict[cam_id] = frame
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Camera feed windows closed by user")
                        self.running = False
                        break
                
                if not self.controller:
                    print("No controller connected")
                    break
                
                # Create joint velocity array
                joint_velocities = [0.0] * 7  # 7 joints including gripper
                
                # Map controller inputs to joint velocities
                # Joint 1 & 2 - Left Stick
                joint_velocities[0] = self.controller.left_stick_x * self.velocity_scale
                joint_velocities[1] = self.controller.left_stick_y * self.velocity_scale
                
                # Joint 3 & 4 - Right Stick
                joint_velocities[2] = -self.controller.right_stick_y * self.velocity_scale if self.controller.right_stick_y != 0 else 0
                joint_velocities[3] = -self.controller.right_stick_x * self.velocity_scale if self.controller.right_stick_x != 0 else 0
                
                # Joint 5 - L2/R2 Triggers
                joint5_velocity = (self.controller.l2_trigger - self.controller.r2_trigger) * self.velocity_scale
                joint_velocities[4] = joint5_velocity
                
                # Joint 6 - L1/R1 Buttons
                joint6_velocity = 0.0
                if self.controller.r1_pressed:
                    joint6_velocity = -self.velocity_scale
                elif self.controller.l1_pressed:
                    joint6_velocity = self.velocity_scale
                joint_velocities[5] = joint6_velocity
                
                # Gripper control - Square/Circle buttons
                gripper_velocity = 0.0
                if self.controller.circle_pressed:
                    gripper_velocity = -self.gripper_scale  # Open
                elif self.controller.square_pressed:
                    gripper_velocity = self.gripper_scale   # Close
                joint_velocities[6] = gripper_velocity

                # Check for recording control (Options button)
                if self.controller.options_pressed:
                    if not self.recording:
                        print("\nStarting new demo recording...")
                        self.data_recorder.start_new_demo()
                        self.recording = True
                    else:
                        print("\nEnding demo recording...")
                        self.data_recorder.end_demo()
                        self.recording = False
                    time.sleep(0.5)  # Debounce
                
                # Check for home position request
                if self.controller.triangle_pressed:
                    print("Moving to home position...")
                    self.arm.move_home()
                    # Wait for movement to complete
                    print("Waiting for home position movement to complete...")
                    time.sleep(5)
                    print("Ready for velocity control again")
                    continue

                # Get current joint angles if recording
                if self.recording:
                    # Assuming get_joint_angles returns a 9-element array (6 joints + 3 fingers)
                    joint_angles = self.arm.get_joint_angles()
                    # Record the frame
                    self.data_recorder.add_frame(frames_dict, joint_angles, joint_velocities)

                # Send velocity commands to the arm
                self.arm.send_angular_velocity(
                    joint_velocities,  # Joint velocities array
                    hand_mode=1,
                    fingers=(gripper_velocity, gripper_velocity, gripper_velocity),
                    duration=0.03333,  # Duration updated for 30Hz control loop
                    period=0.005   # 30Hz update rate
                )

                # Print velocities for debugging if any joint is moving
                if any(abs(v) > 0.1 for v in joint_velocities):
                    print(f"Joint Velocities: {[f'{v:.1f}' for v in joint_velocities]}")
                
                # Sleep for the remainder of the control cycle
                time.sleep(0.03333)
                
            except Exception as e:
                print(f"Error in control loop: {e}")
                break

    def start(self):
        """Start the robot controller and initiate synchronous camera capture in the control loop"""
        # Get task name from user
        task_name = input("Enter task name for recording: ")
        self.data_recorder = DataRecorder(task_name)
        
        if not self.initialize_devices():
            return False
        
        self.running = True
        self.emergency_stop = False
        
        # Start control loop in a separate thread
        self.control_thread = threading.Thread(target=self.record_demo_loop)
        self.control_thread.start()
        
        return True

    def stop(self):
        """Stop the robot controller"""
        self.running = False
        if self.control_thread:
            self.control_thread.join()
        if hasattr(self, 'arm'):
            self.arm.close()
        if self.cameras:
            self.cameras.close()
        if self.data_recorder:
            self.data_recorder.close()
        cv2.destroyAllWindows()
        print("Robot controller stopped")

def main():
    # Optionally, you can set desired camera resolution by modifying the parameters below.
    # For example, to use 224x224 resolution:
    controller = record_demo(debug_mode=False, camera_width=320, camera_height=240)  # Set debug_mode=True to enable debug prints
    try:
        if controller.start():
            # Wait for the control thread to finish
            while controller.running:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nController interrupted by user")
    finally:
        controller.stop()

if __name__ == "__main__":
    main()