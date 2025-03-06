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
    def __init__(self, debug_mode=False, camera_width=224, camera_height=224, 
                 record_modular_demo=False, control_mode='joint'):
        self.velocity_scale = 20.0  # Maximum joint velocity in degrees/second (reduced for safety)
        self.gripper_scale = 5000  # Scale factor for gripper control
        self.cartesian_linear_scale = 0.20  # Maximum Cartesian linear velocity in m/s
        self.cartesian_angular_scale = 40.0  # Maximum Cartesian angular velocity in degrees/s
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
        # Control mode (joint or cartesian)
        self.control_mode = control_mode

        # New: Set flag and initialize modular action attributes if enabled
        self.record_modular_demo = record_modular_demo
        if self.record_modular_demo:
            self.modular_names = []  # List to store the task names for each modular action
            self.current_modular_idx = 0  # Index to track which modular action is current
        # New: Initialize previous state for options button to prevent repeated toggles
        self.prev_options_pressed = False

    def initialize_devices(self):
        """Initialize PS4 controller and Kinova arm"""
        try:
            # Initialize Kinova arm first
            print("Initializing Kinova arm...")
            self.arm = KinovaArmInterface()
            self.arm.connect()
            print("Kinova arm initialized successfully")
            
            # Set appropriate control mode
            if self.control_mode == 'cartesian':
                print("Setting Cartesian control mode...")
                self.arm.set_cartesian_control()
            else:
                print("Setting Angular control mode...")
                self.arm.set_angular_control()
            
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
            
            print(f"\nInitialization complete - ready for {self.control_mode} control")
            return True
        except Exception as e:
            print(f"Error initializing devices: {e}")
            return False

    def record_demo_loop(self):
        """Main control loop for recording demos"""
        if self.control_mode == 'cartesian':
            self.record_cartesian_demo_loop()
        else:
            self.record_joint_demo_loop()

    def record_cartesian_demo_loop(self):
        """Control loop for Cartesian velocity control mode with recording capabilities"""
        print("\nPS4 Controller Mapping (Cartesian Velocity Mode):")
        print("Left Stick X/Y: X/Y translation")
        print("L2/R2 Triggers: Z translation (up/down)")
        print("Right Stick X: Rotation around Y (Pitch)")
        print("Right Stick Y: Rotation around X (Roll)")
        print("L1/R1 Buttons: Rotation around Z (Yaw)")
        print("Square/Circle: Gripper Open/Close")
        print("Triangle: Move to Home Position")
        print("Share: Emergency Stop")
        print("Options: Start/Stop Recording")
        
        print("\nStarting Cartesian control loop for demo recording...")
        print("Note: The arm should now be in home position and ready for Cartesian control")
        
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
                
                # Map controller inputs to Cartesian velocities
                # Linear velocities (X, Y, Z) in m/s
                linear_velocity = [0.0, 0.0, 0.0]
                linear_velocity[0] = -self.controller.left_stick_y * self.cartesian_linear_scale  # X axis
                linear_velocity[1] = -self.controller.left_stick_x * self.cartesian_linear_scale  # Y axis
                
                # Use L2/R2 triggers for Z axis movement
                z_velocity = 0.0
                if self.controller.l2_trigger:
                    z_velocity = self.cartesian_linear_scale
                elif self.controller.r2_trigger:
                    z_velocity = -self.cartesian_linear_scale
                linear_velocity[2] = z_velocity
                
                # Angular velocities (around X, Y, Z) in degrees/s
                angular_velocity = [0.0, 0.0, 0.0]
                
                # Roll (rotation around X axis) - Using Right Stick Y
                angular_velocity[0] = self.controller.right_stick_y * self.cartesian_angular_scale
                
                # Pitch (rotation around Y axis) - Right Stick X
                angular_velocity[1] = -self.controller.right_stick_x * self.cartesian_angular_scale
                
                # Yaw (rotation around Z axis) - L1/R1 Buttons
                if self.controller.r1_pressed:
                    angular_velocity[2] = -self.cartesian_angular_scale
                elif self.controller.l1_pressed:
                    angular_velocity[2] = self.cartesian_angular_scale
                
                # Gripper control - Square/Circle buttons
                gripper_velocity = 0.0
                if self.controller.circle_pressed:
                    gripper_velocity = -self.gripper_scale  # Open
                elif self.controller.square_pressed:
                    gripper_velocity = self.gripper_scale   # Close
                
                # For data recording, we need joint velocities (7 values)
                # When in Cartesian mode, we'll still record them as original format but 
                # calculate them from Cartesian velocities
                joint_velocities = [0.0] * 7
                joint_velocities[6] = gripper_velocity  # Gripper velocity is directly used
                
                # Check for recording control (Options button) using edge detection
                record_button_state = self.controller.options_pressed
                if record_button_state and not self.prev_options_pressed:
                    if not self.recording:
                        if self.record_modular_demo:
                            print(f"\nStarting new demo recording for modular action: {self.modular_names[self.current_modular_idx]}")
                        else:
                            print("\nStarting new demo recording...")
                        self.data_recorder.start_new_demo()
                        self.recording = True
                    else:
                        print("\nEnding demo recording...")
                        save_response = input("Do you want to save this demo? (y/n): ")
                        if save_response.strip().lower() in ['y', 'yes']:
                            self.data_recorder.end_demo()
                            print("Demo saved.")
                            if self.record_modular_demo:
                                # Advance to the next modular action cyclically
                                self.data_recorder.close()  # Close current modular action's file
                                self.current_modular_idx = (self.current_modular_idx + 1) % len(self.modular_names)
                                self.data_recorder = DataRecorder(self.modular_names[self.current_modular_idx])
                        else:
                            if self.record_modular_demo:
                                demo_name = self.data_recorder.current_demo.name.split('/')[-1]
                                print(f"Discarding demo for modular action {self.modular_names[self.current_modular_idx]}...")
                                if demo_name in self.data_recorder.h5_file:
                                    del self.data_recorder.h5_file[demo_name]
                                self.data_recorder.current_demo = None
                                self.data_recorder.frame_idx = 0
                                print("Demo discarded. Resetting to first modular action.")
                                self.data_recorder.close()
                                self.current_modular_idx = 0
                                self.data_recorder = DataRecorder(self.modular_names[self.current_modular_idx])
                            else:
                                demo_name = self.data_recorder.current_demo.name.split('/')[-1]
                                print(f"Discarding demo {demo_name}...")
                                if demo_name in self.data_recorder.h5_file:
                                    del self.data_recorder.h5_file[demo_name]
                                self.data_recorder.current_demo = None
                                self.data_recorder.frame_idx = 0
                                print("Demo discarded.")
                        self.recording = False
                    time.sleep(0.5)  # Debounce
                self.prev_options_pressed = record_button_state
                
                # Check for home position request
                if self.controller.triangle_pressed:
                    print("Moving to home position...")
                    self.arm.move_home()
                    # Wait for movement to complete
                    print("Waiting for home position movement to complete...")
                    time.sleep(5)
                    print("Ready for Cartesian control again")
                    continue
                
                # Get current joint angles if recording
                if self.recording:
                    # Assuming get_joint_angles returns a 9-element array (6 joints + 3 fingers)
                    joint_angles = self.arm.get_joint_angles()
                    
                    # For recording data, we need to construct an action vector
                    # In the case of cartesian control, we'll convert our cartesian commands 
                    # to something compatible with the existing dataset format
                    cartesian_action = list(linear_velocity) + list(angular_velocity) + [gripper_velocity]
                    
                    # Record the frame (using cartesian_action as the action data)
                    self.data_recorder.add_frame(frames_dict, joint_angles, cartesian_action)
                
                # Send Cartesian velocity commands to the arm
                self.arm.send_cartesian_velocity(
                    linear_velocity=tuple(linear_velocity),
                    angular_velocity=tuple(angular_velocity),
                    fingers=(gripper_velocity, gripper_velocity, gripper_velocity),
                    duration=0.03333,  # Duration for 30Hz control loop
                    period=0.005,      # 30Hz update rate
                    hand_mode=1        # Use velocity control for gripper
                )
                
                # Print velocities for debugging if there's movement
                if any(abs(v) > 0.01 for v in linear_velocity) or any(abs(v) > 0.1 for v in angular_velocity):
                    print(f"Linear Velocity: {[f'{v:.3f}' for v in linear_velocity]} m/s, " +
                          f"Angular Velocity: {[f'{v:.1f}' for v in angular_velocity]} deg/s")
                
                # Sleep for the remainder of the control cycle
                time.sleep(0.03333)
                
            except Exception as e:
                print(f"Error in Cartesian control loop: {e}")
                break

    def record_joint_demo_loop(self):
        """Original control loop for joint velocity control mode with recording capabilities"""
        print("\nPS4 Controller Mapping (Joint Velocity Mode):")
        print("Left Stick X/Y: Joint 1 & 2")
        print("Right Stick X/Y: Joint 3 & 4")
        print("L2/R2 Triggers: Joint 5")
        print("L1/R1 Buttons: Joint 6")
        print("Square/Circle: Gripper Open/Close")
        print("Triangle: Move to Home Position")
        print("Share: Emergency Stop")
        print("Options: Start/Stop Recording")
        
        print("\nStarting joint velocity control loop...")
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
                
                # Check for recording control (Options button) using edge detection
                record_button_state = self.controller.options_pressed
                if record_button_state and not self.prev_options_pressed:
                    if not self.recording:
                        if self.record_modular_demo:
                            print(f"\nStarting new demo recording for modular action: {self.modular_names[self.current_modular_idx]}")
                        else:
                            print("\nStarting new demo recording...")
                        self.data_recorder.start_new_demo()
                        self.recording = True
                    else:
                        print("\nEnding demo recording...")
                        save_response = input("Do you want to save this demo? (y/n): ")
                        if save_response.strip().lower() in ['y', 'yes']:
                            self.data_recorder.end_demo()
                            print("Demo saved.")
                            if self.record_modular_demo:
                                # Advance to the next modular action cyclically
                                self.data_recorder.close()  # Close current modular action's file
                                self.current_modular_idx = (self.current_modular_idx + 1) % len(self.modular_names)
                                self.data_recorder = DataRecorder(self.modular_names[self.current_modular_idx])
                        else:
                            if self.record_modular_demo:
                                demo_name = self.data_recorder.current_demo.name.split('/')[-1]
                                print(f"Discarding demo for modular action {self.modular_names[self.current_modular_idx]}...")
                                if demo_name in self.data_recorder.h5_file:
                                    del self.data_recorder.h5_file[demo_name]
                                self.data_recorder.current_demo = None
                                self.data_recorder.frame_idx = 0
                                print("Demo discarded. Resetting to first modular action.")
                                self.data_recorder.close()
                                self.current_modular_idx = 0
                                self.data_recorder = DataRecorder(self.modular_names[self.current_modular_idx])
                            else:
                                demo_name = self.data_recorder.current_demo.name.split('/')[-1]
                                print(f"Discarding demo {demo_name}...")
                                if demo_name in self.data_recorder.h5_file:
                                    del self.data_recorder.h5_file[demo_name]
                                self.data_recorder.current_demo = None
                                self.data_recorder.frame_idx = 0
                                print("Demo discarded.")
                        self.recording = False
                    time.sleep(0.5)  # Debounce
                self.prev_options_pressed = record_button_state
                
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
        # New: Ask for modular actions if record_modular_demo flag is enabled
        if self.record_modular_demo:
            num_actions = int(input("Enter the number of modular actions: "))
            self.modular_names = []
            for i in range(num_actions):
                task = input(f"Enter task name for modular action {i+1}: ").strip()
                self.modular_names.append(task)
            self.current_modular_idx = 0
            # Create the DataRecorder for the first modular action
            self.data_recorder = DataRecorder(self.modular_names[self.current_modular_idx])
        else:
            # Original behavior: single task name for recording
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
        
        # First stop any ongoing motion
        try:
            if self.control_mode == 'cartesian':
                # Stop Cartesian motion
                self.arm.send_cartesian_velocity(
                    linear_velocity=(0.0, 0.0, 0.0),
                    angular_velocity=(0.0, 0.0, 0.0),
                    fingers=(0.0, 0.0, 0.0),
                    duration=0.1,
                    period=0.005
                )
            else:
                # Stop joint motion
                zero_velocities = [0.0] * 7
                if self.arm:
                    self.arm.send_angular_velocity(zero_velocities, hand_mode=1, 
                        fingers=(0.0, 0.0, 0.0), duration=0.1, period=0.005)
            time.sleep(0.2)  # Wait for the command to take effect
        except:
            pass
        
        if self.control_thread:
            self.control_thread.join(timeout=2.0)  # Add timeout to prevent hanging
            
        if hasattr(self, 'arm'):
            self.arm.close()
            self.arm = None
            
        if self.cameras:
            self.cameras.close()
        if self.data_recorder:
            self.data_recorder.close()
        cv2.destroyAllWindows()
        print("Robot controller stopped")

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Robot Demo Recording Application')
    parser.add_argument('--mode', type=str, choices=['joint', 'cartesian'], default='cartesian',
                        help='Control mode (joint or cartesian)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--modular', action='store_true', help='Enable modular demo recording')
    parser.add_argument('--width', type=int, default=320, help='Camera width resolution')
    parser.add_argument('--height', type=int, default=240, help='Camera height resolution')
    args = parser.parse_args()
    
    # Initialize controller with parsed arguments
    controller = record_demo(
        debug_mode=args.debug, 
        camera_width=args.width, 
        camera_height=args.height,
        record_modular_demo=args.modular,
        control_mode=args.mode
    )
    
    print(f"Starting robot controller in {args.mode} control mode")
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