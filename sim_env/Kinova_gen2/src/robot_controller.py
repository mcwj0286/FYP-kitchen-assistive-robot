#!/usr/bin/env python3

from devices.ps4controller_interface import PS4Interface
from devices.kinova_arm_interface import KinovaInterface
import numpy as np
import time

class RobotController:
    def __init__(self):
        """Initialize robot controller with PS4 controller and Kinova arm"""
        self.controller = None
        self.arm = None
        self.running = True
        
        # Control parameters
        self.joint_velocity_scale = 30.0  # Maximum joint velocity in degrees/second
        self.gripper_speed = 0.5      # Gripper movement speed
        
        # Initialize devices
        self.initialize_devices()
        
    def initialize_devices(self):
        """Initialize PS4 controller and Kinova arm"""
        try:
            # Initialize PS4 controller
            self.controller = PS4Interface(interface="/dev/input/js0", connecting_using_ds4drv=False)
            print("PS4 Controller initialized")
            
            # Initialize Kinova arm
            self.arm = KinovaInterface()
            print("Kinova arm initialized")
            
        except Exception as e:
            print(f"Error initializing devices: {e}")
            raise
            
    def map_controller_to_joint_velocities(self):
        """Map controller inputs to joint velocities
        
        Mapping:
        - Left stick X: Joint 1 (base rotation)
        - Left stick Y: Joint 2 (shoulder)
        - Right stick X: Joint 3 (elbow)
        - Right stick Y: Joint 4 (wrist)
        - L2/R2: Joint 5 (wrist rotation)
        - L1/R1: Joint 6 (end effector rotation)
        
        Returns:
            list: Joint velocities [j1, j2, j3, j4, j5, j6]
        """
        # Get controller values
        j1_vel = self.controller.left_stick_x * self.joint_velocity_scale
        j2_vel = self.controller.left_stick_y * self.joint_velocity_scale
        j3_vel = self.controller.right_stick_x * self.joint_velocity_scale
        j4_vel = self.controller.right_stick_y * self.joint_velocity_scale
        
        # Calculate joint 5 velocity from triggers
        j5_vel = (self.controller.r2_trigger - self.controller.l2_trigger) * self.joint_velocity_scale
        
        # Calculate joint 6 velocity from L1/R1 buttons
        j6_vel = 0.0
        if self.controller.r1_pressed:
            j6_vel = self.joint_velocity_scale
        elif self.controller.l1_pressed:
            j6_vel = -self.joint_velocity_scale
            
        return [j1_vel, j2_vel, j3_vel, j4_vel, j5_vel, j6_vel]
        
    def map_controller_to_gripper(self):
        """Map controller face buttons to gripper commands
        
        Mapping:
        - Circle: Close gripper
        - X: Open gripper
        
        Returns:
            float: Gripper velocity (-1 to 1, negative for open, positive for close)
        """
        if self.controller.circle_pressed:
            return self.gripper_speed  # Close
        elif self.controller.x_pressed:
            return -self.gripper_speed  # Open
        return 0.0  # No movement
        
    def control_loop(self):
        """Main control loop for robot operation"""
        print("\nStarting robot control loop...")
        print("Control mapping:")
        print("- Left stick (X,Y): Joints 1 & 2")
        print("- Right stick (X,Y): Joints 3 & 4")
        print("- L2/R2 triggers: Joint 5")
        print("- L1/R1 buttons: Joint 6")
        print("- Circle/X: Close/Open gripper")
        print("- Options: Move to home position")
        print("- Share: Emergency stop")
        print("\nPress Ctrl+C to exit")
        
        try:
            while self.running:
                # Get joint velocities from controller
                joint_velocities = self.map_controller_to_joint_velocities()
                
                # Get gripper command
                gripper_command = self.map_controller_to_gripper()
                
                # Send commands to robot
                if any(abs(v) > 0 for v in joint_velocities):
                    self.arm.set_joint_velocities(joint_velocities)
                    
                if abs(gripper_command) > 0:
                    self.arm.control_gripper(gripper_command)
                    
                # Small delay to prevent CPU overuse
                time.sleep(0.01)  # 100Hz control loop
                
        except KeyboardInterrupt:
            print("\nStopping robot control...")
        finally:
            # Stop all motion
            self.arm.set_joint_velocities([0.0] * 6)
            self.running = False
            
    def cleanup(self):
        """Clean shutdown of controller and arm"""
        if self.arm:
            self.arm.close()
        if self.controller:
            self.controller.close()

def main():
    controller = RobotController()
    try:
        controller.control_loop()
    finally:
        controller.cleanup()

if __name__ == "__main__":
    main() 