#!/usr/bin/env python3

import sys
import time
import numpy as np
from kinova import *

class KinovaArm:
    def __init__(self):
        # Initialize Kinova API
        self.api = KinovaAPI()
        self.api.InitAPI()
        self.api.RefreshDeviceList()
        
        # Connect to the arm
        device_count = self.api.GetDeviceCount()
        if device_count == 0:
            print("No Kinova devices found!")
            sys.exit(1)
            
        # Initialize the arm
        self.init_robot()
        self.home_robot()
        
    def init_robot(self):
        """Initialize the robot and set initial parameters"""
        # Set control type to position
        self.api.SetActiveDevice(0)  # First device
        self.api.StartControlAPI()
        self.api.InitFingers()
        
    def home_robot(self):
        """Send robot to home position"""
        print("Homing robot...")
        self.api.MoveHome()
        
    def move_cartesian(self, x=0.0, y=0.0, z=0.0, theta_x=0.0, theta_y=0.0, theta_z=0.0):
        """Move the end effector in Cartesian space"""
        try:
            position = CartesianPosition()
            position.X = x
            position.Y = y
            position.Z = z
            position.ThetaX = theta_x
            position.ThetaY = theta_y
            position.ThetaZ = theta_z
            
            self.api.SendBasicTrajectory(position)
        except Exception as e:
            print(f"Error moving arm: {e}")
            
    def control_gripper(self, value):
        """Control gripper (0.0 to 1.0)"""
        try:
            fingers = FingersPosition()
            # Convert 0-1 range to finger position range
            finger_value = int(value * 6000)  # Assuming max value is 6000
            fingers.Finger1 = finger_value
            fingers.Finger2 = finger_value
            fingers.Finger3 = finger_value
            
            self.api.SendAdvanceTrajectory(fingers)
        except Exception as e:
            print(f"Error controlling gripper: {e}")
            
    def get_current_position(self):
        """Get current Cartesian position"""
        position = CartesianPosition()
        self.api.GetCartesianPosition(position)
        return {
            'X': position.X,
            'Y': position.Y,
            'Z': position.Z,
            'ThetaX': position.ThetaX,
            'ThetaY': position.ThetaY,
            'ThetaZ': position.ThetaZ
        }
        
    def emergency_stop(self):
        """Emergency stop the robot"""
        print("Emergency stop triggered!")
        self.api.EraseAllTrajectories()
        self.api.StopControlAPI()
        
    def close(self):
        """Clean shutdown of the arm"""
        self.api.StopControlAPI()
        self.api.CloseAPI()

def main():
    """Test function for the Kinova arm interface"""
    arm = KinovaArm()
    
    try:
        # Test basic movements
        print("Current position:", arm.get_current_position())
        
        # Move in X direction
        print("Moving in X direction...")
        arm.move_cartesian(x=0.1)
        time.sleep(2)
        
        # Test gripper
        print("Testing gripper...")
        arm.control_gripper(0.5)
        time.sleep(2)
        
        # Return home
        print("Returning home...")
        arm.home_robot()
        
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        arm.close()

if __name__ == "__main__":
    main() 