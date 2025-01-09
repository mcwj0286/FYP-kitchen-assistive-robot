#!/usr/bin/env python3

import ctypes
from ctypes import *
import sys
import os
# currently just test connection 
# TODO : if connection is successful, then we can start control the movement 
# TODO :  - Implement control modes:
#   - Cartesian position control
#   - Joint velocity control
#   - Gripper control
class KinovaInterface:
    def __init__(self):
        # Load the Kinova API
        self.api = None
        self.load_kinova_api()
        
        # Initialize function pointers
        self.init_api = None
        self.close_api = None
        self.get_devices = None
        self.set_active_device = None
        self.send_basic_trajectory = None
        self.get_angular_command = None
        self.move_home = None
        self.init_fingers = None
        
        # Load all required functions
        self.initialize_functions()
        
        # Initialize the API and connect to the arm
        self.initialize_arm()

    def load_kinova_api(self):
        """Load the Kinova API library"""
        try:
            # The path might need to be adjusted based on your installation
            self.api = CDLL('Kinova.API.USBCommandLayerUbuntu.so')
            print("Successfully loaded Kinova API")
        except Exception as e:
            print(f"Failed to load Kinova API: {e}")
            print("Make sure the Kinova SDK is properly installed")
            sys.exit(1)

    def initialize_functions(self):
        """Initialize all required function pointers from the API"""
        try:
            # Initialize API functions
            self.init_api = self.api.InitAPI
            self.close_api = self.api.CloseAPI
            self.get_devices = self.api.GetDevices
            self.set_active_device = self.api.SetActiveDevice
            self.send_basic_trajectory = self.api.SendBasicTrajectory
            self.get_angular_command = self.api.GetAngularCommand
            self.move_home = self.api.MoveHome
            self.init_fingers = self.api.InitFingers
            
            print("Successfully initialized API functions")
        except Exception as e:
            print(f"Failed to initialize API functions: {e}")
            sys.exit(1)

    def initialize_arm(self):
        """Initialize the connection to the arm"""
        try:
            # Initialize the API
            result = self.init_api()
            if result != 1:
                raise Exception(f"Failed to initialize API. Result: {result}")
            
            # Get list of connected devices
            devices = (KinovaDevice * 20)()  # Array to store up to 20 devices
            result = self.get_devices(devices)
            if result == 0:
                raise Exception("No devices found")
            
            # Set the first device as active
            result = self.set_active_device(devices[0])
            if result != 1:
                raise Exception("Failed to set active device")
            
            print("Successfully initialized Kinova arm")
            print(f"Connected to device: {devices[0].SerialNumber}")
            
            # Initialize fingers
            self.init_fingers()
            print("Initialized fingers")
            
        except Exception as e:
            print(f"Failed to initialize arm: {e}")
            self.close()
            sys.exit(1)

    def move_home(self):
        """Move the arm to home position"""
        try:
            result = self.move_home()
            if result != 1:
                print(f"Warning: Move home returned {result}")
            print("Moved to home position")
        except Exception as e:
            print(f"Failed to move home: {e}")

    def close(self):
        """Close the connection to the arm"""
        if self.api:
            try:
                self.close_api()
                print("Closed connection to Kinova arm")
            except Exception as e:
                print(f"Error closing API: {e}")

# Required structure for device information
class KinovaDevice(Structure):
    _fields_ = [
        ("SerialNumber", c_char * 20),
        ("Model", c_char * 20),
        ("VersionMajor", c_int),
        ("VersionMinor", c_int),
        ("VersionRelease", c_int),
        ("DeviceType", c_int),
        ("DeviceID", c_int)
    ]

def main():
    """Test the Kinova interface"""
    try:
        arm = KinovaInterface()
        print("\nMoving to home position...")
        arm.move_home()
        print("\nTest completed successfully")
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        if 'arm' in locals():
            arm.close()

if __name__ == "__main__":
    main() 