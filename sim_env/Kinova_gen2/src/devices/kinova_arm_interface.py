#!/usr/bin/env python3

import ctypes
from ctypes import *
import sys
import os
import time

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

class KinovaInterface:
    def __init__(self):
        self.api = None
        self.load_kinova_api()
        self.initialize_functions()
        self.initialize_arm()

    def load_kinova_api(self):
        """Load the Kinova API library"""
        try:
            # Try multiple possible library paths
            possible_paths = [
                'Kinova.API.USBCommandLayerUbuntu.so',
                '/usr/lib/Kinova.API.USBCommandLayerUbuntu.so',
                '/usr/local/lib/Kinova.API.USBCommandLayerUbuntu.so',
                os.path.expanduser('~/Kinova.API.USBCommandLayerUbuntu.so')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"Found Kinova API at: {path}")
                    try:
                        self.api = CDLL(path)
                        # Test if we can access a known function
                        if not hasattr(self.api, 'InitAPI'):
                            raise Exception("InitAPI function not found in library")
                        break
                    except Exception as e:
                        print(f"Failed to load {path}: {e}")
                        continue
            
            if self.api is None:
                raise Exception("Could not find valid Kinova API library")
                
            print("Successfully loaded Kinova API")
        except Exception as e:
            print(f"Failed to load Kinova API: {e}")
            sys.exit(1)

    def initialize_functions(self):
        """Initialize all required function pointers from the API"""
        try:
            # Define function prototypes
            self.init_api = self.api.InitAPI
            self.init_api.restype = c_int
            
            self.close_api = self.api.CloseAPI
            self.close_api.restype = c_int
            
            # Get devices list
            self.get_devices = self.api.GetDevices
            self.get_devices.argtypes = [POINTER(KinovaDevice)]
            self.get_devices.restype = c_int
            
            self.set_active_device = self.api.SetActiveDevice
            self.set_active_device.argtypes = [c_int]  # Changed to use device index
            self.set_active_device.restype = c_int
            
            self.start_control_api = self.api.StartControlAPI
            self.start_control_api.restype = c_int
            
            self.move_home = self.api.MoveHome
            self.move_home.restype = c_int
            
            self.init_fingers = self.api.InitFingers
            self.init_fingers.restype = c_int
            
            print("Successfully initialized API functions")
        except Exception as e:
            print(f"Failed to initialize API functions: {e}")
            sys.exit(1)

    def initialize_arm(self):
        """Initialize the connection to the arm"""
        try:
            # Initialize the API
            print("Initializing API...")
            result = self.init_api()
            if result != 1:
                raise Exception(f"Failed to initialize API. Result: {result}")
            print("API initialized successfully")
            
            # Wait a bit for the API to fully initialize
            time.sleep(1)
            
            # Create a single device structure first
            print("Checking for connected devices...")
            device = KinovaDevice()
            result = self.get_devices(byref(device))
            
            if result != 1:
                raise Exception(f"Failed to get device list. Result: {result}")
            
            try:
                serial = device.SerialNumber.decode().strip('\x00')
                model = device.Model.decode().strip('\x00')
                print(f"Found device: Serial={serial}, Model={model}")
            except Exception as e:
                print(f"Warning: Could not decode device info: {e}")
            
            # Set the device as active using index 0
            print("Setting active device...")
            result = self.set_active_device(0)  # Use index 0 for first device
            if result != 1:
                raise Exception(f"Failed to set active device. Result: {result}")
            print("Active device set successfully")
            
            # Start the control API
            print("Starting control API...")
            result = self.start_control_api()
            if result != 1:
                raise Exception(f"Failed to start control API. Result: {result}")
            print("Control API started successfully")
            
            # Initialize fingers
            print("Initializing fingers...")
            result = self.init_fingers()
            if result != 1:
                print(f"Warning: Finger initialization returned {result}")
            else:
                print("Initialized fingers successfully")
            
            print("Successfully initialized Kinova arm")
            
        except Exception as e:
            print(f"Failed to initialize arm: {e}")
            self.close()
            sys.exit(1)

    def move_to_home(self):
        """Move the arm to home position"""
        try:
            print("Moving to home position...")
            result = self.move_home()
            if result != 1:
                print(f"Warning: Move home returned {result}")
            else:
                print("Moved to home position")
        except Exception as e:
            print(f"Failed to move home: {e}")

    def close(self):
        """Close the connection to the arm"""
        if self.api:
            try:
                result = self.close_api()
                if result != 1:
                    print(f"Warning: Close API returned {result}")
                else:
                    print("Closed connection to Kinova arm")
            except Exception as e:
                print(f"Error closing API: {e}")

def main():
    """Test the Kinova interface"""
    try:
        print("Initializing Kinova Interface...")
        arm = KinovaInterface()
        input("Press Enter to move to home position...")
        arm.move_to_home()
        print("\nTest completed successfully")
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        if 'arm' in locals():
            arm.close()

if __name__ == "__main__":
    main() 