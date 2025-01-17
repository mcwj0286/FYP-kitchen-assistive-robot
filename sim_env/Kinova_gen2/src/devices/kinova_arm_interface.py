#!/usr/bin/env python3

import ctypes
from ctypes import *
import sys
import os
import time

# Define the position types as in KinovaTypes.h
ANGULAR_POSITION = 2
ANGULAR_VELOCITY = 3

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

class AngularInfo(Structure):
    _fields_ = [
        ("Actuator1", c_float),
        ("Actuator2", c_float),
        ("Actuator3", c_float),
        ("Actuator4", c_float),
        ("Actuator5", c_float),
        ("Actuator6", c_float),
        ("Actuator7", c_float),
    ]

class FingersPosition(Structure):
    _fields_ = [
        ("Finger1", c_float),
        ("Finger2", c_float),
        ("Finger3", c_float),
    ]

class TrajectoryPoint(Structure):
    _fields_ = [
        ("Position", c_int),  # Type of position
        ("Actuators", AngularInfo),  # Actuator position or velocity
        ("Fingers", FingersPosition),  # Finger position
        ("HandMode", c_int),
        ("Limitations", c_int),
        ("SynchroType", c_int),
        ("Time", c_float),
    ]
    
    def InitStruct(self):
        self.Position = 0
        self.HandMode = 0
        self.Limitations = 0
        self.SynchroType = 0
        self.Time = 0

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
            # Basic API functions
            self.init_api = self.api.InitAPI
            self.init_api.restype = c_int
            
            self.close_api = self.api.CloseAPI
            self.close_api.restype = c_int
            
            # Device functions
            self.get_devices = self.api.GetDevices
            self.get_devices.argtypes = [POINTER(KinovaDevice), POINTER(c_int)]
            self.get_devices.restype = c_int
            
            self.set_active_device = self.api.SetActiveDevice
            self.set_active_device.argtypes = [KinovaDevice]
            self.set_active_device.restype = c_int
            
            # Movement and position functions
            self.move_home = self.api.MoveHome
            self.move_home.restype = c_int
            
            self.init_fingers = self.api.InitFingers
            self.init_fingers.restype = c_int
            
            self.get_angular_command = self.api.GetAngularCommand
            self.get_angular_command.argtypes = [POINTER(AngularInfo)]
            self.get_angular_command.restype = c_int
            
            # Trajectory functions
            self.send_basic_trajectory = self.api.SendBasicTrajectory
            self.send_basic_trajectory.argtypes = [TrajectoryPoint]
            self.send_basic_trajectory.restype = c_int
            
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
            
            # Get devices
            print("Checking for connected devices...")
            devices = (KinovaDevice * 20)()
            result_count = c_int()
            result = self.get_devices(devices, byref(result_count))
            
            if result != 1 or result_count.value == 0:
                raise Exception("No devices found")
            
            device = devices[0]
            try:
                serial = device.SerialNumber.decode().strip('\x00')
                model = device.Model.decode().strip('\x00')
                print(f"Found device: Serial={serial}, Model={model}")
            except Exception as e:
                print(f"Warning: Could not decode device info: {e}")
            
            # Set the device as active
            print("Setting active device...")
            result = self.set_active_device(device)
            if result != 1:
                raise Exception(f"Failed to set active device. Result: {result}")
            print("Active device set successfully")
            
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
            
            # Get current position before moving
            try:
                current_pos = AngularInfo()
                result = self.get_angular_command(byref(current_pos))
                if result == 1:
                    print("Current position:", [
                        current_pos.Actuator1, current_pos.Actuator2,
                        current_pos.Actuator3, current_pos.Actuator4,
                        current_pos.Actuator5, current_pos.Actuator6
                    ])
            except Exception as e:
                print(f"Warning: Could not get current position: {e}")
            
            # Send home command
            print("Sending home command...")
            result = self.move_home()
            if result != 1:
                print(f"Warning: Move home returned {result}")
                return False
            
            print("Home command sent successfully")
            print("The arm is moving to home position...")
            print("Note: The arm will stop automatically when it reaches home position")
            
            return True
            
        except Exception as e:
            print(f"Failed to move home: {e}")
            return False

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