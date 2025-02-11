#!/usr/bin/env python
import ctypes
import sys
import os
import time

# Add angular control type constants
ANGULAR_POSITION = 2
ANGULAR_VELOCITY = 8

# Define the KinovaDevice structure
class KinovaDevice(ctypes.Structure):
    _fields_ = [
        ("SerialNumber", ctypes.c_char * 20),
        ("Model", ctypes.c_char * 20),
        ("VersionMajor", ctypes.c_int),
        ("VersionMinor", ctypes.c_int),
        ("VersionRelease", ctypes.c_int),
        ("DeviceType", ctypes.c_int),
        ("DeviceID", ctypes.c_int)
    ]

# Define AngularInfo structure (7 actuators)
class AngularInfo(ctypes.Structure):
    _fields_ = [
        ("Actuator1", ctypes.c_float),
        ("Actuator2", ctypes.c_float),
        ("Actuator3", ctypes.c_float),
        ("Actuator4", ctypes.c_float),
        ("Actuator5", ctypes.c_float),
        ("Actuator6", ctypes.c_float),
        ("Actuator7", ctypes.c_float)
    ]

# Define FingersPosition structure (3 fingers)
class FingersPosition(ctypes.Structure):
    _fields_ = [
        ("Finger1", ctypes.c_float),
        ("Finger2", ctypes.c_float),
        ("Finger3", ctypes.c_float)
    ]

# Define AngularPosition structure that combines actuators and fingers
class AngularPosition(ctypes.Structure):
    _fields_ = [
        ("Actuators", AngularInfo),
        ("Fingers", FingersPosition)
    ]

# Define CartesianInfo structure (6 floats)
class CartesianInfo(ctypes.Structure):
    _fields_ = [
        ("X", ctypes.c_float),
        ("Y", ctypes.c_float),
        ("Z", ctypes.c_float),
        ("ThetaX", ctypes.c_float),
        ("ThetaY", ctypes.c_float),
        ("ThetaZ", ctypes.c_float)
    ]

# Define Limitation structure
class Limitation(ctypes.Structure):
    _fields_ = [
        ("speedParameter1", ctypes.c_float),
        ("speedParameter2", ctypes.c_float),
        ("speedParameter3", ctypes.c_float),
        ("forceParameter1", ctypes.c_float),
        ("forceParameter2", ctypes.c_float),
        ("forceParameter3", ctypes.c_float),
        ("accelerationParameter1", ctypes.c_float),
        ("accelerationParameter2", ctypes.c_float),
        ("accelerationParameter3", ctypes.c_float)
    ]

# Define UserPosition structure
class UserPosition(ctypes.Structure):
    pass

# Define TrajectoryPoint structure
class TrajectoryPoint(ctypes.Structure):
    pass

# Complete the UserPosition structure with proper references
UserPosition._fields_ = [
    ("Type", ctypes.c_int),
    ("Delay", ctypes.c_float),
    ("CartesianPosition", CartesianInfo),
    ("Actuators", AngularInfo),
    ("HandMode", ctypes.c_int),
    ("Fingers", FingersPosition)
]

# Complete the TrajectoryPoint structure with proper references
TrajectoryPoint._fields_ = [
    ("Position", UserPosition),
    ("LimitationsActive", ctypes.c_int),
    ("SynchroType", ctypes.c_int),
    ("Limitations", Limitation)
]

class KinovaArmInterface:
    NO_ERROR = 1
    MAX_KINOVA_DEVICE = 20

    def __init__(self, lib_name='/opt/JACO-SDK/API/Kinova.API.USBCommandLayerUbuntu.so'):
        self.lib = None
        try:
            # Try multiple possible library paths
            possible_paths = [
                lib_name,
                'Kinova.API.USBCommandLayerUbuntu.so',
                '/usr/lib/Kinova.API.USBCommandLayerUbuntu.so',
                '/usr/local/lib/Kinova.API.USBCommandLayerUbuntu.so',
                os.path.expanduser('~/Kinova.API.USBCommandLayerUbuntu.so')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"Found Kinova API at: {path}")
                    try:
                        self.lib = ctypes.CDLL(path)
                        if not hasattr(self.lib, 'InitAPI'):
                            raise Exception("InitAPI function not found in library")
                        break
                    except Exception as e:
                        print(f"Failed to load {path}: {e}")
                        continue
            
            if self.lib is None:
                raise Exception("Could not find valid Kinova API library")
                
            print("Successfully loaded Kinova API")
        except Exception as e:
            print(f"Failed to load Kinova API: {e}")
            sys.exit(1)

        # Setup function prototypes
        self.lib.InitAPI.restype = ctypes.c_int
        self.lib.GetDevices.argtypes = [ctypes.POINTER(KinovaDevice), ctypes.POINTER(ctypes.c_int)]
        self.lib.GetDevices.restype = ctypes.c_int
        self.lib.SetActiveDevice.argtypes = [KinovaDevice]
        self.lib.SetActiveDevice.restype = ctypes.c_int
        self.lib.MoveHome.restype = ctypes.c_int
        self.lib.SetAngularControl.restype = ctypes.c_int
        self.lib.SendBasicTrajectory.argtypes = [TrajectoryPoint]
        self.lib.SendBasicTrajectory.restype = ctypes.c_int

        # For reading angular positions
        self.POSITION_CURRENT_COUNT = 7
        self.lib.GetPositionCurrentActuators.argtypes = [ctypes.POINTER(ctypes.c_float)]
        self.lib.GetPositionCurrentActuators.restype = ctypes.c_int

        # Setup GetAngularCommand and GetAngularPosition like in the C++ example
        self.lib.GetAngularCommand.argtypes = [ctypes.POINTER(AngularPosition)]
        self.lib.GetAngularCommand.restype = ctypes.c_int
        self.lib.GetAngularPosition.argtypes = [ctypes.POINTER(AngularPosition)]
        self.lib.GetAngularPosition.restype = ctypes.c_int

        self.device = None

    def connect(self):
        # Initialize API
        ret = self.lib.InitAPI()
        if ret != self.NO_ERROR:
            print(f"InitAPI failed with error code: {ret}")
            sys.exit(1)

        # Get available devices
        devices = (KinovaDevice * self.MAX_KINOVA_DEVICE)()
        device_count = ctypes.c_int(0)
        ret = self.lib.GetDevices(devices, ctypes.byref(device_count))
        if ret != self.NO_ERROR:
            print(f"GetDevices failed with error code: {ret}")
            sys.exit(1)

        if device_count.value == 0:
            print("No Kinova devices found.")
            sys.exit(1)

        # Select the first device
        self.device = devices[0]
        ret = self.lib.SetActiveDevice(self.device)
        if ret != self.NO_ERROR:
            print(f"SetActiveDevice failed with error code: {ret}")
            sys.exit(1)

        serial = self.device.SerialNumber.decode('utf-8', errors='ignore').strip('\x00')
        model = self.device.Model.decode('utf-8', errors='ignore').strip('\x00')
        print(f"Connected to device. Serial: {serial}, Model: {model}")

    def move_home(self):
        ret = self.lib.MoveHome()
        if ret != self.NO_ERROR:
            print(f"MoveHome failed with error code: {ret}")
        else:
            print("MoveHome command issued successfully.")

    def set_angular_control(self):
        ret = self.lib.SetAngularControl()
        if ret != self.NO_ERROR:
            print(f"SetAngularControl failed with error code: {ret}")
        else:
            print("Switched to Angular Control mode.")

    def send_angular_trajectory(self, actuator_angles, hand_mode=1, fingers=(0.0, 0.0, 0.0)):
        # actuator_angles should be a list or tuple of 7 floats
        if len(actuator_angles) != 7:
            print("Error: actuator_angles must contain 7 values.")
            return

        point = TrajectoryPoint()
        
        # Set user position type to angular (assumed value 2)
        point.Position.Type = 2
        point.Position.Delay = 0.0

        # Set Cartesian Position to zeros (not used in angular mode)
        point.Position.CartesianPosition = CartesianInfo(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Set angular positions from the provided actuator_angles list
        point.Position.Actuators = AngularInfo(*actuator_angles)
        point.Position.HandMode = hand_mode
        point.Position.Fingers = FingersPosition(*fingers)

        # Disable trajectory limitations
        point.LimitationsActive = 0
        point.SynchroType = 0
        point.Limitations = Limitation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        ret = self.lib.SendBasicTrajectory(point)
        if ret != self.NO_ERROR:
            print(f"SendBasicTrajectory failed with error code: {ret}")
        else:
            print("Angular trajectory command issued successfully.")

    def send_angular_velocity(self, joint_velocities, hand_mode=1, fingers=(0.0, 0.0, 0.0), duration=2.0, period=0.005):
        # joint_velocities should be a list or tuple of 7 floats representing velocity commands
        if len(joint_velocities) != 7:
            print("Error: joint_velocities must contain 7 values.")
            return

        point = TrajectoryPoint()
        
        # Set user position type to angular velocity (assumed value 8)
        point.Position.Type = ANGULAR_VELOCITY
        point.Position.Delay = 0.0

        # Set Cartesian Position to zeros (not used in angular velocity mode)
        point.Position.CartesianPosition = CartesianInfo(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Set angular velocities from the provided joint_velocities list
        point.Position.Actuators = AngularInfo(*joint_velocities)
        point.Position.HandMode = hand_mode
        point.Position.Fingers = FingersPosition(*fingers)

        # Disable trajectory limitations
        point.LimitationsActive = 0
        point.SynchroType = 0
        point.Limitations = Limitation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        num_cycles = int(duration / period)
        for i in range(num_cycles):
            ret = self.lib.SendBasicTrajectory(point)
            if ret != self.NO_ERROR:
                print(f"SendBasicTrajectory (velocity) failed with error code: {ret}")
            time.sleep(period)
        # print("Angular velocity command issued successfully.")

 

    def get_current_angular_positions(self):
        arr = (ctypes.c_float * self.POSITION_CURRENT_COUNT)()
        ret = self.lib.GetPositionCurrentActuators(arr)
        if ret != self.NO_ERROR:
            print(f"GetPositionCurrentActuators failed with error code: {ret}")
            return None
        return [arr[i] for i in range(self.POSITION_CURRENT_COUNT)]

    def print_angular_info(self):
        positions = self.get_current_angular_positions()
        if positions is not None:
            print("Current Angular Positions:")
            for i, pos in enumerate(positions, start=1):
                print(f" Actuator {i}: {pos}")

    def print_finger_info(self):
        # Get both command and position data for fingers like in the C++ example
        command_data = AngularPosition()
        position_data = AngularPosition()
        
        ret_cmd = self.lib.GetAngularCommand(ctypes.byref(command_data))
        ret_pos = self.lib.GetAngularPosition(ctypes.byref(position_data))
        
        if ret_cmd == self.NO_ERROR and ret_pos == self.NO_ERROR:
            print("*********************************")
            print(f"  Finger 1   command: {command_data.Fingers.Finger1}     Position: {position_data.Fingers.Finger1}")
            print(f"  Finger 2   command: {command_data.Fingers.Finger2}     Position: {position_data.Fingers.Finger2}")
            print(f"  Finger 3   command: {command_data.Fingers.Finger3}     Position: {position_data.Fingers.Finger3}")
            print("*********************************")
        else:
            print(f"Failed to get finger information. Command error: {ret_cmd}, Position error: {ret_pos}")

    def get_joint_angles(self):
        # Create an AngularPosition instance
        angular_pos = AngularPosition()
        ret = self.lib.GetAngularPosition(ctypes.byref(angular_pos))
        if ret != self.NO_ERROR:
            print(f"GetAngularPosition failed with error code: {ret}")
            return None
        
        # Extract the joint angles: take first 6 actuators (assumed joints) and then 3 finger positions
        joint_angles = [
            angular_pos.Actuators.Actuator1,
            angular_pos.Actuators.Actuator2,
            angular_pos.Actuators.Actuator3,
            angular_pos.Actuators.Actuator4,
            angular_pos.Actuators.Actuator5,
            angular_pos.Actuators.Actuator6
        ]
        finger_angles = [
            angular_pos.Fingers.Finger1,
            angular_pos.Fingers.Finger2,
            angular_pos.Fingers.Finger3
        ]
        return joint_angles + finger_angles

    def close(self):
        try:
            if self.lib:
                # Stop control API first
                if hasattr(self.lib, "StopControlAPI"):
                    self.lib.StopControlAPI()
                
                # Then close the API
                ret = self.lib.CloseAPI()
                if ret != self.NO_ERROR:
                    print(f"Warning: Close API returned {ret}")
                else:
                    print("API closed successfully")
                
                # Clear the library reference
                self.lib = None
        except Exception as e:
            print(f"Error closing API: {e}")

def main():
    arm = None
    try:
        arm = KinovaArmInterface()
        arm.connect()
        arm.move_home()
        arm.set_angular_control()
        
        # Example of position control
        print("\nTesting position control...")
        arm.send_angular_trajectory([273.0, 167.0, 57.0, 240.0, 82.0, 65.0, 0.0], 
                                  hand_mode=1, fingers=(0.0, 0.0, 0.0))
        time.sleep(2)
        arm.print_finger_info()

        # Test simulated velocity control
        print("\nTesting simulated velocity control (closing fingers)...")
        # Move joint 6 at 20 deg/s while closing fingers
        arm.send_angular_velocity(
            [0.0, 0.0, 0.0, 0.0, 0.0, -40.0, 0.0],  # Joint velocities
            hand_mode=1,
            fingers=(2000.0, 0.0, 2000.0),  # Close fingers
            duration=2.0,
            period=0.005  # 50Hz update rate
        )
        # time.sleep(2)
        arm.print_finger_info()

        print("\nTesting simulated velocity control (opening fingers)...")
        # Move joint 6 at -20 deg/s while opening fingers
        arm.send_angular_velocity(
            [0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0],  # Joint velocities
            hand_mode=1,
            fingers=(-1000.0, -1000.0, -1000.0),  # Open fingers
            duration=2.0,
            period=0.005  # 50Hz update rate
        )
        time.sleep(2)
        arm.print_finger_info()

    finally:
        # Ensure we close the API even if an error occurs
        if arm is not None:
            arm.close()
            # Give some time for the system to clean up before exiting
            time.sleep(0.1)
    
    # Exit cleanly
    sys.exit(0)

if __name__ == '__main__':
    main()