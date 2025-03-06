#!/usr/bin/env python
import ctypes
import sys
import os
import time

# Add angular control type constants
ANGULAR_POSITION = 2
ANGULAR_VELOCITY = 8

# Add cartesian control type constants
CARTESIAN_POSITION = 1
CARTESIAN_VELOCITY = 7

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

    def send_cartesian_position(self, position, rotation, fingers=(0.0, 0.0, 0.0)):
        """
        Send a Cartesian position command to the arm.
        
        Args:
            position: Tuple of (x, y, z) position in meters
            rotation: Tuple of (theta_x, theta_y, theta_z) rotation in degrees/radians
            fingers: Tuple of finger positions
        """
        point = TrajectoryPoint()
        
        # Set user position type to Cartesian position
        point.Position.Type = CARTESIAN_POSITION
        point.Position.Delay = 0.0
        
        # Set Cartesian Position 
        point.Position.CartesianPosition = CartesianInfo(
            position[0], position[1], position[2],
            rotation[0], rotation[1], rotation[2]
        )
        
        # Set fingers
        point.Position.Fingers = FingersPosition(*fingers)
        
        # Disable trajectory limitations
        point.LimitationsActive = 0
        point.SynchroType = 0
        point.Limitations = Limitation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        ret = self.lib.SendBasicTrajectory(point)
        if ret != self.NO_ERROR:
            print(f"SendBasicTrajectory (Cartesian) failed with error code: {ret}")
        else:
            print("Cartesian position command issued successfully.")

    def send_cartesian_velocity(self, linear_velocity, angular_velocity, 
                               fingers=(0.0, 0.0, 0.0), duration=1.0, period=0.005, hand_mode=1):
        """
        Send a Cartesian velocity command to the arm.
        
        Args:
            linear_velocity: Tuple of (vx, vy, vz) linear velocities in m/s
            angular_velocity: Tuple of (wx, wy, wz) angular velocities in deg/s
            fingers: Tuple of finger velocities
            duration: Duration to apply the velocity command in seconds
            period: Update period in seconds (typically 5ms)
            hand_mode: Hand mode (1 for velocity control of fingers)
        """
        point = TrajectoryPoint()
        
        # Set user position type to Cartesian velocity
        point.Position.Type = CARTESIAN_VELOCITY
        point.Position.Delay = 0.0
        
        # Set Cartesian velocities
        point.Position.CartesianPosition = CartesianInfo(
            linear_velocity[0], linear_velocity[1], linear_velocity[2],
            angular_velocity[0], angular_velocity[1], angular_velocity[2]
        )
        
        # Set hand mode and fingers
        point.Position.HandMode = hand_mode
        point.Position.Fingers = FingersPosition(*fingers)
        
        # Disable trajectory limitations
        point.LimitationsActive = 0
        point.SynchroType = 0
        point.Limitations = Limitation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Calculate number of cycles based on duration and period
        num_cycles = int(duration / period)
        for i in range(num_cycles):
            ret = self.lib.SendBasicTrajectory(point)
            if ret != self.NO_ERROR:
                print(f"SendBasicTrajectory (Cartesian velocity) failed with error code: {ret}")
                break
            time.sleep(period)
        
        # Send zero velocity to stop motion
        if duration > 0.1:  # Only if duration was significant
            stop_point = TrajectoryPoint()
            stop_point.Position.Type = CARTESIAN_VELOCITY
            stop_point.Position.CartesianPosition = CartesianInfo(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            stop_point.Position.Fingers = FingersPosition(*fingers)
            stop_point.LimitationsActive = 0
            stop_point.SynchroType = 0
            self.lib.SendBasicTrajectory(stop_point)

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

    def get_cartesian_force(self):
        """Retrieve the current Cartesian force values (X,Y,Z, Torque values) from the robot sensor."""
        force = CartesianInfo()
        ret = self.lib.GetCartesianForce(ctypes.byref(force))
        if ret != self.NO_ERROR:
            print(f"GetCartesianForce failed with error code: {ret}")
            return None
        return force

    def get_cartesian_position(self):
        """Get the current Cartesian position of the robot end-effector."""
        cartesian_position = CartesianInfo()
        ret = self.lib.GetCartesianCommand(ctypes.byref(cartesian_position))
        if ret != self.NO_ERROR:
            print(f"GetCartesianCommand failed with error code: {ret}")
            return None
        return cartesian_position

    def set_cartesian_control(self):
        """Switch to Cartesian control mode."""
        ret = self.lib.SetCartesianControl()
        if ret != self.NO_ERROR:
            print(f"SetCartesianControl failed with error code: {ret}")
        else:
            print("Switched to Cartesian Control mode.")

    def close(self):
        try:
            if self.lib:
                # First stop any ongoing motion
                try:
                    # Stop with appropriate control type based on last command
                    # Send zero velocity in both control modes to be safe
                    self.send_angular_velocity([0.0] * 7, hand_mode=1, 
                        fingers=(0.0, 0.0, 0.0), duration=0.1, period=0.005)
                    time.sleep(0.1)
                    
                    self.send_cartesian_velocity(
                        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 
                        fingers=(0.0, 0.0, 0.0), duration=0.1, period=0.005)
                    time.sleep(0.1)
                except:
                    pass
                
                # Stop control API first
                if hasattr(self.lib, "StopControlAPI"):
                    self.lib.StopControlAPI()
                    time.sleep(0.1)  # Give it time to stop
                
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
    manager = None
    try:
        from kinova_arm_manager import KinovaArmManager
        
        manager = KinovaArmManager()
        manager.initialize()
        # manager.print_angular_info()

        # Example of Cartesian position control
        # print("\nTesting Cartesian position control...")
        # manager.send_cartesian_position(
        #     position=(0.0, -0.4, 0.5),     # X, Y, Z position in meters
        #     rotation=(180.0, 0.0, 90.0),   # Rotation angles in degrees
        #     fingers=(1000.0, 1000.0, 1000.0)        # Finger positions
        # )
        # time.sleep(3)  # Wait for movement to complete
        # manager.print_angular_info()

        # Example of Cartesian velocity control
        print("\nTesting Cartesian velocity control...")
        # Move along Y axis at 0.15 m/s for 1 second
        # manager.send_cartesian_velocity(
        #     linear_velocity=(0.0, 0.15, 0.0),  # X, Y, Z velocities in m/s
        #     angular_velocity=(0.0, 0.0, 0.0),  # Angular velocities in deg/s
        #     duration=1.0
        # )
        
        # Get angular information
        manager.print_angular_info()
        
        # Move along Z axis at 0.1 m/s for 1 second
        manager.send_cartesian_velocity(
            linear_velocity=(0.0, 0.0, 0.0),  # X, Y, Z velocities in m/s
            angular_velocity=(50.0, 0.0, 0.0),  # Angular velocities in deg/s
            fingers=(6000,6000,6000),
            handmode=1,
            duration=3.0
        )
        
        # Get angular information again
        manager.print_angular_info()
        
        print("Moving back to home position...")
        # manager.move_home()
        time.sleep(2)

    finally:
        # Ensure we close the manager even if an error occurs
        if manager:
            manager.close()
            # Give some time for the system to clean up before exiting
            time.sleep(0.5)
    
    # Exit cleanly
    print("Program completed successfully")

if __name__ == '__main__':
    main()