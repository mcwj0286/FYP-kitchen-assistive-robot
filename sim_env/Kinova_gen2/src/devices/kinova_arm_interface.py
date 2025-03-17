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

class CartesianPosition(ctypes.Structure):
    _fields_ = [
        ("X", ctypes.c_float),
        ("Y", ctypes.c_float),
        ("Z", ctypes.c_float),
        ("ThetaX", ctypes.c_float),
        ("ThetaY", ctypes.c_float),
        ("ThetaZ", ctypes.c_float),
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

    def send_angular_position(self, joint_angles, fingers=(0.0, 0.0, 0.0), speed_factor=0.3):
        """
        Send an angular position command to move the arm to specific joint angles.
        
        Args:
            joint_angles: List or tuple of 6 joint angles in degrees
            fingers: Tuple of finger positions (0.0 for open, ~6000.0 for closed)
            speed_factor: Factor to control movement speed (0.1 to 1.0, lower is slower/safer)
        """
        if speed_factor < 0.1 or speed_factor > 1.0:
            print(f"Warning: speed_factor {speed_factor} out of safe range [0.1, 1.0], using default of 0.3")
            speed_factor = 0.3
            
        point = TrajectoryPoint()
        
        # Set user position type to Angular position
        point.Position.Type = ANGULAR_POSITION
        point.Position.Delay = 2
        
        # Set the angular position
        point.Position.Actuators = AngularInfo(
            joint_angles[0], joint_angles[1], joint_angles[2],
            joint_angles[3], joint_angles[4], joint_angles[5], 0.0
        )
        
        # Set fingers
        point.Position.Fingers = FingersPosition(*fingers)
        
        # Enable trajectory limitations for safety
        point.LimitationsActive = 1
        point.SynchroType = 0
        
        # Set speed limitations (values determined empirically for safe motion)
        # Lower values = slower movement
        max_speed = 25.0 * speed_factor  # Reduced maximum speed
        max_force = 50.0
        max_accel = 15.0 * speed_factor
        
        point.Limitations = Limitation(
            max_speed, max_speed, max_speed,       # Speed parameters
            max_force, max_force, max_force,       # Force parameters
            max_accel, max_accel, max_accel        # Acceleration parameters
        )
        
        ret = self.lib.SendBasicTrajectory(point)
        if ret != self.NO_ERROR:
            print(f"SendBasicTrajectory (Angular position) failed with error code: {ret}")
        else:
            print(f"Angular position command issued successfully (speed factor: {speed_factor}).")

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

    def send_cartesian_position(self, position, rotation, fingers=(0.0, 0.0, 0.0), duration=5.0, period=0.05):
        """
        Send a Cartesian position command to the arm with proper trajectory planning.
        
        Args:
            position: Tuple of (x, y, z) position in meters
            rotation: Tuple of (theta_x, theta_y, theta_z) rotation in degrees/radians
            fingers: Tuple of finger positions
            duration: Duration to keep sending the command (seconds)
            period: Update period in seconds
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
        
        # Enable trajectory limitations with reasonable values
        point.LimitationsActive = 1
        point.SynchroType = 0
        
        # Set appropriate limitation values for safe movement
        max_speed = 20.0       # Maximum linear speed
        max_accel = 10.0       # Maximum acceleration
        point.Limitations = Limitation(
            max_speed, max_speed, max_speed,   # Speed parameters
            50.0, 50.0, 50.0,                  # Force parameters 
            max_accel, max_accel, max_accel    # Acceleration parameters
        )
        
        # Send command repeatedly for the specified duration
        end_time = time.time() + duration
        success = False
        
        while time.time() < end_time:
            ret = self.lib.SendBasicTrajectory(point)
            if ret != self.NO_ERROR:
                print(f"SendBasicTrajectory (Cartesian) failed with error code: {ret}")
                break
            elif not success:
                print("Cartesian position command issued successfully. Continuing to send...")
                success = True
            
            # Wait before sending the command again
            time.sleep(period)
        
        if success:
            print("Finished sending Cartesian position commands")

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
        """Get the current Cartesian position of the robot end-effector and actual finger positions."""
        # Get Cartesian position from GetCartesianCommand
        # cartesian_position = CartesianInfo()
        # fingers_command = FingersPosition()
        cartesian_position = CartesianPosition()
        ret = self.lib.GetCartesianPosition(ctypes.byref(cartesian_position))
        if ret != self.NO_ERROR:
            print(f"GetCartesianPosition failed with error code: {ret}")
            return None
        
        cartesian_position = [cartesian_position.X,
                            cartesian_position.Y,
                            cartesian_position.Z, 
                            cartesian_position.ThetaX, 
                            cartesian_position.ThetaY,
                              cartesian_position.ThetaZ,
                              cartesian_position.Fingers.Finger1]
      
        
        # Return cartesian position and actual finger positions
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

    def monitor_position_change(self, target_position, timeout=10.0):
        start_time = time.time()
        initial_pos = self.get_cartesian_position()
        if initial_pos is None:
            return False
        
        print(f"Starting position: {initial_pos}")
        print(f"Target position: {target_position}")
        
        while time.time() - start_time < timeout:
            current_pos = self.get_cartesian_position()
            if current_pos is None:
                continue
            
            distance = sum((c - t)**2 for c, t in zip(current_pos[:3], target_position[:3]))**0.5
            print(f"Current position: {current_pos[:3]}, Distance to target: {distance:.4f}m")
            
            # Check if we've reached the target (within tolerance)
            if distance < 0.02:  # 2cm tolerance
                print("Target position reached!")
                return True
            
            time.sleep(0.5)
        
        print("Failed to reach target position within timeout")
        return False

def main():
    # Create arm interface instance
    arm = KinovaArmInterface()
    
    try:
        # Initialize the connection first
        arm.connect()
        arm.move_home()
        time.sleep(3)
        # Get current cartesian position
        cartesian_pos_data = arm.get_cartesian_position()
        
        if cartesian_pos_data is not None:
            cartesian_pos = cartesian_pos_data
            print("\nCurrent Cartesian Position:")
            print(cartesian_pos)
            # Print actual finger positions using GetAngularPosition
            print("\nDetailed Finger Information:")
            arm.print_finger_info()
            
            # Set the arm to Cartesian control mode
            arm.set_cartesian_control()
            
            # Test sending a specific Cartesian position
            # Values from user: ['0.2518', '0.0191', '0.4252', '1.9647', '0.3992', '0.0389', '0.0000']
            position = (0.2518, 0.0191, 0.4252)  # X, Y, Z in meters
            rotation = (1.9647, 0.3992, 0.0389)  # ThetaX, ThetaY, ThetaZ in radians
            fingers = (0.0, 0.0, 0.0)  # Finger positions
            
            print("\nSending Cartesian position command:")
            print(f"Position (X,Y,Z): {position}")
            print(f"Rotation (ThetaX,ThetaY,ThetaZ): {rotation}")
            
            # Send the Cartesian position command with longer duration
            arm.send_cartesian_position(position, rotation, fingers, duration=10.0)
            
            # Wait for movement to complete
            print("Waiting for movement to complete...")
            time.sleep(5)
            
            # Get the new position after movement
            new_pos = arm.get_cartesian_position()
            if new_pos is not None:
                print("\nNew Cartesian Position after movement:")
                print(new_pos)
        else:
            print("Failed to get cartesian position")
            
    except Exception as e:
        print(f"Error testing cartesian position: {e}")
    finally:
        # Clean up
        arm.close()
            
if __name__ == '__main__':
    main()