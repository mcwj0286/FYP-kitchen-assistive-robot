#!/usr/bin/env python3

# from .devices.kinova_arm_interface import KinovaArmInterface
# from .devices.ps4controller_interface import PS4Interface
from devices.kinova_arm_interface import KinovaArmInterface
from devices.ps4controller_interface import PS4Interface
import time
import threading
import argparse


class RobotController:
    def __init__(self, debug_mode=False, enable_controller=True, control_mode='joint'):
        self.velocity_scale = 30.0  # Maximum joint velocity in degrees/second (reduced for safety)
        self.gripper_scale = 3000.0 # Scale factor for gripper control
        self.cartesian_linear_scale = 0.20  # Maximum Cartesian linear velocity in m/s
        self.cartesian_angular_scale = 40.0  # Maximum Cartesian angular velocity in degrees/s
        self.running = False
        self.emergency_stop = False
        self.enable_controller = enable_controller
        self.controller = None
        self.arm = None
        self.control_thread = None
        self.debug_mode = debug_mode
        self.control_mode = control_mode  # 'joint' or 'cartesian'

    def initialize_devices(self, move_home=True):
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
            if move_home:
                print("\nMoving to home position before starting control...")
                self.arm.move_home()
                
            # Wait for movement to complete (5 seconds should be enough)
            print("Waiting for home position movement to complete...")
            time.sleep(5)
            
            # Initialize PS4 controller
            if self.enable_controller:
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
            
            print(f"\nInitialization complete - ready for {self.control_mode} control")
            return True
        except Exception as e:
            print(f"Error initializing devices: {e}")
            return False

    def control_loop(self):
        """Main control loop for the robot using joint velocity control"""
        if self.control_mode == 'cartesian':
            self.cartesian_control_loop()
        else:
            self.joint_control_loop()
            
    def joint_control_loop(self):
        """Control loop for joint velocity control mode"""
        print("\nPS4 Controller Mapping (Joint Velocity Mode):")
        print("Left Stick X/Y: Joint 1 & 2")
        print("Right Stick X/Y: Joint 3 & 4")
        print("L2/R2 Triggers: Joint 5")
        print("L1/R1 Buttons: Joint 6")
        print("Square/Circle: Gripper Open/Close")
        print("Triangle: Move to Home Position")
        print("Share: Emergency Stop")
        print("Options: Exit")
        
        print("\nStarting joint velocity control loop...")
        print("Note: The arm should now be in home position and ready for velocity control")
        
        while self.running and not self.emergency_stop:
            try:
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
                joint_velocities[2] = -self.controller.right_stick_y * self.velocity_scale
                joint_velocities[3] = -self.controller.right_stick_x * self.velocity_scale
                
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
                
                # Check for home position request
                if self.controller.triangle_pressed:
                    print("Moving to home position...")
                    self.arm.move_home()
                    # Wait for movement to complete
                    print("Waiting for home position movement to complete...")
                    time.sleep(5)
                    print("Ready for velocity control again")
                    continue
                
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
                
                # Sleep for exactly 5ms for the next control cycle
                time.sleep(0.03333)
                
            except Exception as e:
                print(f"Error in joint control loop: {e}")
                break
    
    def cartesian_control_loop(self):
        """Control loop for Cartesian velocity control mode"""
        print("\nPS4 Controller Mapping (Cartesian Velocity Mode):")
        print("Left Stick X/Y: X/Y translation")
        print("Up/Down Arrows: Z translation (up/down)")
        print("Right Stick X: Rotation around Y (Pitch)")
        print("Right Stick Y: Rotation around X (Roll)")
        print("L1/R1 Buttons: Rotation around Z (Yaw)")
        print("Square/Circle: Gripper Open/Close")
        print("Triangle: Move to Home Position")
        print("Share: Emergency Stop")
        print("Options: Exit")
        
        print("\nStarting Cartesian velocity control loop...")
        print("Note: The arm should now be in home position and ready for Cartesian control")
        
        while self.running and not self.emergency_stop:
            try:
                if not self.controller:
                    print("No controller connected")
                    break
                
                # Map controller inputs to Cartesian velocities
                # Linear velocities (X, Y, Z) in m/s
                linear_velocity = [0.0, 0.0, 0.0]
                linear_velocity[0] = -self.controller.left_stick_x * self.cartesian_linear_scale  # X axis
                linear_velocity[1] = self.controller.left_stick_y * self.cartesian_linear_scale  # Y axis
                
                # Use up/down arrows for Z axis movement
                z_velocity = 0.0
                if self.controller.l2_trigger:
                    z_velocity = self.cartesian_linear_scale
                elif self.controller.r2_trigger:
                    z_velocity = -self.cartesian_linear_scale
              
                linear_velocity[2]=z_velocity
                # Angular velocities (around X, Y, Z) in degrees/s
                angular_velocity = [0.0, 0.0, 0.0]
                
                # Roll (rotation around X axis) - Now using Right Stick Y
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
                
                # Check for home position request
                if self.controller.triangle_pressed:
                    print("Moving to home position...")
                    self.arm.move_home()
                    # Wait for movement to complete
                    print("Waiting for home position movement to complete...")
                    time.sleep(5)
                    print("Ready for Cartesian control again")
                    continue
                
                # Send Cartesian velocity commands to the arm
                self.arm.send_cartesian_velocity(
                    linear_velocity=tuple(linear_velocity),
                    angular_velocity=tuple(angular_velocity),
                    fingers=(gripper_velocity, gripper_velocity, gripper_velocity),
                    duration=0.03333,  # Duration for 30Hz control loop
                    period=0.005,  # 30Hz update rate
                    hand_mode=1  # Add this parameter
                )
                
                # Print velocities for debugging if there's movement
                if any(abs(v) > 0.01 for v in linear_velocity) or any(abs(v) > 0.1 for v in angular_velocity):
                    print(f"Linear Velocity: {[f'{v:.3f}' for v in linear_velocity]} m/s, " +
                          f"Angular Velocity: {[f'{v:.1f}' for v in angular_velocity]} deg/s")
                
                # Sleep for next control cycle
                time.sleep(0.03333)
                
            except Exception as e:
                print(f"Error in Cartesian control loop: {e}")
                break

    def send_action(self, joint_velocities, gripper_velocity):
        joint_velocities *= self.velocity_scale
        gripper_velocity *= self.gripper_scale
        joint_velocities = joint_velocities.tolist()
        self.arm.send_angular_velocity(joint_velocities, hand_mode=1, fingers=(gripper_velocity, gripper_velocity, gripper_velocity), duration=0.03333, period=0.005)
        joint_velocities[6] = gripper_velocity
        print(f"Joint Velocities: {[f'{v:.1f}' for v in joint_velocities]}")


    def start(self):
        """Start the robot controller"""
        if not self.initialize_devices():
            return False
        
        self.running = True
        self.emergency_stop = False
        
        # Start control loop in a separate thread
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.start()
        
        return True

    def stop(self):
        """Stop the robot controller"""
        print("Stopping robot controller...")
        
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

        self.running = False
        self.emergency_stop = True

        if self.control_thread:
            print("Waiting for control thread to finish...")
            self.control_thread.join(timeout=2.0)  # Add timeout to prevent hanging
            
        if hasattr(self, 'arm'):
            print("Closing Kinova arm...")
            try:
                self.arm.close()
                self.arm = None
            except Exception as e:
                print(f"Error closing arm: {e}")
                
        print("Robot controller stopped successfully")

def main():
    # Modify to allow command-line selection of control mode
    parser = argparse.ArgumentParser(description='Robot Controller Application')
    parser.add_argument('--mode', type=str, choices=['joint', 'cartesian'], default='cartesian',
                        help='Control mode (joint or cartesian)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    controller = RobotController(debug_mode=args.debug, control_mode=args.mode)
    
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