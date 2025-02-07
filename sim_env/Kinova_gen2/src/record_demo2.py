#!/usr/bin/env python3

from devices.kinova_arm_interface import KinovaArmInterface
from devices.ps4controller_interface import PS4Interface
import time
import threading
import cv2


class record_demo:
    def __init__(self, debug_mode=False):
        self.velocity_scale = 30.0  # Maximum joint velocity in degrees/second (reduced for safety)
        self.gripper_scale = 3000
        0  # Scale factor for gripper control
        self.running = False
        self.emergency_stop = False
        self.controller = None
        self.arm = None
        self.control_thread = None
        self.debug_mode = debug_mode
        self.cameras = None
        # Removed asynchronous camera thread and shared frame storage as we'll capture synchronously

        #TODO: add camera interface
    def initialize_devices(self):
        """Initialize PS4 controller and Kinova arm"""
        try:
            # Initialize Kinova arm first
            print("Initializing Kinova arm...")
            self.arm = KinovaArmInterface()
            self.arm.connect()
            print("Kinova arm initialized successfully")
            
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
                self.cameras = MultiCameraInterface()
                if not self.cameras.cameras:
                    print("No cameras found!")
                else:
                    print(f"Successfully initialized {len(self.cameras.cameras)} cameras")
            except Exception as e:
                print(f"Camera initialization failed: {e}")
            
            print("\nInitialization complete - ready for velocity control")
            return True
        except Exception as e:
            print(f"Error initializing devices: {e}")
            return False

    def record_demo_loop(self):
        """Main control loop for the robot using velocity control. Captures camera frames synchronously at the start of each iteration."""
        print("\nPS4 Controller Mapping:")
        print("Left Stick X/Y: Joint 1 & 2")
        print("Right Stick X/Y: Joint 3 & 4")
        print("L2/R2 Triggers: Joint 5")
        print("L1/R1 Buttons: Joint 6")
        print("Square/Circle: Gripper Open/Close")
        print("Triangle: Move to Home Position")
        print("Share: Emergency Stop")
        print("Options: Exit")
        
        print("\nStarting velocity control loop...")
        print("Note: The arm should now be in home position and ready for velocity control")
        
        while self.running and not self.emergency_stop:
            try:
                # Synchronously capture frames from all cameras at the start of each iteration
                if self.cameras is not None and self.cameras.cameras:
                    frames = self.cameras.capture_frames()
                    for cam_id, (success, frame) in frames.items():
                        if success and frame is not None:
                            cv2.imshow(f"Camera {cam_id}", frame)
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
                
                # Sleep for the remainder of the control cycle
                time.sleep(0.03333)
                
            except Exception as e:
                print(f"Error in control loop: {e}")
                break

    def start(self):
        """Start the robot controller and initiate synchronous camera capture in the control loop"""
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
        if self.control_thread:
            self.control_thread.join()
        if hasattr(self, 'arm'):
            self.arm.close()
        if self.cameras:
            self.cameras.close()
        cv2.destroyAllWindows()
        print("Robot controller stopped")

def main():
    controller = record_demo(debug_mode=False)  # Set debug_mode=True to enable debug prints
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