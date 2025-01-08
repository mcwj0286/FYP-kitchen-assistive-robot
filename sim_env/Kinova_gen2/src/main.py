#!/usr/bin/env python3

import time
import threading
import json
from datetime import datetime
from controller.ps4_interface import KinovaPS4Controller
from kinova_control.kinova_interface import KinovaArm

class RobotControl:
    def __init__(self):
        self.arm = KinovaArm()
        self.controller = None
        self.running = True
        self.recording_data = []
        
    def start_controller(self):
        """Initialize and start PS4 controller"""
        self.controller = KinovaPS4Controller(interface="/dev/input/js0", connecting_using_ds4drv=True)
        self.controller_thread = threading.Thread(target=self.controller.listen)
        self.controller_thread.start()
        
    def control_loop(self):
        """Main control loop"""
        movement_scale = 0.01  # Scale factor for movement (adjust as needed)
        rotation_scale = 0.1   # Scale factor for rotation
        
        while self.running:
            try:
                if not hasattr(self.controller, 'left_stick_x'):
                    time.sleep(0.1)
                    continue
                
                # Get current position
                current_pos = self.arm.get_current_position()
                
                # Calculate new position based on controller input
                x_move = self.controller.left_stick_x * movement_scale
                y_move = self.controller.left_stick_y * movement_scale
                z_move = self.controller.right_stick_y * movement_scale
                rotation = self.controller.right_stick_x * rotation_scale
                
                # Move arm
                self.arm.move_cartesian(
                    x=current_pos['X'] + x_move,
                    y=current_pos['Y'] + y_move,
                    z=current_pos['Z'] + z_move,
                    theta_z=current_pos['ThetaZ'] + rotation
                )
                
                # Control gripper
                gripper_value = (self.controller.r2_analog - self.controller.l2_analog + 1) / 2
                self.arm.control_gripper(gripper_value)
                
                # Record data if recording is enabled
                if self.controller.is_recording:
                    self.record_data(current_pos, x_move, y_move, z_move, rotation, gripper_value)
                
                time.sleep(0.05)  # 20Hz control loop
                
            except Exception as e:
                print(f"Error in control loop: {e}")
                
    def record_data(self, position, x_move, y_move, z_move, rotation, gripper):
        """Record movement data"""
        data_point = {
            'timestamp': datetime.now().isoformat(),
            'position': position,
            'control_input': {
                'x_move': x_move,
                'y_move': y_move,
                'z_move': z_move,
                'rotation': rotation,
                'gripper': gripper
            }
        }
        self.recording_data.append(data_point)
        
    def save_recording(self):
        """Save recorded data to file"""
        if self.recording_data:
            filename = f"data/recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(self.recording_data, f, indent=2)
            print(f"Recording saved to {filename}")
            self.recording_data = []
            
    def shutdown(self):
        """Clean shutdown of the system"""
        self.running = False
        if self.controller:
            self.save_recording()
        if hasattr(self, 'controller_thread'):
            self.controller_thread.join()
        self.arm.close()

def main():
    robot = RobotControl()
    
    try:
        print("Initializing robot control system...")
        robot.start_controller()
        print("PS4 controller initialized")
        print("Starting control loop...")
        robot.control_loop()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        robot.shutdown()

if __name__ == "__main__":
    main() 