#!/usr/bin/env python3

from pyPS4Controller.controller import Controller
import threading
import time

class KinovaPS4Controller(Controller):
    def __init__(self, **kwargs):
        Controller.__init__(self, **kwargs)
        self.reset_values()
        
    def reset_values(self):
        """Reset all controller values to default"""
        self.left_stick_x = 0.0
        self.left_stick_y = 0.0
        self.right_stick_x = 0.0
        self.right_stick_y = 0.0
        self.l2_analog = 0.0
        self.r2_analog = 0.0
        self.is_recording = False

    # Left stick (X-Y translation)
    def on_L3_up(self, value):
        self.left_stick_y = value / 32767.0
        
    def on_L3_down(self, value):
        self.left_stick_y = value / 32767.0
        
    def on_L3_left(self, value):
        self.left_stick_x = value / 32767.0
        
    def on_L3_right(self, value):
        self.left_stick_x = value / 32767.0
        
    # Right stick (Z translation and rotation)
    def on_R3_up(self, value):
        self.right_stick_y = value / 32767.0
        
    def on_R3_down(self, value):
        self.right_stick_y = value / 32767.0
        
    def on_R3_left(self, value):
        self.right_stick_x = value / 32767.0
        
    def on_R3_right(self, value):
        self.right_stick_x = value / 32767.0
        
    # Triggers (Gripper control)
    def on_L2_press(self, value):
        self.l2_analog = value / 32767.0
        
    def on_R2_press(self, value):
        self.r2_analog = value / 32767.0

    # Recording control
    def on_circle_press(self):
        self.is_recording = not self.is_recording
        print(f"Recording: {'Started' if self.is_recording else 'Stopped'}")

    # Emergency stop
    def on_options_press(self):
        print("EMERGENCY STOP TRIGGERED")
        self.reset_values()
        # Add emergency stop logic here

def main():
    controller = KinovaPS4Controller(interface="/dev/input/js0", connecting_using_ds4drv=True)
    controller.listen(timeout=60)

if __name__ == "__main__":
    main() 