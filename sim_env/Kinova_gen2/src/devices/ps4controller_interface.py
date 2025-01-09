#!/usr/bin/env python3

from pyPS4Controller.controller import Controller
# check connection for 'ls /dev/input/js*' , it should be something like 'js0'
class PS4Interface(Controller):
    def __init__(self, **kwargs):
        Controller.__init__(self, **kwargs)
        # Initialize controller state values
        self.reset_values()
        # Define deadzone threshold (adjust this value as needed)
        self.deadzone = 2000  # Typical values between 1000-3000
        print("PS4 Controller initialized")

    def reset_values(self):
        """Reset all controller values to default"""
        # Joystick values (-1.0 to 1.0)
        self.left_stick_x = 0.0
        self.left_stick_y = 0.0
        self.right_stick_x = 0.0
        self.right_stick_y = 0.0
        
        # Trigger values (0.0 to 1.0)
        self.l2_trigger = 0.0
        self.r2_trigger = 0.0
        
        # Button states
        self.l1_pressed = False
        self.r1_pressed = False
        self.square_pressed = False
        self.circle_pressed = False
        self.triangle_pressed = False
        self.x_pressed = False

    def apply_deadzone(self, value):
        """Apply deadzone to analog inputs
        
        Args:
            value (int): Raw analog value from controller
            
        Returns:
            float: Processed value with deadzone applied, normalized to -1.0 to 1.0
        """
        if abs(value) < self.deadzone:
            return 0.0
        return value / 32767.0

    # Stick Callbacks with deadzone
    def on_L3_up(self, value):
        self.left_stick_y = self.apply_deadzone(value)
        if abs(self.left_stick_y) > 0:  # Only print if outside deadzone
            print(f"Left stick Y: {self.left_stick_y:.2f}")
    
    def on_L3_down(self, value):
        self.left_stick_y = self.apply_deadzone(value)
        if abs(self.left_stick_y) > 0:
            print(f"Left stick Y: {self.left_stick_y:.2f}")
    
    def on_L3_left(self, value):
        self.left_stick_x = self.apply_deadzone(value)
        if abs(self.left_stick_x) > 0:
            print(f"Left stick X: {self.left_stick_x:.2f}")
    
    def on_L3_right(self, value):
        self.left_stick_x = self.apply_deadzone(value)
        if abs(self.left_stick_x) > 0:
            print(f"Left stick X: {self.left_stick_x:.2f}")
    
    def on_R3_up(self, value):
        self.right_stick_y = self.apply_deadzone(value)
        if abs(self.right_stick_y) > 0:
            print(f"Right stick Y: {self.right_stick_y:.2f}")
    
    def on_R3_down(self, value):
        self.right_stick_y = self.apply_deadzone(value)
        if abs(self.right_stick_y) > 0:
            print(f"Right stick Y: {self.right_stick_y:.2f}")
    
    def on_R3_left(self, value):
        self.right_stick_x = self.apply_deadzone(value)
        if abs(self.right_stick_x) > 0:
            print(f"Right stick X: {self.right_stick_x:.2f}")
    
    def on_R3_right(self, value):
        self.right_stick_x = self.apply_deadzone(value)
        if abs(self.right_stick_x) > 0:
            print(f"Right stick X: {self.right_stick_x:.2f}")

    # Button Callbacks
    def on_L1_press(self):
        self.l1_pressed = True
        print("L1 button pressed")
    
    def on_L1_release(self):
        self.l1_pressed = False
        print("L1 button released")
    
    def on_L2_press(self, value):
        self.l2_trigger = value / 32767.0
        print(f"L2 trigger: {self.l2_trigger:.2f}")
    
    def on_L2_release(self):
        self.l2_trigger = 0.0
        print("L2 trigger released")
    
    def on_R1_press(self):
        self.r1_pressed = True
        print("R1 button pressed")
    
    def on_R1_release(self):
        self.r1_pressed = False
        print("R1 button released")
    
    def on_R2_press(self, value):
        self.r2_trigger = value / 32767.0
        print(f"R2 trigger: {self.r2_trigger:.2f}")
    
    def on_R2_release(self):
        self.r2_trigger = 0.0
        print("R2 trigger released")

    # Face Buttons
    def on_x_press(self):
        self.x_pressed = True
        print("X button pressed")
    
    def on_x_release(self):
        self.x_pressed = False
        print("X button released")
    
    def on_triangle_press(self):
        self.triangle_pressed = True
        print("Triangle button pressed")
    
    def on_triangle_release(self):
        self.triangle_pressed = False
        print("Triangle button released")
    
    def on_square_press(self):
        self.square_pressed = True
        print("Square button pressed")
    
    def on_square_release(self):
        self.square_pressed = False
        print("Square button released")
    
    def on_circle_press(self):
        self.circle_pressed = True
        print("Circle button pressed")
    
    def on_circle_release(self):
        self.circle_pressed = False
        print("Circle button released")

    # Special Buttons
    def on_options_press(self):
        print("Options button pressed")
    
    def on_options_release(self):
        print("Options button released")
    
    def on_share_press(self):
        print("Share button pressed (Emergency Stop)")
    
    def on_share_release(self):
        print("Share button released")


def main():
    # Initialize the PS4 controller
    # Note: The interface path might need to be adjusted based on your system
    # Common paths are:
    # - /dev/input/js0
    # - /dev/input/js1
    # You can check available devices using: ls /dev/input/js*
    
    try:
        controller = PS4Interface(interface="/dev/input/js0", connecting_using_ds4drv=False)
        print("PS4 Controller initialized. Press buttons to see output.")
        print("Press Ctrl+C to exit.")
        
        # Start listening for controller inputs
        controller.listen(timeout=60)  # 60 second timeout, adjust as needed
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your PS4 controller is connected and the correct interface path is specified.")
        print("You can check available devices using: ls /dev/input/js*")

if __name__ == "__main__":
    main() 